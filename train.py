"""Training entrypoint
W&B is commented out for now (validation run). Re-enable for final full training.

Best-model selection uses the SAME metric Gradescope grades on:
  classification  -> val Macro-F1          (higher = better)
  localization    -> val Acc@IoU>=0.5      (higher = better)
  segmentation    -> val Macro-Dice        (higher = better)
  multitask       -> mean(F1, Acc@IoU0.5, Dice)  (higher = better)
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier, VGG11Localizer, VGG11UNet, MultiTaskPerceptionModel
from losses import IoULoss



def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality  = (probs + targets_oh).sum(dim=dims)
        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()



def compute_macro_f1(logits: torch.Tensor, targets: torch.Tensor,
                     num_classes: int = 37) -> float:
    """Macro-averaged F1 — same formula Gradescope uses."""
    preds = logits.argmax(dim=1).cpu().numpy()
    tgts  = targets.cpu().numpy()
    f1_sum = 0.0
    for c in range(num_classes):
        tp = ((preds == c) & (tgts == c)).sum()
        fp = ((preds == c) & (tgts != c)).sum()
        fn = ((preds != c) & (tgts == c)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1_sum += 2 * prec * rec / (prec + rec + 1e-8)
    return float(f1_sum / num_classes)


def compute_acc_at_iou(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                       threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Fraction of predictions whose IoU with GT >= threshold.
    Gradescope uses this for 4.2a (thr=0.5) and 4.2b (thr=0.75).
    """
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    tgt_x1  = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1  = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2  = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2  = target_boxes[:, 1] + target_boxes[:, 3] / 2

    ix1 = torch.max(pred_x1, tgt_x1); iy1 = torch.max(pred_y1, tgt_y1)
    ix2 = torch.min(pred_x2, tgt_x2); iy2 = torch.min(pred_y2, tgt_y2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa  = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    ta  = (tgt_x2  - tgt_x1).clamp(0) * (tgt_y2  - tgt_y1).clamp(0)
    iou = (inter + eps) / (pa + ta - inter + eps)
    return (iou >= threshold).float().mean().item()


def compute_macro_dice(logits: torch.Tensor, targets: torch.Tensor,
                       num_classes: int = 3, eps: float = 1e-6) -> float:
    """Macro-averaged Dice — same formula Gradescope uses."""
    preds = logits.argmax(dim=1)
    dice_sum = 0.0
    for c in range(num_classes):
        p = (preds == c).float()
        t = (targets == c).float()
        dice_sum += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return float(dice_sum / num_classes)


def compute_iou_mean(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                     eps: float = 1e-6) -> float:
    """Mean IoU — used for console logging only."""
    pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
    pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2
    tgt_x1  = target_boxes[:, 0] - target_boxes[:, 2] / 2
    tgt_y1  = target_boxes[:, 1] - target_boxes[:, 3] / 2
    tgt_x2  = target_boxes[:, 0] + target_boxes[:, 2] / 2
    tgt_y2  = target_boxes[:, 1] + target_boxes[:, 3] / 2
    ix1 = torch.max(pred_x1, tgt_x1); iy1 = torch.max(pred_y1, tgt_y1)
    ix2 = torch.min(pred_x2, tgt_x2); iy2 = torch.min(pred_y2, tgt_y2)
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    pa  = (pred_x2 - pred_x1).clamp(0) * (pred_y2 - pred_y1).clamp(0)
    ta  = (tgt_x2  - tgt_x1).clamp(0) * (tgt_y2  - tgt_y1).clamp(0)
    return ((inter + eps) / (pa + ta - inter + eps)).mean().item()


def compute_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == targets).float().mean().item()



def primary_metric(metrics: dict, task: str) -> float:
    """Return the single scalar used to decide whether to save best.pth.

    Mapping mirrors Gradescope:
      classification -> val_f1          (Macro-F1,       higher=better)
      localization   -> val_acc_iou05   (Acc@IoU>=0.5,   higher=better)
      segmentation   -> val_dice        (Macro-Dice,     higher=better)
      multitask      -> mean of all three above
    """
    if task == "classification":
        return metrics.get("f1", 0.0)
    elif task == "localization":
        return metrics.get("acc_iou05", 0.0)
    elif task == "segmentation":
        return metrics.get("dice", 0.0)
    elif task == "multitask":
        f1       = metrics.get("f1",        0.0)
        acc_iou  = metrics.get("acc_iou05", 0.0)
        dice     = metrics.get("dice",      0.0)
        return (f1 + acc_iou + dice) / 3.0
    return 0.0




def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] Saved -> {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    if not os.path.isfile(path):
        print(f"  [ckpt] No checkpoint at '{path}' — starting from scratch.")
        return 0, -1.0   # best_metric starts at -1 (we maximise)
    ckpt = torch.load(path, map_location="cpu")
    
    model.load_state_dict(ckpt.get("state_dict", ckpt.get("model")))
    
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch       = ckpt["epoch"]
    best_metric = ckpt.get("best_metric", -1.0)
    print(f"  [ckpt] Resumed from epoch {epoch}  "
          f"(best_metric={best_metric:.4f})")
    return epoch, best_metric




def train_one_epoch(model, loader, optimizer, criterion, device, task):
    model.train()
    total_loss = 0.0
    # accumulators
    all_logits_cls, all_labels     = [], []
    all_pred_boxes, all_tgt_boxes  = [], []
    all_logits_seg, all_masks      = [], []
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()

        if task == "classification":
            logits = model(images)
            loss   = criterion(logits, labels)
            all_logits_cls.append(logits.detach().cpu())
            all_labels.append(labels.cpu())

        elif task == "localization":
            preds = model(images)
            loss  = criterion(preds, bboxes)
            all_pred_boxes.append(preds.detach().cpu())
            all_tgt_boxes.append(bboxes.cpu())

        elif task == "segmentation":
            logits = model(images)
            loss   = criterion(logits, masks)
            all_logits_seg.append(logits.detach().cpu())
            all_masks.append(masks.cpu())

        elif task == "multitask":
            out = model(images)
            loss = (criterion["cls"](out["classification"], labels) +
                    criterion["loc"](out["localization"],   bboxes) +
                    criterion["seg"](out["segmentation"],   masks))
            all_logits_cls.append(out["classification"].detach().cpu())
            all_labels.append(labels.cpu())
            all_pred_boxes.append(out["localization"].detach().cpu())
            all_tgt_boxes.append(bboxes.cpu())
            all_logits_seg.append(out["segmentation"].detach().cpu())
            all_masks.append(masks.cpu())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    return _compute_metrics(total_loss, n_batches,
                            all_logits_cls, all_labels,
                            all_pred_boxes, all_tgt_boxes,
                            all_logits_seg, all_masks)


@torch.no_grad()
def validate(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    all_logits_cls, all_labels     = [], []
    all_pred_boxes, all_tgt_boxes  = [], []
    all_logits_seg, all_masks      = [], []
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        if task == "classification":
            logits = model(images)
            loss   = criterion(logits, labels)
            all_logits_cls.append(logits.cpu())
            all_labels.append(labels.cpu())

        elif task == "localization":
            preds = model(images)
            loss  = criterion(preds, bboxes)
            all_pred_boxes.append(preds.cpu())
            all_tgt_boxes.append(bboxes.cpu())

        elif task == "segmentation":
            logits = model(images)
            loss   = criterion(logits, masks)
            all_logits_seg.append(logits.cpu())
            all_masks.append(masks.cpu())

        elif task == "multitask":
            out = model(images)
            loss = (criterion["cls"](out["classification"], labels) +
                    criterion["loc"](out["localization"],   bboxes) +
                    criterion["seg"](out["segmentation"],   masks))
            all_logits_cls.append(out["classification"].cpu())
            all_labels.append(labels.cpu())
            all_pred_boxes.append(out["localization"].cpu())
            all_tgt_boxes.append(bboxes.cpu())
            all_logits_seg.append(out["segmentation"].cpu())
            all_masks.append(masks.cpu())

        total_loss += loss.item()
        n_batches  += 1

    return _compute_metrics(total_loss, n_batches,
                            all_logits_cls, all_labels,
                            all_pred_boxes, all_tgt_boxes,
                            all_logits_seg, all_masks)


def _compute_metrics(total_loss, n_batches,
                     all_logits_cls, all_labels,
                     all_pred_boxes, all_tgt_boxes,
                     all_logits_seg, all_masks):
    """Compute all metrics from epoch-level accumulated tensors."""
    m = {"loss": total_loss / max(n_batches, 1)}

    if all_logits_cls:
        logits  = torch.cat(all_logits_cls)
        labels  = torch.cat(all_labels)
        correct = (logits.argmax(1) == labels).sum().item()
        m["acc"] = correct / len(labels)
        m["f1"]  = compute_macro_f1(logits, labels, num_classes=logits.shape[1])

    if all_pred_boxes:
        preds   = torch.cat(all_pred_boxes)
        targets = torch.cat(all_tgt_boxes)
        m["iou"]        = compute_iou_mean(preds, targets)
        m["acc_iou05"]  = compute_acc_at_iou(preds, targets, threshold=0.5)
        m["acc_iou075"] = compute_acc_at_iou(preds, targets, threshold=0.75)

    if all_logits_seg:
        logits  = torch.cat(all_logits_seg)
        masks   = torch.cat(all_masks)
        m["dice"]      = compute_macro_dice(logits, masks, num_classes=logits.shape[1])
        m["pixel_acc"] = compute_pixel_accuracy(logits, masks)

    return m




def build_model(task: str, cfg: dict, device):
    if task == "classification":
        return VGG11Classifier(
            num_classes=cfg["num_classes"],
            in_channels=3,
            dropout_p=cfg["dropout_p"],
        ).to(device)
    elif task == "localization":
        return VGG11Localizer(in_channels=3, dropout_p=cfg["dropout_p"]).to(device)
    elif task == "segmentation":
        return VGG11UNet(
            num_classes=cfg["seg_classes"],
            in_channels=3,
            dropout_p=cfg["dropout_p"],
        ).to(device)
    elif task == "multitask":
        return MultiTaskPerceptionModel(
            num_breeds=cfg["num_classes"],
            seg_classes=cfg["seg_classes"],
            in_channels=3,
            classifier_path=cfg.get("classifier_path", ""),
            localizer_path=cfg.get("localizer_path",  ""),
            unet_path=cfg.get("unet_path",            ""),
            dropout_p=cfg["dropout_p"],
        ).to(device)
    raise ValueError(f"Unknown task: {task}")


def build_criterion(task: str, device, seg_classes: int = 3):
    if task == "classification":
        return nn.CrossEntropyLoss()

    elif task == "localization":
        iou_loss = IoULoss(reduction="mean")
        mse_loss = nn.MSELoss()
        class CombinedLoc(nn.Module):
            def forward(self, preds, targets):
                return mse_loss(preds, targets) / (224.0 ** 2) + iou_loss(preds, targets)
        return CombinedLoc()

    elif task == "segmentation":
        dice = DiceLoss(num_classes=seg_classes)
        ce   = nn.CrossEntropyLoss()
        class CombinedSeg(nn.Module):
            def forward(self, logits, masks):
                return ce(logits, masks) + dice(logits, masks)
        return CombinedSeg()

    elif task == "multitask":
        dice   = DiceLoss(num_classes=seg_classes)
        ce_seg = nn.CrossEntropyLoss()
        iou_fn = IoULoss(reduction="mean")
        mse_fn = nn.MSELoss()
        class CombinedSegMT(nn.Module):
            def forward(self, logits, masks):
                return ce_seg(logits, masks) + dice(logits, masks)
        class CombinedLocMT(nn.Module):
            def forward(self, preds, targets):
                return mse_fn(preds, targets) / (224.0 ** 2) + iou_fn(preds, targets)
        return {
            "cls": nn.CrossEntropyLoss(),
            "loc": CombinedLocMT(),
            "seg": CombinedSegMT(),
        }

    raise ValueError(f"Unknown task: {task}")




def main():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 – Training")
    parser.add_argument("--task",        type=str,   default="classification",
                        choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--data_root",   type=str,
                        default="/kaggle/working/data/oxford-iiit-pet")
    parser.add_argument("--ckpt_dir",    type=str,
                        default="/kaggle/working/checkpoints")
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout_p",   type=float, default=0.5)
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=2)
    # W&B args — kept for compatibility, used when you uncomment wandb blocks
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    # Paths to individual task checkpoints (only needed for multitask)
    parser.add_argument("--classifier_path", type=str, default="")
    parser.add_argument("--localizer_path",  type=str, default="")
    parser.add_argument("--unet_path",       type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    primary_metric_name = {
        "classification": "Macro-F1",
        "localization":   "Acc@IoU≥0.5",
        "segmentation":   "Macro-Dice",
        "multitask":      "mean(F1+Acc@IoU0.5+Dice)",
    }[args.task]

    print("=" * 60)
    print(f"  Task         : {args.task}")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch        : {args.batch_size}  |  LR: {args.lr}")
    print(f"  Best metric  : {primary_metric_name}  (higher = better)")
    print(f"  Data root    : {args.data_root}")
    print(f"  Ckpt dir     : {args.ckpt_dir}/{args.task}/")
    print("=" * 60)

    cfg = {
        "task":            args.task,
        "epochs":          args.epochs,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "dropout_p":       args.dropout_p,
        "image_size":      args.image_size,
        "num_classes":     37,
        "seg_classes":     3,
        "seed":            args.seed,
        "classifier_path": args.classifier_path,
        "localizer_path":  args.localizer_path,
        "unet_path":       args.unet_path,
    }


    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=cfg,
        name=f"{args.task}_ep{args.epochs}_lr{args.lr}",
    )


    train_ds = OxfordIIITPetDataset(
        args.data_root, split="train",
        image_size=args.image_size, seed=args.seed,
    )
    val_ds = OxfordIIITPetDataset(
        args.data_root, split="val",
        image_size=args.image_size, seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"\nTrain samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}\n")


    model     = build_model(args.task, cfg, device)
    criterion = build_criterion(args.task, device, seg_classes=cfg["seg_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )


    ckpt_dir  = Path(args.ckpt_dir) / args.task
    best_path = str(ckpt_dir / "best.pth")
    last_path = str(ckpt_dir / "last.pth")


    start_epoch, best_metric = load_checkpoint(
        last_path, model, optimizer, scheduler,
    )

    if start_epoch >= args.epochs:
        print(f"\nAlready completed {start_epoch}/{args.epochs} epochs. "
              f"Increase --epochs to continue training.")
        return


    for epoch in range(start_epoch, args.epochs):
        print(f"\n── Epoch [{epoch+1}/{args.epochs}] " + "─" * 40)

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.task,
        )
        val_m = validate(
            model, val_loader, criterion, device, args.task,
        )
        scheduler.step()


        def fmt(m, prefix=""):
            parts = [f"loss={m['loss']:.4f}"]
            if "acc"        in m: parts.append(f"acc={m['acc']:.3f}")
            if "f1"         in m: parts.append(f"F1={m['f1']:.4f}")
            if "acc_iou05"  in m: parts.append(f"Acc@IoU0.5={m['acc_iou05']:.4f}")
            if "acc_iou075" in m: parts.append(f"Acc@IoU0.75={m['acc_iou075']:.4f}")
            if "dice"       in m: parts.append(f"Dice={m['dice']:.4f}")
            if "pixel_acc"  in m: parts.append(f"pix={m['pixel_acc']:.3f}")
            return "  " + prefix + "  " + "  ".join(parts)

        print(fmt(train_m, "TRAIN"))
        print(fmt(val_m,   "VAL  "))

        cur_metric = primary_metric(val_m, args.task)
        print(f"  [{primary_metric_name}] = {cur_metric:.4f}  "
              f"(best so far = {max(best_metric, 0):.4f})")

        log_dict = {f"train/{k}": v for k, v in train_m.items()}
        log_dict.update({f"val/{k}": v for k, v in val_m.items()})
        log_dict["epoch"]                    = epoch + 1
        log_dict["lr"]                       = scheduler.get_last_lr()[0]
        log_dict[f"val/{primary_metric_name}"] = cur_metric
        wandb.log(log_dict)


        save_checkpoint({
            "epoch":        epoch + 1,
            "state_dict":   model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "val_loss":     val_m["loss"],
            "best_metric":  best_metric,
        }, last_path)


        if cur_metric > best_metric:
            best_metric = cur_metric
            save_checkpoint({
                "epoch":        epoch + 1,
                "state_dict":   model.state_dict(),
                "val_loss":     val_m["loss"],
                "best_metric":  best_metric,
            }, best_path)
            print(f"  ✓ New best model saved  "
                  f"({primary_metric_name} = {best_metric:.4f})")
            wandb.run.summary[primary_metric_name] = best_metric 

    wandb.finish()   
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  last checkpoint : {last_path}")
    print(f"  best checkpoint : {best_path}")
    print(f"  best {primary_metric_name} : {best_metric:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
