"""Training entrypoint
W&B is commented out for now (validation run). Re-enable for final full training.
"""

import os
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# import wandb   # <-- uncomment when you want W&B logging

from data.pets_dataset import OxfordIIITPetDataset
from models import VGG11Classifier, VGG11Localizer, VGG11UNet, MultiTaskPerceptionModel
from losses import IoULoss


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Dice loss ─────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss for multi-class segmentation."""

    def __init__(self, num_classes: int = 3, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)             # [B,C,H,W]
        targets_oh = torch.zeros_like(probs)
        targets_oh.scatter_(1, targets.unsqueeze(1), 1)  # one-hot [B,C,H,W]
        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        cardinality  = (probs + targets_oh).sum(dim=dims)
        dice = (2 * intersection + self.eps) / (cardinality + self.eps)
        return 1.0 - dice.mean()


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_iou_metric(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                       eps: float = 1e-6) -> float:
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
    return iou.mean().item()


def compute_dice(logits: torch.Tensor, targets: torch.Tensor,
                 num_classes: int = 3, eps: float = 1e-6) -> float:
    preds = logits.argmax(dim=1)
    dice_sum = 0.0
    for c in range(num_classes):
        p = (preds == c).float(); t = (targets == c).float()
        dice_sum += (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)
    return (dice_sum / num_classes).item()


def compute_pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    print(f"  [ckpt] Saved -> {path}")


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None):
    if not os.path.isfile(path):
        print(f"  [ckpt] No checkpoint at '{path}' — starting from scratch.")
        return 0, float("inf")
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    epoch     = ckpt["epoch"]
    val_loss  = ckpt.get("val_loss", float("inf"))
    best_loss = ckpt.get("best_val_loss", float("inf"))
    print(f"  [ckpt] Resumed from epoch {epoch}  "
          f"(val_loss={val_loss:.4f}, best={best_loss:.4f})")
    return epoch, best_loss


# ── Training loops ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, task):
    model.train()
    total_loss = 0.0
    correct = 0; total = 0
    iou_sum = 0.0; dice_sum = 0.0; pix_sum = 0.0; n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()

        if task == "classification":
            logits = model(images)
            loss   = criterion(logits, labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        elif task == "localization":
            preds = model(images)
            loss  = criterion(preds, bboxes)
            iou_sum += compute_iou_metric(preds.detach(), bboxes)

        elif task == "segmentation":
            logits = model(images)
            loss   = criterion(logits, masks)
            dice_sum += compute_dice(logits.detach(), masks)
            pix_sum  += compute_pixel_accuracy(logits.detach(), masks)

        elif task == "multitask":
            out = model(images)
            loss_cls = criterion["cls"](out["classification"], labels)
            loss_loc = criterion["loc"](out["localization"],   bboxes)
            loss_seg = criterion["seg"](out["segmentation"],   masks)
            loss = loss_cls + loss_loc + loss_seg
            correct  += (out["classification"].argmax(1) == labels).sum().item()
            total    += labels.size(0)
            iou_sum  += compute_iou_metric(out["localization"].detach(), bboxes)
            dice_sum += compute_dice(out["segmentation"].detach(), masks)
            pix_sum  += compute_pixel_accuracy(out["segmentation"].detach(), masks)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches  += 1

    metrics = {"loss": total_loss / n_batches}
    if total    > 0: metrics["acc"]       = correct   / total
    if iou_sum  > 0: metrics["iou"]       = iou_sum   / n_batches
    if dice_sum > 0: metrics["dice"]      = dice_sum  / n_batches
    if pix_sum  > 0: metrics["pixel_acc"] = pix_sum   / n_batches
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, device, task):
    model.eval()
    total_loss = 0.0
    correct = 0; total = 0
    iou_sum = 0.0; dice_sum = 0.0; pix_sum = 0.0; n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        bboxes = batch["bbox"].to(device)
        masks  = batch["mask"].to(device)

        if task == "classification":
            logits = model(images)
            loss   = criterion(logits, labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)

        elif task == "localization":
            preds = model(images)
            loss  = criterion(preds, bboxes)
            iou_sum += compute_iou_metric(preds, bboxes)

        elif task == "segmentation":
            logits = model(images)
            loss   = criterion(logits, masks)
            dice_sum += compute_dice(logits, masks)
            pix_sum  += compute_pixel_accuracy(logits, masks)

        elif task == "multitask":
            out = model(images)
            loss_cls = criterion["cls"](out["classification"], labels)
            loss_loc = criterion["loc"](out["localization"],   bboxes)
            loss_seg = criterion["seg"](out["segmentation"],   masks)
            loss = loss_cls + loss_loc + loss_seg
            correct  += (out["classification"].argmax(1) == labels).sum().item()
            total    += labels.size(0)
            iou_sum  += compute_iou_metric(out["localization"], bboxes)
            dice_sum += compute_dice(out["segmentation"], masks)
            pix_sum  += compute_pixel_accuracy(out["segmentation"], masks)

        total_loss += loss.item()
        n_batches  += 1

    metrics = {"loss": total_loss / n_batches}
    if total    > 0: metrics["acc"]       = correct   / total
    if iou_sum  > 0: metrics["iou"]       = iou_sum   / n_batches
    if dice_sum > 0: metrics["dice"]      = dice_sum  / n_batches
    if pix_sum  > 0: metrics["pixel_acc"] = pix_sum   / n_batches
    return metrics


# ── Model / criterion builders ────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 – Training")
    parser.add_argument("--task",        type=str,   default="classification",
                        choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--data_root",   type=str,
                        default="/content/drive/MyDrive/dl_assignment/oxford-iiit-pet")
    parser.add_argument("--ckpt_dir",    type=str,
                        default="/content/drive/MyDrive/dl_assignment/checkpoints")
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--dropout_p",   type=float, default=0.5)
    parser.add_argument("--image_size",  type=int,   default=224)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--num_workers", type=int,   default=2)
    # W&B args kept so the script stays compatible when you re-enable wandb
    parser.add_argument("--wandb_project", type=str, default="da6401-assignment2")
    parser.add_argument("--wandb_entity",  type=str, default=None)
    # Individual task checkpoint paths (only needed for multitask)
    parser.add_argument("--classifier_path", type=str, default="")
    parser.add_argument("--localizer_path",  type=str, default="")
    parser.add_argument("--unet_path",       type=str, default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Task      : {args.task}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch_size}  |  LR: {args.lr}")
    print(f"  Data root : {args.data_root}")
    print(f"  Ckpt dir  : {args.ckpt_dir}/{args.task}/")
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

    # ── W&B init — uncomment this block when you want W&B logging ─────────────
    # wandb.init(
    #     project=args.wandb_project,
    #     entity=args.wandb_entity,
    #     config=cfg,
    #     name=f"{args.task}_ep{args.epochs}_lr{args.lr}",
    # )

    # ── Datasets ──────────────────────────────────────────────────────────────
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

    # ── Model, criterion, optimiser ───────────────────────────────────────────
    model     = build_model(args.task, cfg, device)
    criterion = build_criterion(args.task, device, seg_classes=cfg["seg_classes"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    # ── Checkpoint paths ──────────────────────────────────────────────────────
    ckpt_dir  = Path(args.ckpt_dir) / args.task
    best_path = str(ckpt_dir / "best.pth")
    last_path = str(ckpt_dir / "last.pth")

    # Resume automatically if last.pth exists (picks up from where it stopped)
    start_epoch, best_val_loss = load_checkpoint(
        last_path, model, optimizer, scheduler,
    )

    if start_epoch >= args.epochs:
        print(f"\nAlready completed {start_epoch}/{args.epochs} epochs. "
              f"Increase --epochs to continue training.")
        return

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        print(f"\n── Epoch [{epoch+1}/{args.epochs}] " + "─" * 40)

        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.task,
        )
        val_m = validate(
            model, val_loader, criterion, device, args.task,
        )
        scheduler.step()

        # Console summary
        def fmt(m):
            s = f"loss={m['loss']:.4f}"
            if "acc"       in m: s += f"  acc={m['acc']:.3f}"
            if "iou"       in m: s += f"  iou={m['iou']:.3f}"
            if "dice"      in m: s += f"  dice={m['dice']:.3f}"
            if "pixel_acc" in m: s += f"  pix={m['pixel_acc']:.3f}"
            return s

        print(f"  TRAIN  {fmt(train_m)}")
        print(f"  VAL    {fmt(val_m)}")

        # ── W&B logging — uncomment this block when you want W&B logging ──────
        # log_dict = {f"train/{k}": v for k, v in train_m.items()}
        # log_dict.update({f"val/{k}": v for k, v in val_m.items()})
        # log_dict["epoch"] = epoch + 1
        # log_dict["lr"]    = scheduler.get_last_lr()[0]
        # wandb.log(log_dict)

        # Save last checkpoint — always, so we can resume
        save_checkpoint({
            "epoch":         epoch + 1,
            "model":         model.state_dict(),
            "optimizer":     optimizer.state_dict(),
            "scheduler":     scheduler.state_dict(),
            "val_loss":      val_m["loss"],
            "best_val_loss": best_val_loss,
        }, last_path)

        # Save best checkpoint
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            save_checkpoint({
                "epoch":    epoch + 1,
                "model":    model.state_dict(),
                "val_loss": best_val_loss,
            }, best_path)
            print(f"  ✓ New best model saved  (val_loss={best_val_loss:.4f})")
            # wandb.run.summary["best_val_loss"] = best_val_loss  # uncomment with W&B

    # wandb.finish()   # uncomment with W&B
    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  last checkpoint : {last_path}")
    print(f"  best checkpoint : {best_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
