"""Inference and evaluation
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import MultiTaskPerceptionModel, VGG11Classifier, VGG11Localizer, VGG11UNet


def get_inference_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


@torch.no_grad()
def run_inference(model, image_path: str, device, image_size: int = 224, task: str = "multitask"):
    transforms = get_inference_transforms(image_size)
    image = np.array(Image.open(image_path).convert("RGB"))
    tensor = transforms(image=image)["image"].unsqueeze(0).to(device)

    model.eval()
    if task == "multitask":
        out = model(tensor)
        return {
            "classification": out["classification"].softmax(dim=1).cpu(),
            "localization":   out["localization"].cpu(),
            "segmentation":   out["segmentation"].argmax(dim=1).cpu(),
        }
    elif task == "classification":
        logits = model(tensor)
        return {"classification": logits.softmax(dim=1).cpu()}
    elif task == "localization":
        bbox = model(tensor)
        return {"localization": bbox.cpu()}
    elif task == "segmentation":
        logits = model(tensor)
        return {"segmentation": logits.argmax(dim=1).cpu()}


def main():
    parser = argparse.ArgumentParser(description="DA6401 Assignment 2 – Inference")
    parser.add_argument("--task",       type=str, default="multitask",
                        choices=["classification", "localization", "segmentation", "multitask"])
    parser.add_argument("--image",      type=str, required=True, help="Path to input image")
    parser.add_argument("--ckpt",       type=str, required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_classes",type=int, default=37)
    parser.add_argument("--seg_classes",type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.task == "multitask":
        model = MultiTaskPerceptionModel(num_breeds=args.num_classes, seg_classes=args.seg_classes).to(device)
    elif args.task == "classification":
        model = VGG11Classifier(num_classes=args.num_classes).to(device)
    elif args.task == "localization":
        model = VGG11Localizer().to(device)
    elif args.task == "segmentation":
        model = VGG11UNet(num_classes=args.seg_classes).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from {args.ckpt}")

    results = run_inference(model, args.image, device, args.image_size, args.task)
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()