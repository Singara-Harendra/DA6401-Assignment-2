"""Dataset skeleton for Oxford-IIIT Pet.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---- Shared augmentation pipelines ----
def get_train_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",          # x_min, y_min, width, height
            label_fields=["bbox_labels"],
            min_visibility=0.3,
        ),
    )


def get_val_transforms(image_size: int = 224) -> A.Compose:
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["bbox_labels"],
            min_visibility=0.3,
        ),
    )


class OxfordIIITPetDataset(Dataset):
    """Oxford-IIIT Pet multi-task dataset loader.

    Provides:
        - Image tensor [3, H, W]
        - Class label (0-36)
        - Bounding box [4] in (x_center, y_center, width, height) pixel format
        - Segmentation mask [H, W] with values {1=fg, 2=bg, 3=boundary} -> remapped to {0,1,2}
    """

    # Oxford trimap pixel values -> 0-indexed class ids
    TRIMAP_MAP = {1: 0, 2: 1, 3: 2}

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: int = 224,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Args:
            root: Path to the Oxford-IIIT Pet dataset root
                  (should contain images/, annotations/xmls/, annotations/trimaps/).
            split: one of 'train', 'val', 'test'.
            image_size: Resize target.
            val_ratio: Fraction of data for validation.
            test_ratio: Fraction of data for test.
            seed: Random seed for reproducible splits.
        """
        assert split in ("train", "val", "test"), f"Unknown split '{split}'"
        self.root = Path(root)
        self.split = split
        self.image_size = image_size

        self.transforms = (
            get_train_transforms(image_size)
            if split == "train"
            else get_val_transforms(image_size)
        )

        # Build sample list from images directory
        images_dir = self.root / "images"
        samples = sorted(images_dir.glob("*.jpg"))

        # Derive class name -> index mapping (sorted for determinism)
        breed_names = sorted({p.stem.rsplit("_", 1)[0] for p in samples})
        self.class_to_idx = {name: idx for idx, name in enumerate(breed_names)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Reproducible train/val/test split (NO LEAKAGE)
        rng = random.Random(seed)
        samples_list = list(samples)
        rng.shuffle(samples_list)

        n = len(samples_list)
        n_test = int(n * test_ratio)
        n_val  = int(n * val_ratio)

        test_set  = samples_list[:n_test]
        val_set   = samples_list[n_test : n_test + n_val]
        train_set = samples_list[n_test + n_val :]

        split_map = {"train": train_set, "val": val_set, "test": test_set}
        self.samples = split_map[split]

        self.annotations_dir = self.root / "annotations"
        self.xmls_dir        = self.annotations_dir / "xmls"
        self.trimaps_dir     = self.annotations_dir / "trimaps"

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        stem     = img_path.stem                       # e.g. "Abyssinian_1"
        breed    = stem.rsplit("_", 1)[0]              # "Abyssinian"
        label    = self.class_to_idx[breed]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))
        H_orig, W_orig = image.shape[:2]

        # Load trimap (PNG)
        trimap_path = self.trimaps_dir / f"{stem}.png"
        if trimap_path.exists():
            trimap_raw = np.array(Image.open(trimap_path).convert("L"))  # {1,2,3}
            # Remap to 0-indexed: 1->0 (fg), 2->1 (bg), 3->2 (border)
            trimap = np.zeros_like(trimap_raw, dtype=np.uint8)
            for src, dst in self.TRIMAP_MAP.items():
                trimap[trimap_raw == src] = dst
        else:
            trimap = np.zeros((H_orig, W_orig), dtype=np.uint8)

        # Load bounding box from XML
        bbox_cxcywh = self._load_bbox(stem, W_orig, H_orig)

        # Convert bbox to COCO format (x_min, y_min, w, h) for albumentations
        cx, cy, bw, bh = bbox_cxcywh
        x_min = cx - bw / 2
        y_min = cy - bh / 2
        # Clamp to image bounds
        x_min = max(0.0, x_min)
        y_min = max(0.0, y_min)
        bw    = min(bw, W_orig - x_min)
        bh    = min(bh, H_orig - y_min)

        transformed = self.transforms(
            image=image,
            mask=trimap,
            bboxes=[[x_min, y_min, bw, bh]],
            bbox_labels=[label],
        )

        img_tensor = transformed["image"]           # [3,H,W] float
        mask_tensor = torch.as_tensor(
            transformed["mask"], dtype=torch.long
        )                                           # [H,W]

        # Recover transformed bbox -> convert back to cxcywh in pixel space
        if transformed["bboxes"]:
            tx, ty, tw, th = transformed["bboxes"][0]
            tcx = tx + tw / 2
            tcy = ty + th / 2
        else:
            # Fallback: bbox became invalid after transform; use image centre
            tcx, tcy = self.image_size / 2, self.image_size / 2
            tw,  th  = self.image_size * 0.5, self.image_size * 0.5

        bbox_tensor = torch.tensor([tcx, tcy, tw, th], dtype=torch.float32)

        return {
            "image":      img_tensor,
            "label":      torch.tensor(label, dtype=torch.long),
            "bbox":       bbox_tensor,
            "mask":       mask_tensor,
        }

    # ------------------------------------------------------------------
    def _load_bbox(self, stem: str, W: int, H: int) -> Tuple[float, float, float, float]:
        """Parse bounding box from Pascal VOC XML; return (cx, cy, w, h) in pixels."""
        xml_path = self.xmls_dir / f"{stem}.xml"
        if not xml_path.exists():
            # No annotation: return full image box
            return float(W / 2), float(H / 2), float(W), float(H)

        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(xml_path)
            root = tree.getroot()
            obj  = root.find(".//bndbox")
            xmin = float(obj.find("xmin").text)
            ymin = float(obj.find("ymin").text)
            xmax = float(obj.find("xmax").text)
            ymax = float(obj.find("ymax").text)
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            bw = xmax - xmin
            bh = ymax - ymin
            return cx, cy, bw, bh
        except Exception:
            return float(W / 2), float(H / 2), float(W), float(H)