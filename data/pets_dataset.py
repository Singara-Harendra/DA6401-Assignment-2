"""Dataset for Oxford-IIIT Pet — train/val split only.

The autograder provides its own private test set, so we use a simple
90 % train / 10 % val split, maximising training data.
"""

import os
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


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
            format="coco",
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


def _find_images_dir(root: Path) -> Path:
    """Locate the 'images/' folder, tolerating common Drive nesting issues.

    Handles these layouts:
        root/images/*.jpg                  <- standard
        root/oxford-iiit-pet/images/*.jpg  <- nested after tar extract
        root/*.jpg                         <- flat (unlikely but handled)
    """
    candidate = root / "images"
    if candidate.is_dir() and any(candidate.glob("*.jpg")):
        return candidate

    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            candidate = sub / "images"
            if candidate.is_dir() and any(candidate.glob("*.jpg")):
                return candidate

    if any(root.glob("*.jpg")):
        return root

    return root / "images"


def _find_annotations_dir(root: Path, images_dir: Path) -> Path:
    """Find annotations/ as a sibling of wherever images/ was found."""
    ann = images_dir.parent / "annotations"
    if ann.is_dir():
        return ann
    return root / "annotations"


class OxfordIIITPetDataset(Dataset):
   

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
        assert split in ("train", "val", "test"), f"Unknown split '{split}'"
        self.root       = Path(root)
        self.split      = split
        self.image_size = image_size

        self.transforms = (
            get_train_transforms(image_size)
            if split == "train"
            else get_val_transforms(image_size)
        )

        images_dir = _find_images_dir(self.root)
        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Cannot find the 'images/' folder inside '{root}'.\n"
                f"Expected layout:  {root}/images/*.jpg\n"
                f"Diagnose with:\n"
                f"  from pathlib import Path\n"
                f"  for p in sorted(Path('{root}').rglob('*.jpg'))[:5]: print(p)"
            )


        samples = sorted([
            p for p in images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        if len(samples) == 0:
            raise FileNotFoundError(
                f"No images found in '{images_dir}'.\n"
                f"Check the dataset extracted correctly."
            )

      
      breed_names = [
            'Abyssinian', 'Bengal', 'Birman', 'Bombay', 'British_Shorthair',
            'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue',
            'Siamese', 'Sphynx', 'american_bulldog', 'american_pit_bull_terrier',
            'basset_hound', 'beagle', 'boxer', 'chihuahua', 'english_cocker_spaniel',
            'english_setter', 'german_shorthaired', 'great_pyrenees', 'havanese',
            'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher',
            'newfoundland', 'pomeranian', 'pug', 'saint_bernard', 'samoyed',
            'scottish_terrier', 'shiba_inu', 'staffordshire_bull_terrier',
            'wheaten_terrier', 'yorkshire_terrier'
        ]
        
        self.class_to_idx = {name: idx for idx, name in enumerate(breed_names)}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        
        rng = random.Random(seed)
        samples_list = list(samples)
        rng.shuffle(samples_list)

        n     = len(samples_list)
        n_val = int(n * val_ratio)     

        val_set   = samples_list[:n_val]
        train_set = samples_list[n_val:]


        split_map = {"train": train_set, "val": val_set, "test": val_set}
        self.samples = split_map[split]

        print(
            f"[Dataset] split={split}  |  total={n}  "
            f"train={len(train_set)}  val={len(val_set)}  "
            f"this_split={len(self.samples)}"
        )


        ann_dir          = _find_annotations_dir(self.root, images_dir)
        self.xmls_dir    = ann_dir / "xmls"
        self.trimaps_dir = ann_dir / "trimaps"



    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path = self.samples[idx]
        stem     = img_path.stem                
        breed    = stem.rsplit("_", 1)[0]       
        label    = self.class_to_idx[breed]


        image          = np.array(Image.open(img_path).convert("RGB"))
        H_orig, W_orig = image.shape[:2]


        trimap_path = self.trimaps_dir / f"{stem}.png"
        if trimap_path.exists():
            trimap_raw = np.array(Image.open(trimap_path).convert("L"))
            trimap = np.zeros_like(trimap_raw, dtype=np.uint8)
            for src, dst in self.TRIMAP_MAP.items():
                trimap[trimap_raw == src] = dst
        else:
            trimap = np.zeros((H_orig, W_orig), dtype=np.uint8)


        cx, cy, bw, bh = self._load_bbox(stem, W_orig, H_orig)
        x_min = max(0.0, cx - bw / 2)
        y_min = max(0.0, cy - bh / 2)
        bw    = min(bw, W_orig - x_min)
        bh    = min(bh, H_orig - y_min)

        transformed = self.transforms(
            image=image,
            mask=trimap,
            bboxes=[[x_min, y_min, bw, bh]],
            bbox_labels=[label],
        )

        img_tensor  = transformed["image"]                                    # [3,H,W]
        mask_tensor = torch.as_tensor(transformed["mask"], dtype=torch.long)  # [H,W]

        if transformed["bboxes"]:
            tx, ty, tw, th = transformed["bboxes"][0]
            tcx = tx + tw / 2
            tcy = ty + th / 2
        else:
            # bbox clipped out of frame after augmentation — use image centre
            tcx = tcy = self.image_size / 2
            tw  = th  = self.image_size * 0.5

        bbox_tensor = torch.tensor([tcx, tcy, tw, th], dtype=torch.float32)

        return {
            "image": img_tensor,
            "label": torch.tensor(label, dtype=torch.long),
            "bbox":  bbox_tensor,
            "mask":  mask_tensor,
        }



    def _load_bbox(self, stem: str, W: int, H: int) -> Tuple[float, float, float, float]:
        """Return (cx, cy, w, h) in pixels from Pascal VOC XML annotation."""
        xml_path = self.xmls_dir / f"{stem}.xml"
        if not xml_path.exists():

            return float(W / 2), float(H / 2), float(W), float(H)
        try:
            import xml.etree.ElementTree as ET
            root = ET.parse(xml_path).getroot()
            obj  = root.find(".//bndbox")
            xmin = float(obj.find("xmin").text)
            ymin = float(obj.find("ymin").text)
            xmax = float(obj.find("xmax").text)
            ymax = float(obj.find("ymax").text)
            return (xmin + xmax) / 2, (ymin + ymax) / 2, xmax - xmin, ymax - ymin
        except Exception:
            return float(W / 2), float(H / 2), float(W), float(H)