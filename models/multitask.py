"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn
import gdown

from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

#cid = https://drive.google.com/file/d/1oTkQJXHqKh7K4VFYJ8UAv7-XTNt8F26j/view?usp=sharing
#lid = https://drive.google.com/file/d/1TGGdkTiSjinKB7pckM9rE87tUjH_YUdK/view?usp=sharing
#uid = https://drive.google.com/file/d/1cFT1mGimel4w_Zq1vw6De8Rje7Q3ntFb/view?usp=sharing

CLASSIFIER_DRIVE_ID = '1oTkQJXHqKh7K4VFYJ8UAv7-XTNt8F26j'
LOCALIZER_DRIVE_ID = '1TGGdkTiSjinKB7pckM9rE87tUjH_YUdK' 
UNET_DRIVE_ID = '1cFT1mGimel4w_Zq1vw6De8Rje7Q3ntFb'

def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class MultiTaskPerceptionModel(nn.Module):
    """Shared-backbone multi-task model.

    A single VGG11 encoder is shared across three task heads:
        1. Classification head   -> 37-class breed logits
        2. Localisation head     -> 4 bounding-box coordinates
        3. Segmentation decoder  -> pixel-wise 3-class trimap logits

    Weights are loaded from individually-trained task models and the backbone
    is then fine-tuned jointly.
    """

    def __init__(
        self,
        num_breeds: int = 37,
        seg_classes: int = 3,
        in_channels: int = 3,
        classifier_path: str = "classifier.pth",
        localizer_path: str = "localizer.pth",
        unet_path: str = "unet.pth",
        dropout_p: float = 0.5,
    ):
        """
        Initialize the shared backbone/heads using pre-trained weights when available.

        Args:
            num_breeds: Number of output classes for classification head.
            seg_classes: Number of output classes for segmentation head.
            in_channels: Number of input channels.
            classifier_path: Path to trained classifier weights.
            localizer_path: Path to trained localizer weights.
            unet_path: Path to trained U-Net weights.
            dropout_p: Dropout probability for all heads.
        """
        super().__init__()

        # ---- Download checkpoints from Google Drive ----
        gdown.download(id=CLASSIFIER_DRIVE_ID, output=classifier_path, quiet=False)
        gdown.download(id=LOCALIZER_DRIVE_ID, output=localizer_path, quiet=False)
        gdown.download(id=UNET_DRIVE_ID, output=unet_path, quiet=False)

        # ---- Shared backbone ----
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ---- Classification head ----
        self.adaptive_pool_cls = nn.AdaptiveAvgPool2d((7, 7))
        self.cls_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds),
        )

        # ---- Localisation head ----
        self.adaptive_pool_loc = nn.AdaptiveAvgPool2d((7, 7))
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
        )

        # ---- Segmentation decoder (U-Net expansive path) ----
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _double_conv(32 + 64, 32)
        self.seg_dropout = CustomDropout(p=dropout_p)
        self.seg_final = nn.Conv2d(32, seg_classes, kernel_size=1)

        # ---- Load pre-trained weights if available ----
        self._load_pretrained(classifier_path, localizer_path, unet_path)

    # ------------------------------------------------------------------
   def _load_pretrained(self, clf_path: str, loc_path: str, unet_path: str):
    """Load weights from individually-trained models into the unified model."""

    def _try_load(path: str):
        if path and os.path.isfile(path):
            ckpt = torch.load(path, map_location="cpu")
            return ckpt.get("state_dict", ckpt.get("model", ckpt))
        return None

    clf_state  = _try_load(clf_path)
    loc_state  = _try_load(loc_path)
    unet_state = _try_load(unet_path)

    # Build one combined state dict for self (MultiTaskPerceptionModel)
    # by explicitly renaming keys from each individual model checkpoint.
    combined = {}

    # ---- From classifier.pth (VGG11Classifier) ----
    # Keys: encoder.* and classifier.*
    # Maps: classifier.* -> cls_head.*  (same indices, just different module name)
    if clf_state is not None:
        for k, v in clf_state.items():
            if k.startswith("encoder."):
                combined[k] = v                          # encoder.* -> encoder.*
            elif k.startswith("classifier."):
                new_k = "cls_head." + k[len("classifier."):]
                combined[new_k] = v                      # classifier.* -> cls_head.*

    # ---- From localizer.pth (VGG11Localizer) ----
    # Keys: encoder.* and regressor.*
    # Maps: regressor.* -> loc_head.*
    if loc_state is not None:
        for k, v in loc_state.items():
            if k.startswith("regressor."):
                new_k = "loc_head." + k[len("regressor."):]
                combined[new_k] = v                      # regressor.* -> loc_head.*
            # skip encoder.* — already loaded from classifier

    # ---- From unet.pth (VGG11UNet) ----
    # Keys: encoder.*, up5.*, dec5.*, ..., up1.*, dec1.*, dropout.*, final_conv.*
    # Maps: final_conv.* -> seg_final.*  (everything else same name)
    if unet_state is not None:
        for k, v in unet_state.items():
            if k.startswith("encoder."):
                pass  # already loaded from classifier
            elif k.startswith("final_conv."):
                new_k = "seg_final." + k[len("final_conv."):]
                combined[new_k] = v                      # final_conv.* -> seg_final.*
            elif k.startswith("dropout."):
                new_k = "seg_dropout." + k[len("dropout."):]
                combined[new_k] = v                      # dropout.* -> seg_dropout.*
            else:
                combined[k] = v  # up5.*, dec5.*, up4.*, dec4.* etc. match exactly

    if combined:
        missing, unexpected = [], []
        result = self.load_state_dict(combined, strict=False)
        print(f"[MultiTask] Loaded {len(combined)} keys.")
        print(f"[MultiTask] Missing keys : {result.missing_keys}")
        print(f"[MultiTask] Unexpected   : {result.unexpected_keys}")
    else:
        print("[MultiTask] WARNING: No checkpoint weights loaded — all heads are random!")


    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Single forward pass over the shared backbone.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            dict with keys:
                'classification': [B, num_breeds] logits
                'localization':   [B, 4] bounding box coordinates (pixel space)
                'segmentation':   [B, seg_classes, H, W] logits
        """
        H, W = x.shape[2], x.shape[3]

        # Shared encoder with skip connections for segmentation
        bottleneck, skips = self.encoder(x, return_features=True)

        # ---- Classification ----
        cls_feat = self.adaptive_pool_cls(bottleneck)   # [B,512,7,7]
        cls_out  = self.cls_head(cls_feat)               # [B,37]

        # ---- Localisation ----
        loc_feat = self.adaptive_pool_loc(bottleneck)   # [B,512,7,7]
        raw_bbox = self.loc_head(loc_feat)               # [B,4]
        scale    = torch.tensor([W, H, W, H], dtype=x.dtype, device=x.device)
        bbox_out = torch.sigmoid(raw_bbox) * scale       # [B,4] pixel coords

        # ---- Segmentation ----
        d = self.up5(bottleneck)
        d = torch.cat([d, skips["block5"]], dim=1)
        d = self.dec5(d)

        d = self.up4(d)
        d = torch.cat([d, skips["block4"]], dim=1)
        d = self.dec4(d)

        d = self.up3(d)
        d = torch.cat([d, skips["block3"]], dim=1)
        d = self.dec3(d)

        d = self.up2(d)
        d = torch.cat([d, skips["block2"]], dim=1)
        d = self.dec2(d)

        d = self.up1(d)
        d = torch.cat([d, skips["block1"]], dim=1)
        d = self.dec1(d)

        d = self.seg_dropout(d)
        seg_out = self.seg_final(d)                      # [B,seg_classes,H,W]

        return {
            "classification": cls_out,
            "localization":   bbox_out,
            "segmentation":   seg_out,
        }