"""Unified multi-task model
"""

import os
import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout
from .classification import VGG11Classifier
from .localization import VGG11Localizer
from .segmentation import VGG11UNet

#https://drive.google.com/file/d/1hPMCBM5kHexkK4mI_HPUI-wYLMxEOyr_/view?usp=sharing

CLASSIFIER_DRIVE_ID = '1hPMCBM5kHexkK4mI_HPUI-wYLMxEOyr_'

LOCALIZER_DRIVE_ID = '1oxy2Xk2pdUTX_g0pq7odch9vlD8G0-uj' 
UNET_DRIVE_ID = '1ogUFUjDZZ7sNMiwttB0RxSia9BIec5ca'

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
        """Copy weights from individually-trained task models."""
        def _try_load(path: str):
            if path and os.path.isfile(path):
                ckpt = torch.load(path, map_location="cpu")
                return ckpt.get("state_dict", ckpt.get("model", ckpt))
            return None

        clf_state  = _try_load(clf_path)
        loc_state  = _try_load(loc_path)
        unet_state = _try_load(unet_path)

        if clf_state is not None:
            # Backbone from classifier (first available wins)
            enc_state = {k[len("encoder."):]: v for k, v in clf_state.items()
                         if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc_state, strict=False)

            cls_state = {k[len("classifier."):]: v for k, v in clf_state.items()
                         if k.startswith("classifier.")}
            self.cls_head.load_state_dict(cls_state, strict=False)

        if loc_state is not None:
            loc_head_state = {k[len("regressor."):]: v for k, v in loc_state.items()
                              if k.startswith("regressor.")}
            self.loc_head.load_state_dict(loc_head_state, strict=False)

        if unet_state is not None:
            # Maps: (key_in_unet_state_dict, attribute_on_self)
            seg_key_map = [
                ("up5", "up5"), ("dec5", "dec5"),
                ("up4", "up4"), ("dec4", "dec4"),
                ("up3", "up3"), ("dec3", "dec3"),
                ("up2", "up2"), ("dec2", "dec2"),
                ("up1", "up1"), ("dec1", "dec1"),
                ("dropout", "seg_dropout"),
                ("final_conv", "seg_final"),   # VGG11UNet uses final_conv; we store as seg_final
            ]
            for unet_key, self_attr in seg_key_map:
                module = getattr(self, self_attr, None)
                if module is None:
                    continue
                sub_state = {k[len(unet_key) + 1:]: v for k, v in unet_state.items()
                             if k.startswith(unet_key + ".")}
                if sub_state:
                    try:
                        module.load_state_dict(sub_state, strict=False)
                    except Exception:
                        pass

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
