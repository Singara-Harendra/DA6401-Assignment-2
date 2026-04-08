import os
import torch
import torch.nn as nn
import gdown

from .vgg11 import VGG11Encoder
from .layers import CustomDropout

CLASSIFIER_DRIVE_ID = '1zdSg4gsWsN_e9OYdutC2pnJLcXBSMto3'
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
        super().__init__()

        # Download weights
        gdown.download(id=CLASSIFIER_DRIVE_ID, output=classifier_path, quiet=True)
        gdown.download(id=LOCALIZER_DRIVE_ID,  output=localizer_path,  quiet=True)
        gdown.download(id=UNET_DRIVE_ID,       output=unet_path,       quiet=True)

        # 1. Shared Backbone
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # 2. Classification Head
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

        # 3. Localization Head
        self.adaptive_pool_loc = nn.AdaptiveAvgPool2d((7, 7))
        self.loc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),
        )

        # 4. Segmentation Decoder
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

        self._load_pretrained(classifier_path, localizer_path, unet_path)

    def _load_pretrained(self, clf_path, loc_path, unet_path):
        def load_state(path):
            return torch.load(path, map_location="cpu") if os.path.exists(path) else None

        # Fix: Bridge the gap between standalone model keys and multi-task keys
        clf_state = load_state(clf_path)
        if clf_state:
            # If your VGG11Classifier had a .encoder and .classifier
            new_enc = {k.replace("encoder.", ""): v for k, v in clf_state.items() if k.startswith("encoder.")}
            self.encoder.load_state_dict(new_enc, strict=False)
            
            new_cls = {k.replace("classifier.", ""): v for k, v in clf_state.items() if k.startswith("classifier.")}
            self.cls_head.load_state_dict(new_cls, strict=False)

        loc_state = load_state(loc_path)
        if loc_state:
            # If your VGG11Localizer had a .regressor
            new_loc = {k.replace("regressor.", ""): v for k, v in loc_state.items() if k.startswith("regressor.")}
            self.loc_head.load_state_dict(new_loc, strict=False)

        unet_state = load_state(unet_path)
        if unet_state:
            # UNet keys often match directly if they weren't nested in a .model attribute
            self.load_state_dict(unet_state, strict=False)

    def forward(self, x):
        # Shared encoder
        bottleneck, skips = self.encoder(x, return_features=True)

        # 1. Classification
        cls_feat = self.adaptive_pool_cls(bottleneck)
        classification_logits = self.cls_head(cls_feat)

        # 2. Localization
        loc_feat = self.adaptive_pool_loc(bottleneck)
        # IMPORTANT: Removed Sigmoid. Use raw coordinates if trained that way.
        localization_output = self.loc_head(loc_feat)

        # 3. Segmentation (Expansive Path)
        d5 = self.up5(bottleneck)
        d5 = torch.cat([d5, skips["block5"]], dim=1)
        d5 = self.dec5(d5)

        d4 = self.up4(d5)
        d4 = torch.cat([d4, skips["block4"]], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, skips["block3"]], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, skips["block2"]], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, skips["block1"]], dim=1)
        d1 = self.dec1(d1)

        segmentation_logits = self.seg_final(self.seg_dropout(d1))

        return {
            "classification": classification_logits,
            "localization":   localization_output,
            "segmentation":   segmentation_logits
        }