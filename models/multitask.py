import os
import torch
import torch.nn as nn
from .vgg11 import VGG11Encoder
from .layers import CustomDropout

class MultiTaskPerceptionModel(nn.Module):
    def __init__(self, num_breeds=37, seg_classes=3, in_channels=3, 
                 classifier_path="classifier.pth", localizer_path="localizer.pth", 
                 unet_path="unet.pth", dropout_p=0.5):
        super().__init__()

        self.encoder = VGG11Encoder(in_channels=in_channels)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # MUST match VGG11Classifier variable name exactly
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(4096, num_breeds)
        )

        # MUST match VGG11Localizer variable name exactly
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4)
        )

        # Segmentation Decoder - Variable names must match segmentation.py
        self.up5 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.dec5 = self._double_conv(1024, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = self._double_conv(768, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = self._double_conv(384, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._double_conv(192, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self._double_conv(96, 32)
        
        self.seg_dropout = CustomDropout(p=dropout_p)
        self.final_conv = nn.Conv2d(32, seg_classes, 1) # Match name in segmentation.py

        self._load_pretrained(classifier_path, localizer_path, unet_path)

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True)
        )

    def _load_pretrained(self, clf_p, loc_p, unet_p):
        for path in [clf_p, loc_p, unet_p]:
            if os.path.exists(path):
                state = torch.load(path, map_location='cpu')
                # Unwrap model if saved from DataParallel or specific dict keys
                actual_state = state.get("model", state)
                clean_state = {k.replace('model.', '').replace('module.', ''): v for k, v in actual_state.items()}
                self.load_state_dict(clean_state, strict=False)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        bottleneck, skips = self.encoder(x, return_features=True)
        
        # 1. Classification
        cls_out = self.classifier(self.adaptive_pool(bottleneck))
        
        # 2. Localization - MUST mirror your standalone logic
        raw_loc = self.regressor(self.adaptive_pool(bottleneck))
        loc_out = torch.sigmoid(raw_loc) * torch.tensor([W, H, W, H], dtype=x.dtype, device=x.device)
        
        # 3. Segmentation
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

        seg_out = self.final_conv(self.seg_dropout(d))
        
        return {"classification": cls_out, "localization": loc_out, "segmentation": seg_out}