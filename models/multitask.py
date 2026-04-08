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

        # RENAME: 'classifier' matches your VGG11Classifier class
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

        # RENAME: 'regressor' matches your VGG11Localizer class
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4)
        )

        # Segmentation Decoder
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
        self.seg_final = nn.Conv2d(32, seg_classes, 1)

        self._load_pretrained(classifier_path, localizer_path, unet_path)

    def _double_conv(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(True)
        )

    def _load_pretrained(self, clf_p, loc_p, unet_p):
        """Loads weights by checking for prefix matches."""
        for path in [clf_p, loc_p, unet_p]:
            if os.path.exists(path):
                state = torch.load(path, map_location='cpu')
                # This helps if weights are wrapped in 'model.' or 'module.'
                clean_state = {k.replace('model.', '').replace('module.', ''): v for k, v in state.items()}
                self.load_state_dict(clean_state, strict=False)

    def forward(self, x):
        H, W = x.shape[2], x.shape[3]
        bottleneck, skips = self.encoder(x, return_features=True)
        
        # 1. Classification
        cls_out = self.classifier(self.adaptive_pool(bottleneck))
        
        # 2. Localization (Must match your localizer's sigmoid + scale logic)
        raw_loc = self.regressor(self.adaptive_pool(bottleneck))
        loc_out = torch.sigmoid(raw_loc) * torch.tensor([W, H, W, H], device=x.device)
        
        # 3. Segmentation (Concatenate skips)
        d5 = self.dec5(torch.cat([self.up5(bottleneck), skips['block5']], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d5), skips['block4']], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), skips['block3']], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), skips['block2']], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), skips['block1']], dim=1))
        
        return {"classification": cls_out, "localization": loc_out, "segmentation": self.seg_final(d1)}