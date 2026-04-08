"""Segmentation model
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


def _double_conv(in_ch: int, out_ch: int) -> nn.Sequential:
    """Two Conv-BN-ReLU layers used inside each decoder block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class VGG11UNet(nn.Module):
    """U-Net style segmentation network.

    Encoder : VGG11 convolutional backbone (5 blocks, each halving H×W).
    Decoder : Symmetric 5-stage expansive path using Transposed Convolutions
              for upsampling (NO bilinear interpolation, as required).
    Skip connections: pre-pool feature maps from each encoder block are
              concatenated with the corresponding upsampled decoder output.

    Loss choice: Combined Dice + Binary-Cross-Entropy.
        The Oxford-IIIT Pet trimap has 3 classes (foreground / background /
        boundary), so cross-entropy handles multi-class supervision. Dice loss
        is added because it directly optimises the overlap metric and is robust
        to class imbalance (boundary pixels are rare relative to foreground /
        background).

    Design choice — backbone fine-tuning:
        The encoder backbone weights are fine-tuned end-to-end.  Low-level
        edge features (block1/2) are already somewhat transferable, but
        higher-level blocks (4/5) capture semantic content that is task-
        specific.  Freezing the backbone would limit the decoder's ability to
        receive the right gradient signal through skip connections.
    """

    def __init__(self, num_classes: int = 3, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11UNet model.

        Args:
            num_classes: Number of output classes (3 for Pet trimap).
            in_channels: Number of input channels.
            dropout_p: Dropout probability for decoder.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        # ---- Decoder ----
        # Each stage: ConvTranspose2d doubles H×W, then _double_conv refines.
        # Skip-connection channels are concatenated before _double_conv.
        # Encoder skip channel sizes (pre-pool):
        #   block5: 512, block4: 512, block3: 256, block2: 128, block1: 64

        # Stage 5 -> 4:  bottleneck(512) + skip_block5(512) -> 512
        self.up5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.dec5 = _double_conv(512 + 512, 512)

        # Stage 4 -> 3:  dec5(512) + skip_block4(512) -> 256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = _double_conv(256 + 512, 256)

        # Stage 3 -> 2:  dec4(256) + skip_block3(256) -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = _double_conv(128 + 256, 128)

        # Stage 2 -> 1:  dec3(128) + skip_block2(128) -> 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = _double_conv(64 + 128, 64)

        # Stage 1 -> 0:  dec2(64) + skip_block1(64) -> 32
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = _double_conv(32 + 64, 32)

        self.dropout = CustomDropout(p=dropout_p)

        # 1×1 conv for final class logits
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for segmentation model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Segmentation logits [B, num_classes, H, W].
        """
        # Encoder with skip features
        bottleneck, skips = self.encoder(x, return_features=True)
        # skips keys: block1..block5 (pre-pool activations)

        # Decoder
        d = self.up5(bottleneck)                              # H/16
        d = torch.cat([d, skips["block5"]], dim=1)
        d = self.dec5(d)                                      # [B,512,H/16,W/16]

        d = self.up4(d)                                       # H/8
        d = torch.cat([d, skips["block4"]], dim=1)
        d = self.dec4(d)                                      # [B,256,H/8,W/8]

        d = self.up3(d)                                       # H/4
        d = torch.cat([d, skips["block3"]], dim=1)
        d = self.dec3(d)                                      # [B,128,H/4,W/4]

        d = self.up2(d)                                       # H/2
        d = torch.cat([d, skips["block2"]], dim=1)
        d = self.dec2(d)                                      # [B,64,H/2,W/2]

        d = self.up1(d)                                       # H
        d = torch.cat([d, skips["block1"]], dim=1)
        d = self.dec1(d)                                      # [B,32,H,W]

        d = self.dropout(d)
        return self.final_conv(d)                             # [B,num_classes,H,W]
