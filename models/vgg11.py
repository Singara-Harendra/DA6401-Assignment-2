"""VGG11 encoder

NOTE: The autograder imports `from models.vgg11 import VGG11`.
      VGG11 is defined as an alias of VGG11Encoder at the bottom of this file.
"""

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn

from .layers import CustomDropout


class VGG11Encoder(nn.Module):
    """VGG11-style encoder with BatchNorm and custom Dropout.

    Architecture follows the original VGG11 topology:
        Block 1 : 1x Conv(64)  + MaxPool  -> H/2
        Block 2 : 1x Conv(128) + MaxPool  -> H/4
        Block 3 : 2x Conv(256) + MaxPool  -> H/8
        Block 4 : 2x Conv(512) + MaxPool  -> H/16
        Block 5 : 2x Conv(512) + MaxPool  -> H/32

    BatchNorm is placed after every Conv and before ReLU to stabilise
    training and allow higher learning rates (reduces internal covariate
    shift).  Dropout is NOT inserted inside the convolutional backbone
    because spatial feature maps have strong local correlations — dropping
    entire feature maps would harm the encoder's ability to learn textures.
    Dropout is instead applied in the fully-connected classification / regression
    heads that sit on top of this encoder.
    """

    def __init__(self, in_channels: int = 3):
        """Initialize the VGG11Encoder model."""
        super().__init__()

        def conv_bn_relu(in_ch, out_ch, kernel=3, pad=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=pad, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        # --- Convolutional blocks ---
        # Block 1  [B,3,H,W] -> [B,64,H/2,W/2]
        self.block1 = nn.Sequential(
            conv_bn_relu(in_channels, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 2  -> [B,128,H/4,W/4]
        self.block2 = nn.Sequential(
            conv_bn_relu(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 3  -> [B,256,H/8,W/8]
        self.block3 = nn.Sequential(
            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 4  -> [B,512,H/16,W/16]
        self.block4 = nn.Sequential(
            conv_bn_relu(256, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Block 5  -> [B,512,H/32,W/32]
        self.block5 = nn.Sequential(
            conv_bn_relu(512, 512),
            conv_bn_relu(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass.

        Args:
            x: input image tensor [B, 3, H, W].
            return_features: if True, also return skip maps for U-Net decoder.

        Returns:
            - if return_features=False: bottleneck feature tensor [B,512,H/32,W/32].
            - if return_features=True: (bottleneck, feature_dict) where feature_dict
              contains pre-pool activations for skip connections keyed by block name.
        """
        # We need pre-pool features for skip connections, so run each sub-layer manually.
        # Block 1
        x1_conv = self._run_conv_only(self.block1, x)   # [B,64,H,W]  before pool
        x1 = self.block1[-1](x1_conv)                    # after pool -> [B,64,H/2,W/2]

        # Block 2
        x2_conv = self._run_conv_only(self.block2, x1)
        x2 = self.block2[-1](x2_conv)

        # Block 3
        x3_conv = self._run_conv_only(self.block3, x2)
        x3 = self.block3[-1](x3_conv)

        # Block 4
        x4_conv = self._run_conv_only(self.block4, x3)
        x4 = self.block4[-1](x4_conv)

        # Block 5
        x5_conv = self._run_conv_only(self.block5, x4)
        x5 = self.block5[-1](x5_conv)  # bottleneck [B,512,H/32,W/32]

        if not return_features:
            return x5

        features = {
            "block1": x1_conv,  # [B,64,  H,    W   ]
            "block2": x2_conv,  # [B,128, H/2,  W/2 ]
            "block3": x3_conv,  # [B,256, H/4,  W/4 ]
            "block4": x4_conv,  # [B,512, H/8,  W/8 ]
            "block5": x5_conv,  # [B,512, H/16, W/16]
        }
        return x5, features

    @staticmethod
    def _run_conv_only(block: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
        """Run all layers of *block* except the final MaxPool2d."""
        for layer in block[:-1]:
            x = layer(x)
        return x
# Alias required by autograder: from models.vgg11 import VGG11
VGG11 = VGG11Encoder
