"""Localization modules
"""

import torch
import torch.nn as nn

from .vgg11 import VGG11Encoder
from .layers import CustomDropout


class VGG11Localizer(nn.Module):
    """VGG11-based single-object localizer.

    The convolutional backbone from VGG11 (the encoder) is used as a feature
    extractor.  A regression head consisting of two FC layers with BatchNorm1d
    and CustomDropout is appended.  The output is passed through a Sigmoid and
    then scaled to pixel coordinates so that predictions are bounded and
    interpretable.

    Design choice — freezing vs fine-tuning:
        The backbone weights are fine-tuned (NOT frozen).  The spatial
        bounding-box task requires different feature sensitivities than the
        classification task: localisation benefits from position-aware features
        in the later convolutional blocks.  Fine-tuning the full backbone lets
        those blocks adapt, whereas freezing them forces the network to solve a
        geometric task using features optimised purely for recognition.
    """

    IMAGE_SIZE: int = 224 

    def __init__(self, in_channels: int = 3, dropout_p: float = 0.5):
        """
        Initialize the VGG11Localizer model.

        Args:
            in_channels: Number of input channels.
            dropout_p: Dropout probability for the localization head.
        """
        super().__init__()
        self.encoder = VGG11Encoder(in_channels=in_channels)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))


        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            CustomDropout(p=dropout_p),
            nn.Linear(1024, 4),  
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for localization model.

        Args:
            x: Input tensor of shape [B, in_channels, H, W].

        Returns:
            Bounding box coordinates [B, 4] in (x_center, y_center, width, height)
            format expressed in original image pixel space (not normalised).
        """
        H, W = x.shape[2], x.shape[3]
        features = self.encoder(x)           
        pooled   = self.adaptive_pool(features)  
        raw      = self.regressor(pooled)    


        normalised = torch.sigmoid(raw)        
        scale = torch.tensor([W, H, W, H], dtype=x.dtype, device=x.device)
        return normalised * scale              
