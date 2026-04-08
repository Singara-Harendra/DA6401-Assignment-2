"""Reusable custom layers 
"""

import torch
import torch.nn as nn


class CustomDropout(nn.Module):
    """Custom Dropout layer using inverted dropout scaling.

    During training:
        - A binary mask is sampled from Bernoulli(1 - p).
        - The mask is applied element-wise to the input.
        - The output is scaled by 1/(1-p) so the expected value of each
          activation is preserved at test time (inverted / "keep-scale" dropout).
    During evaluation (self.training == False):
        - The input is returned unchanged.
    """

    def __init__(self, p: float = 0.5):
        """
        Initialize the CustomDropout layer.

        Args:
            p: Dropout probability (probability of zeroing an element).
               Must be in [0, 1).
        """
        super().__init__()
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CustomDropout layer.

        Args:
            x: Input tensor of any shape.

        Returns:
            Output tensor with same shape as input.
        """
        # Eval mode or p==0: identity
        if not self.training or self.p == 0.0:
            return x

        keep_prob = 1.0 - self.p
        # Sample binary keep-mask: 1 = keep, 0 = drop
        mask = torch.bernoulli(
            torch.full(x.shape, keep_prob, dtype=x.dtype, device=x.device)
        )
        # Inverted dropout: scale surviving activations so E[output] == E[input]
        return x * mask / keep_prob
