

import torch
import torch.nn as nn

class IoULoss(nn.Module):
    """IoU loss for bounding box regression.

    Boxes are expected in (x_center, y_center, width, height) format.
    The loss is defined as  1 - IoU  so that minimising it maximises overlap.
    A small epsilon is added to numerator and denominator for numerical stability
    and to ensure a non-zero gradient even for non-overlapping boxes.
    """

    def __init__(self, eps: float = 1e-6, reduction: str = "mean"):
        
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got '{reduction}'")
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
      

        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        tgt_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        tgt_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        tgt_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        tgt_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2


        inter_x1 = torch.max(pred_x1, tgt_x1)
        inter_y1 = torch.max(pred_y1, tgt_y1)
        inter_x2 = torch.min(pred_x2, tgt_x2)
        inter_y2 = torch.min(pred_y2, tgt_y2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        intersection = inter_w * inter_h


        pred_area = (pred_x2 - pred_x1).clamp(min=0) * (pred_y2 - pred_y1).clamp(min=0)
        tgt_area  = (tgt_x2 - tgt_x1).clamp(min=0) * (tgt_y2 - tgt_y1).clamp(min=0)

        union = pred_area + tgt_area - intersection + self.eps
        iou = (intersection + self.eps) / union  # [B]

        loss = 1.0 - iou  # [B]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss