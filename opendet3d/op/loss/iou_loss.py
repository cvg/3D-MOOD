"""IoU Loss."""

import torch
from torch import Tensor
from vis4d.op.loss.base import Loss
from vis4d.op.loss.reducer import LossReducer, mean_loss

from opendet3d.op.box.box2d import bbox_overlaps


def giou_loss(pred: Tensor, target: Tensor, eps: float = 1e-7) -> Tensor:
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    # avoid fp16 overflow
    if pred.dtype == torch.float16:
        fp16 = True
        pred = pred.to(torch.float32)
    else:
        fp16 = False

    gious = bbox_overlaps(pred, target, mode="giou", is_aligned=True, eps=eps)

    if fp16:
        gious = gious.to(torch.float16)

    loss = 1 - gious
    return loss


class GIoULoss(Loss):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        reducer: LossReducer = mean_loss,
    ) -> None:
        super().__init__(reducer)
        self.eps = eps

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        reducer: LossReducer | None = None,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        reducer = reducer or self.reducer

        loss = giou_loss(pred, target, eps=self.eps)

        return reducer(loss)
