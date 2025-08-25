"""Focal Loss."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor
from torchvision.ops import sigmoid_focal_loss
from vis4d.op.loss.base import Loss
from vis4d.op.loss.reducer import LossReducer, mean_loss


class FocalLoss(Loss):
    """Focal loss <https://arxiv.org/abs/1708.02002>`_."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reducer: LossReducer = mean_loss,
    ) -> None:
        """Creates an instance of the class.

        Args:
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            reducer (LossReducer, optional): Reducer for the loss function.
                Defaults to mean_loss.
        """
        super().__init__(reducer)
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, pred: Tensor, target: Tensor, reducer: LossReducer | None = None
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): The prediction.
            target (Tensor): The learning label of the prediction.

        Returns:
            Tensor: The calculated loss.
        """
        # this means that target is not in One-Hot form.
        if pred.dim() != target.dim():
            num_classes = pred.size(1)
            target = F.one_hot(target, num_classes=num_classes + 1).float()
            target = target[:, :num_classes]

        reducer = reducer or self.reducer

        focal_loss = sigmoid_focal_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
        )

        return reducer(focal_loss)
