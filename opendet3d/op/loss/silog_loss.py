"""SILog loss for depth estimation."""

from __future__ import annotations

import torch
from torch import Tensor
from vis4d.common.typing import ArgsType
from vis4d.op.loss.base import Loss

from .util import masked_mean_var


class SILogLoss(Loss):
    """SILogLoss."""

    def __init__(
        self,
        *args: ArgsType,
        scale_pred_weight: float = 0.15,
        eps: float = 1e-5,
        min_depth: float = 0.0,
        **kwargs: ArgsType,
    ) -> None:
        """Init."""
        super().__init__(*args, **kwargs)
        self.scale_pred_weight = scale_pred_weight
        self.eps = eps
        self.min_depth = min_depth

    def forward(
        self, depths: Tensor, target_depths: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        """Forward function.

        Args:
            depths (Tensor): Predicted depth. Shape: (B, H, W)
            target_depths (Tensor): Target depth. Shape: (B, H, W)
            mask (Tensor | None): Mask. Shape: (B, H, W)
        """
        if mask is None:
            mask = target_depths > self.min_depth
        else:
            mask = mask.to(torch.bool)
            mask = torch.logical_and(mask, target_depths > self.min_depth)

        log_depths = torch.log(depths.clamp(min=self.eps))
        log_target_depths = torch.log(target_depths.clamp(min=self.eps))

        log_error = log_depths - log_target_depths

        mean_error, var_error = masked_mean_var(log_error, mask=mask)

        scale_error = mean_error**2

        loss = var_error + self.scale_pred_weight * scale_error

        out_loss = torch.sqrt(loss.clamp(min=self.eps))

        return out_loss.mean()
