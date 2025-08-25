"""Loss util."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_mean_var(error: Tensor, mask: Tensor | None = None) -> Tensor:
    """Compute mean and variance of error with mask."""
    if mask is None:
        return error.mean(dim=[-2, -1], keepdim=True), error.var(
            dim=[-2, -1], keepdim=True
        )
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=[-2, -1], keepdim=True)
    mask_mean = torch.sum(
        error * mask, dim=[-2, -1], keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    mask_var = torch.sum(
        mask * (error - mask_mean) ** 2, dim=[-2, -1], keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean.squeeze([-2, -1]), mask_var.squeeze([-2, -1])


def masked_mean(data: Tensor, mask: Tensor | None):
    """Compute mean of data with mask."""
    if mask is None:
        return data.mean(dim=[-2, -1], keepdim=True)
    mask = mask.float()
    mask_sum = torch.sum(mask, dim=[-2, -1], keepdim=True)
    mask_mean = torch.sum(
        data * mask, dim=[-2, -1], keepdim=True
    ) / torch.clamp(mask_sum, min=1.0)
    return mask_mean
