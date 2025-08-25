"""Resize transformation."""

from __future__ import annotations

import math

import numpy as np
import torch
from vis4d.common.typing import NDArrayF32, NDArrayI64
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.base import Transform
from vis4d.data.transforms.resize import ResizeParam, resize_tensor


@Transform(K.images, ["transforms.resize", K.input_hw])
class GenResizeParameters:
    """Generate the parameters for a resize operation."""

    def __init__(
        self, shape: tuple[int, int], scales: tuple[float, float] | float = 1.0
    ) -> None:
        """Create a new instance of the class."""
        self.shape = shape
        self.scales = scales

    def __call__(
        self, images: list[NDArrayF32]
    ) -> tuple[list[ResizeParam], list[tuple[int, int]]]:
        """Compute the parameters and put them in the data dict."""
        if isinstance(self.scales, float):
            random_scale = self.scales
        else:
            random_scale = np.random.uniform(self.scales[0], self.scales[1])

        shape = (
            math.ceil(self.shape[0] * random_scale - 0.5),
            math.ceil(self.shape[1] * random_scale - 0.5),
        )

        output_ratio = shape[1] / shape[0]

        image = images[0]

        input_h, input_w = (image.shape[1], image.shape[2])
        input_ratio = input_w / input_h

        if output_ratio > input_ratio:
            scale = shape[0] / input_h
        else:
            scale = shape[1] / input_w

        target_shape = (
            math.ceil(input_h * scale - 0.5),
            math.ceil(input_w * scale - 0.5),
        )

        scale_factor = (target_shape[0] / input_h, target_shape[1] / input_w)

        resize_params = [
            ResizeParam(target_shape=target_shape, scale_factor=scale_factor)
        ] * len(images)
        target_shapes = [target_shape] * len(images)

        return resize_params, target_shapes


@Transform(
    [K.panoptic_masks, "transforms.resize.target_shape"], K.panoptic_masks
)
class ResizePanopticMasks:
    """Resize panoptic segmentation masks."""

    def __call__(
        self,
        masks_list: list[NDArrayI64],
        target_shape_list: list[tuple[int, int]],
    ) -> list[NDArrayI64]:
        """Resize masks."""
        for i, (masks, target_shape) in enumerate(
            zip(masks_list, target_shape_list)
        ):
            masks_ = torch.from_numpy(masks)
            masks_ = (
                resize_tensor(
                    masks_.float().unsqueeze(0).unsqueeze(0),
                    target_shape,
                    interpolation="nearest",
                )
                .type(masks_.dtype)
                .squeeze(0)
                .squeeze(0)
            )
            masks_list[i] = masks_.numpy()
        return masks_list


@Transform([K.boxes3d, "transforms.resize.scale_factor"], K.boxes3d)
class ResizeBoxes3D:
    """Resize list of 2D bounding boxes."""

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        scale_factors: list[tuple[float, float]],
    ) -> list[NDArrayF32]:
        """Resize 2D bounding boxes.

        Args:
            boxes_list: (list[NDArrayF32]): The bounding boxes to be resized.
            scale_factors (list[tuple[float, float]]): scaling factors.

        Returns:
            list[NDArrayF32]: Resized bounding boxes according to parameters in
                resize.
        """
        for i, (boxes, scale_factor) in enumerate(
            zip(boxes_list, scale_factors)
        ):
            boxes[:, 2] /= scale_factor[0]
            boxes_list[i] = boxes
        return boxes_list
