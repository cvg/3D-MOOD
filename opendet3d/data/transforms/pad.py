"""Pad transformation."""

from __future__ import annotations

from typing import TypedDict

import torch
import torch.nn.functional as F
from vis4d.common.typing import NDArrayF32
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.base import Transform
from vis4d.data.transforms.pad import _get_max_shape


class PadParam(TypedDict):
    """Parameters for Reshape."""

    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


@Transform(
    [K.images, K.input_hw],
    [K.images, "transforms.pad", K.input_hw, "padding"],
)
class CenterPadImages:
    """Pad batch of images at the bottom right."""

    def __init__(
        self,
        stride: int = 32,
        mode: str = "constant",
        value: float = 0.0,
        update_input_hw: bool = False,
        shape: tuple[int, int] | None = None,
        pad2square: bool = False,
    ) -> None:
        """Creates an instance of PadImage.

        Args:
            stride (int, optional): Chooses padding size so that the input will
                be divisible by stride. Defaults to 32.
            mode (str, optional): Padding mode. One of constant, reflect,
                replicate or circular. Defaults to "constant".
            value (float, optional): Value for constant padding.
                Defaults to 0.0.
            shape (tuple[int, int], optional): Shape of the padded image
                (H, W). Defaults to None.
            pad2square (bool, optional): Pad to square. Defaults to False.
        """
        self.stride = stride
        self.mode = mode
        self.value = value
        self.update_input_hw = update_input_hw
        self.shape = shape
        self.pad2square = pad2square

    def __call__(
        self, images: list[NDArrayF32], input_hw: list[tuple[int, int]]
    ) -> tuple[list[NDArrayF32], list[PadParam], list[tuple[int, int]]]:
        """Pad images to consistent size."""
        heights = [im.shape[1] for im in images]
        widths = [im.shape[2] for im in images]

        max_hw = _get_max_shape(
            heights, widths, self.stride, self.shape, self.pad2square
        )

        # generate params for torch pad
        pad_params = []
        target_input_hw = []
        paddings = []
        for i, (image, h, w) in enumerate(zip(images, heights, widths)):
            pad_top, pad_bottom = (max_hw[0] - h) // 2, max_hw[0] - h - (
                max_hw[0] - h
            ) // 2

            pad_left, pad_right = (max_hw[1] - w) // 2, max_hw[1] - w - (
                max_hw[1] - w
            ) // 2

            image_ = torch.from_numpy(image).permute(0, 3, 1, 2)
            image_ = F.pad(
                image_,
                (pad_left, pad_right, pad_top, pad_bottom),
                self.mode,
                self.value,
            )
            images[i] = image_.permute(0, 2, 3, 1).numpy()

            pad_params.append(
                PadParam(
                    pad_top=pad_top,
                    pad_bottom=pad_bottom,
                    pad_left=pad_left,
                    pad_right=pad_right,
                )
            )

            paddings.append([pad_left, pad_right, pad_top, pad_bottom])

            target_input_hw.append(max_hw)

        if self.update_input_hw:
            input_hw = target_input_hw

        return images, pad_params, input_hw, paddings


@Transform([K.intrinsics, "transforms.pad"], K.intrinsics)
class CenterPadIntrinsics:
    """Resize Intrinsics."""

    def __call__(
        self, intrinsics: list[NDArrayF32], pad_params: list[PadParam]
    ) -> list[NDArrayF32]:
        """Scale camera intrinsics when resizing."""
        for i, intrinsic in enumerate(intrinsics):
            intrinsic[0, 2] += pad_params[i]["pad_left"]
            intrinsic[1, 2] += pad_params[i]["pad_top"]

            intrinsics[i] = intrinsic
        return intrinsics


@Transform([K.boxes2d, "transforms.pad"], K.boxes2d)
class CenterPadBoxes2D:
    """Pad batch of depth maps at the bottom right."""

    def __call__(
        self, boxes_list: list[NDArrayF32], pad_params: list[PadParam]
    ) -> list[NDArrayF32]:
        """Scale camera intrinsics when resizing."""
        for i, boxes in enumerate(boxes_list):
            boxes[:, 0] += pad_params[i]["pad_left"]
            boxes[:, 1] += pad_params[i]["pad_top"]
            boxes[:, 2] += pad_params[i]["pad_left"]
            boxes[:, 3] += pad_params[i]["pad_top"]

            boxes_list[i] = boxes

        return boxes_list


@Transform([K.depth_maps, "transforms.pad"], K.depth_maps)
class CenterPadDepthMaps:
    """Pad batch of depth maps at the bottom right."""

    def __init__(self, mode: str = "constant", value: int = 0) -> None:
        """Creates an instance."""
        self.mode = mode
        self.value = value

    def __call__(
        self, depth_maps: list[NDArrayF32], pad_params: list[PadParam]
    ) -> list[NDArrayF32]:
        """Pad images to consistent size."""

        # generate params for torch pad
        for i, (depth, pad_param_dict) in enumerate(
            zip(depth_maps, pad_params)
        ):
            pad_param = (
                pad_param_dict["pad_left"],
                pad_param_dict["pad_right"],
                pad_param_dict["pad_top"],
                pad_param_dict["pad_bottom"],
            )

            depth_ = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            depth_ = F.pad(depth_, pad_param, self.mode, self.value)
            depth_maps[i] = depth_.squeeze(0).squeeze(0).numpy()

        return depth_maps
