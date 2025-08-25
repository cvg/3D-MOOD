"""Resize and center padding transformation config."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.crop import (
    CropBoxes2D,
    CropDepthMaps,
    CropImages,
    CropIntrinsics,
    GenCentralCropParameters,
)
from vis4d.data.transforms.flip import (
    FlipBoxes2D,
    FlipBoxes3D,
    FlipDepthMaps,
    FlipImages,
    FlipIntrinsics,
)
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.resize import (
    ResizeBoxes2D,
    ResizeDepthMaps,
    ResizeImages,
    ResizeIntrinsics,
)

from opendet3d.data.transforms.crop import CropBoxes3D
from opendet3d.data.transforms.pad import (
    CenterPadBoxes2D,
    CenterPadDepthMaps,
    CenterPadImages,
    CenterPadIntrinsics,
)
from opendet3d.data.transforms.resize import GenResizeParameters


def get_train_transforms_cfg(
    shape: tuple[int, int] = (800, 1333)
) -> ConfigDict:
    """Get train data transforms."""
    preprocess_transforms = [
        class_config(GenResizeParameters, shape=shape, scales=[0.75, 1.25]),
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
        class_config(ResizeIntrinsics),
        class_config(ResizeDepthMaps),
    ]

    # Center Crop
    preprocess_transforms += [
        class_config(GenCentralCropParameters, shape=shape),
        class_config(CropImages),
        class_config(CropBoxes2D),
        class_config(CropBoxes3D),
        class_config(CropIntrinsics),
        class_config(CropDepthMaps),
    ]

    flip_transforms = [
        class_config(FlipImages),
        class_config(FlipIntrinsics),
        class_config(FlipBoxes2D),
        class_config(FlipBoxes3D),
        class_config(FlipDepthMaps),
    ]

    preprocess_transforms.append(
        class_config(RandomApply, transforms=flip_transforms, probability=0.5)
    )

    preprocess_transforms.append(class_config(NormalizeImages))

    # Center Pad
    preprocess_transforms += [
        class_config(
            CenterPadImages,
            stride=1,
            shape=shape,
            update_input_hw=True,
        ),
        class_config(CenterPadBoxes2D),
        class_config(CenterPadIntrinsics),
        class_config(CenterPadDepthMaps),
    ]

    train_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    return train_preprocess_cfg


def get_test_transforms_cfg(
    shape: tuple[int, int] = (800, 1333)
) -> ConfigDict:
    """Get test data transforms."""
    preprocess_transforms = [
        class_config(GenResizeParameters, shape=shape),
        class_config(ResizeImages),
        class_config(ResizeIntrinsics),
    ]

    preprocess_transforms.append(class_config(NormalizeImages))

    preprocess_transforms += [
        class_config(
            CenterPadImages, stride=1, shape=shape, update_input_hw=True
        ),
        class_config(CenterPadIntrinsics),
    ]

    test_preprocess_cfg = class_config(
        compose, transforms=preprocess_transforms
    )

    return test_preprocess_cfg
