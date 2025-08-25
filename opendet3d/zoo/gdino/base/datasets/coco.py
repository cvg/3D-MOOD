"""Grounding DINO COCO config."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.data.data_pipe import DataPipe
from vis4d.data.datasets.coco import COCO
from vis4d.data.transforms.base import RandomApply, compose
from vis4d.data.transforms.crop import (
    CropBoxes2D,
    CropImages,
    GenCropParameters,
)
from vis4d.data.transforms.flip import FlipBoxes2D, FlipImages
from vis4d.data.transforms.normalize import NormalizeImages
from vis4d.data.transforms.resize import (
    GenResizeParameters,
    ResizeBoxes2D,
    ResizeImages,
)


def get_coco_detection_train_cfg(
    data_root: str = "data/coco",
    train_split: str = "train2017",
    train_keys_to_load: Sequence[str] = (
        K.images,
        K.original_images,
        K.boxes2d,
        K.boxes2d_classes,
    ),
    data_backend: None | ConfigDict = None,
    cache_as_binary: bool = True,
    train_cached_file_path: str | None = "data/coco/train.pkl",
) -> ConfigDict:
    """Get the DataPipe config for COCO detection train set."""
    dataset_cfg = class_config(
        COCO,
        keys_to_load=train_keys_to_load,
        data_root=data_root,
        split=train_split,
        remove_empty=True,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=train_cached_file_path,
    )

    preprocess_transforms = [
        class_config(
            RandomApply,
            transforms=[
                class_config(
                    GenResizeParameters,
                    shape=[
                        (400, 4200),
                        (500, 4200),
                        (600, 4200),
                    ],
                    multiscale_mode="list",
                    keep_ratio=True,
                    align_long_edge=True,
                ),
                class_config(ResizeImages),
                class_config(ResizeBoxes2D),
                class_config(
                    GenCropParameters,
                    shape=(384, 600),
                    allow_empty_crops=False,
                ),
                class_config(CropImages),
                class_config(CropBoxes2D),
            ],
            probability=0.5,
        )
    ]

    preprocess_transforms += [
        class_config(
            GenResizeParameters,
            shape=[
                (480, 1333),
                (512, 1333),
                (544, 1333),
                (576, 1333),
                (608, 1333),
                (640, 1333),
                (672, 1333),
                (704, 1333),
                (736, 1333),
                (768, 1333),
                (800, 1333),
            ],
            multiscale_mode="list",
            keep_ratio=True,
            align_long_edge=True,
        ),
        class_config(ResizeImages),
        class_config(ResizeBoxes2D),
    ]

    preprocess_transforms.append(
        class_config(
            RandomApply,
            transforms=[class_config(FlipImages), class_config(FlipBoxes2D)],
            probability=0.5,
        )
    )

    preprocess_transforms.append(class_config(NormalizeImages))

    preprocess_fn_cfg = class_config(compose, transforms=preprocess_transforms)

    return class_config(
        DataPipe, datasets=dataset_cfg, preprocess_fn=preprocess_fn_cfg
    )


def get_coco_detection_test_cfg(
    data_root: str = "data/coco",
    test_split: str = "val2017",
    test_keys_to_load: Sequence[str] = (
        K.images,
        K.original_images,
        K.boxes2d,
        K.boxes2d_classes,
    ),
    test_cached_file_path: str | None = "data/coco/val.pkl",
    cache_as_binary: bool = True,
    data_backend: None | ConfigDict = None,
) -> ConfigDict:
    """Get the DataPipe config for COCO detection test set."""
    dataset_cfg = class_config(
        COCO,
        keys_to_load=test_keys_to_load,
        data_root=data_root,
        split=test_split,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=test_cached_file_path,
    )

    preprocess_transforms = [
        class_config(
            GenResizeParameters,
            shape=(800, 1333),
            keep_ratio=True,
            align_long_edge=True,
            fixed_scale=True,
        ),
        class_config(ResizeImages),
    ]

    preprocess_transforms.append(class_config(NormalizeImages))

    preprocess_fn_cfg = class_config(compose, transforms=preprocess_transforms)

    return class_config(
        DataPipe, datasets=dataset_cfg, preprocess_fn=preprocess_fn_cfg
    )
