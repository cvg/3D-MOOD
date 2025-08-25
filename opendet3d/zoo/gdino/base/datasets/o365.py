"""Grounding DINO Objects365 config."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe
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

from opendet3d.data.datasets.odvg import ODVGDataset
from opendet3d.data.transforms.language import RandomSamplingNegPos


def get_objects365v1_detection_train_cfg(
    data_root: str = "data/objects365v1",
    ann_file="o365v1_train_odvg.json",
    label_map_file="o365v1_label_map.json",
    dataset_prefix="train",
    data_backend: None | ConfigDict = None,
    cache_as_binary: bool = True,
    train_cached_file_path: str | None = "data/objects365v1/train.pkl",
) -> ConfigDict:
    """Get the DataPipe config for Objects365v1 detection train set."""
    dataset_cfg = class_config(
        ODVGDataset,
        data_root=data_root,
        ann_file=ann_file,
        label_map_file=label_map_file,
        dataset_prefix=dataset_prefix,
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

    preprocess_transforms.append(class_config(RandomSamplingNegPos))

    preprocess_transforms.append(class_config(NormalizeImages))

    preprocess_fn_cfg = class_config(compose, transforms=preprocess_transforms)

    return class_config(
        DataPipe, datasets=dataset_cfg, preprocess_fn=preprocess_fn_cfg
    )
