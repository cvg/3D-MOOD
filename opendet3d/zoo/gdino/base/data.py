"""G-DINO data config."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.pad import PadImages
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)


def get_data_cfg(
    train_datasets: ConfigDict | Sequence[ConfigDict],
    test_datasets: ConfigDict | Sequence[ConfigDict],
    pad_stride: int = 1,
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for COCO detection."""
    data = DataConfig()

    # Train
    train_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages, stride=pad_stride),
            class_config(ToTensor),
        ],
    )

    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_datasets,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    # Test
    test_batchprocess_cfg = class_config(
        compose,
        transforms=[
            class_config(PadImages, stride=pad_stride),
            class_config(ToTensor),
        ],
    )

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=test_batchprocess_cfg,
    )

    return data
