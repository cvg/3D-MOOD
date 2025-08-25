"""3D-MOOD data config."""

from __future__ import annotations

from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.config.typing import DataConfig
from vis4d.data.data_pipe import DataPipe
from vis4d.data.transforms.base import compose
from vis4d.data.transforms.to_tensor import ToTensor
from vis4d.zoo.base import (
    get_inference_dataloaders_cfg,
    get_train_dataloader_cfg,
)


def get_data_cfg(
    train_datasets: ConfigDict | Sequence[ConfigDict],
    test_datasets: ConfigDict | Sequence[ConfigDict],
    samples_per_gpu: int = 2,
    workers_per_gpu: int = 2,
) -> DataConfig:
    """Get the default config for COCO detection."""
    data = DataConfig()

    # Train
    train_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    data.train_dataloader = get_train_dataloader_cfg(
        datasets_cfg=train_datasets,
        batchprocess_cfg=train_batchprocess_cfg,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
    )

    # Test
    test_batchprocess_cfg = class_config(
        compose, transforms=[class_config(ToTensor)]
    )

    if isinstance(test_datasets, list):
        test_datasets_cfg = class_config(DataPipe, datasets=test_datasets)
    else:
        test_datasets_cfg = test_datasets

    data.test_dataloader = get_inference_dataloaders_cfg(
        datasets_cfg=test_datasets_cfg,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        batchprocess_cfg=test_batchprocess_cfg,
    )

    return data
