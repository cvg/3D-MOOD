"""3D Grounding DINO Omni3D config."""

from __future__ import annotations

import os
from collections.abc import Sequence

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from opendet3d.data.datasets.omni3d.arkitscenes import ARKitScenes
from opendet3d.data.datasets.omni3d.hypersim import Hypersim
from opendet3d.data.datasets.omni3d.kitti_object import KITTIObject
from opendet3d.data.datasets.omni3d.nuscenes import nuScenes
from opendet3d.data.datasets.omni3d.objectron import Objectron
from opendet3d.data.datasets.omni3d.sunrgbd import SUNRGBD
from opendet3d.data.datasets.omni3d.util import get_dataset_det_map

from .transform import get_test_transforms_cfg, get_train_transforms_cfg


def get_omni3d_dataset_cfg(
    data_root: str,
    datasets: Sequence[str],
    data_backend: None | ConfigDict = None,
    omni3d50: bool = True,
    remove_empty: bool = False,
    with_depth: bool = False,
    cache_as_binary: bool = False,
) -> list[ConfigDict]:
    """Get the dataset configs for Omni3D."""
    if omni3d50:
        cached_dir = os.path.join(data_root, "cache_omni3d50")
    else:
        cached_dir = os.path.join(data_root, "cache")

    dataset_cfg_list = []
    for dataset in datasets:
        if "KITTI" in dataset:
            dataset_obj = KITTIObject
        elif "nuScenes" in dataset:
            dataset_obj = nuScenes
        elif "SUNRGBD" in dataset:
            dataset_obj = SUNRGBD
        elif "ARKitScenes" in dataset:
            dataset_obj = ARKitScenes
        elif "Hypersim" in dataset:
            dataset_obj = Hypersim
        elif "Objectron" in dataset:
            dataset_obj = Objectron
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        det_map = get_dataset_det_map(dataset_name=dataset, omni3d50=omni3d50)

        dataset_cfg = class_config(
            dataset_obj,
            data_backend=data_backend,
            data_root=data_root,
            dataset_name=dataset,
            det_map=det_map,
            with_depth=with_depth,
            remove_empty=remove_empty,
            data_prefix="data",
            cache_as_binary=cache_as_binary,
            cached_file_path=os.path.join(cached_dir, f"{dataset}.pkl"),
        )

        dataset_cfg_list.append(dataset_cfg)

    return dataset_cfg_list


def get_omni3d_train_cfg(
    data_root: str = "data/omni3d",
    train_datasets: Sequence[str] = (
        "KITTI_train",
        "KITTI_val",
        "nuScenes_train",
        "nuScenes_val",
        "Objectron_train",
        "Objectron_val",
        "Hypersim_train",
        "Hypersim_val",
        "SUNRGBD_train",
        "SUNRGBD_val",
        "ARKitScenes_train",
        "ARKitScenes_val",
    ),
    data_backend: None | ConfigDict = None,
    omni3d50: bool = True,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
) -> ConfigDict:
    """Get the train config for Omni3D."""
    train_dataset_cfg = get_omni3d_dataset_cfg(
        data_root=data_root,
        datasets=train_datasets,
        data_backend=data_backend,
        omni3d50=omni3d50,
        remove_empty=True,
        with_depth=True,
        cache_as_binary=cache_as_binary,
    )

    train_preprocess_cfg = get_train_transforms_cfg(shape=shape)

    return class_config(
        DataPipe,
        datasets=train_dataset_cfg,
        preprocess_fn=train_preprocess_cfg,
    )


def get_omni3d_test_cfg(
    data_root: str = "data/omni3d",
    test_datasets: Sequence[str] = (
        "KITTI_test",
        "nuScenes_test",
        "SUNRGBD_test",
        "Hypersim_test",
        "ARKitScenes_test",
        "Objectron_test",
    ),
    data_backend: None | ConfigDict = None,
    omni3d50: bool = True,
    with_depth: bool = False,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
) -> ConfigDict:
    """Get the test config for Omni3D."""
    test_dataset_cfg = get_omni3d_dataset_cfg(
        data_root=data_root,
        datasets=test_datasets,
        data_backend=data_backend,
        omni3d50=omni3d50,
        with_depth=with_depth,
        cache_as_binary=cache_as_binary,
    )

    test_preprocess_cfg = get_test_transforms_cfg(shape=shape)

    return class_config(
        DataPipe, datasets=test_dataset_cfg, preprocess_fn=test_preprocess_cfg
    )
