"""Open data config."""

from __future__ import annotations

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.data_pipe import DataPipe

from opendet3d.data.datasets.argoverse import (
    AV2SensorDataset,
    av2_class_map,
    av2_det_map,
)
from opendet3d.data.datasets.scannet import (
    ScanNetDataset,
    scannet200_class_map,
    scannet200_det_map,
    scannet_class_map,
    scannet_det_map,
)

from .transform import get_test_transforms_cfg


def get_scannet_data_cfg(
    data_root: str = "data/scannet",
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    scannet200: bool = False,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
    test_cached_file_path: str = "data/scannet/val.pkl",
) -> ConfigDict:
    """Get the default config for ScanNetV2."""
    if scannet200:
        dataset_name = "ScanNet200_val"
        class_map = scannet200_class_map
        det_map = scannet200_det_map
        test_cached_file_path = test_cached_file_path.replace(
            ".pkl", "_200.pkl"
        )
    else:
        dataset_name = "ScanNet_val"
        class_map = scannet_class_map
        det_map = scannet_det_map

    test_dataset_cfg = class_config(
        ScanNetDataset,
        data_root=data_root,
        dataset_name=dataset_name,
        class_map=class_map,
        det_map=det_map,
        with_depth=with_depth,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=test_cached_file_path,
    )

    test_preprocess_cfg = get_test_transforms_cfg(shape=shape)

    return class_config(
        DataPipe, datasets=test_dataset_cfg, preprocess_fn=test_preprocess_cfg
    )


def get_av2_data_cfg(
    data_root: str = "data/argoverse",
    data_backend: None | ConfigDict = None,
    with_depth: bool = False,
    shape: tuple[int, int] = (800, 1333),
    cache_as_binary: bool = True,
    test_cached_file_path: str = "data/argoverse/val.pkl",
) -> ConfigDict:
    """Get the default config for Argoverse V2."""
    text_prompt_mapping = {
        "regular vehicle": {"name": "car"},
        "bicyclist": {"name": "cyclist"},
        "construction cone": {"name": "traffic cone"},
        "construction barrel": {"name": "barrier"},
        "large vehicle": {"name": "van"},
        "vehicular trailer": {"name": "trailer"},
    }

    test_dataset_cfg = class_config(
        AV2SensorDataset,
        data_root=data_root,
        dataset_name="Argoverse_val",
        class_map=av2_class_map,
        det_map=av2_det_map,
        with_depth=with_depth,
        text_prompt_mapping=text_prompt_mapping,
        data_backend=data_backend,
        cache_as_binary=cache_as_binary,
        cached_file_path=test_cached_file_path,
    )

    test_preprocess_cfg = get_test_transforms_cfg(shape=shape)

    return class_config(
        DataPipe, datasets=test_dataset_cfg, preprocess_fn=test_preprocess_cfg
    )
