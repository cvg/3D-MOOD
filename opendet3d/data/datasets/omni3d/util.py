"""Omni3D data util."""

from __future__ import annotations

from .arkitscenes import arkitscenes_det_map, omni3d_arkitscenes_det_map
from .hypersim import get_hypersim_det_map, omni3d_hypersim_det_map
from .kitti_object import get_kitti_det_map, omni3d_kitti_det_map
from .nuscenes import nusc_det_map
from .objectron import objectron_det_map
from .sunrgbd import get_sunrgbd_det_map, omni3d_sun_rgbd_det_map

DATASET_ID_MAP = {
    0: "KITTI_train",
    1: "KITTI_val",
    2: "KITTI_test",
    3: "nuScenes_train",
    4: "nuScenes_val",
    5: "nuScenes_test",
    6: "Objectron_train",
    7: "Objectron_val",
    8: "Objectron_test",
    9: "Hypersim_train",
    10: "Hypersim_val",
    11: "Hypersim_test",
    12: "SUNRGBD_train",
    13: "SUNRGBD_val",
    14: "SUNRGBD_test",
    15: "ARKitScenes_train",
    16: "ARKitScenes_val",
    17: "ARKitScenes_test",
}


def get_dataset_det_map(
    dataset_name: str,
    omni3d50: bool = True,
) -> tuple[str, dict[str, int]]:
    """Get the detection map."""
    if "train" in dataset_name:
        split = "train"
    elif "val" in dataset_name:
        split = "val"
    elif "test" in dataset_name:
        split = "test"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if "nuScenes" in dataset_name:
        det_map = nusc_det_map
    elif "KITTI" in dataset_name:
        if omni3d50:
            det_map = omni3d_kitti_det_map
        else:
            det_map = get_kitti_det_map(split)
    elif "Objectron" in dataset_name:
        det_map = objectron_det_map
    elif "SUNRGBD" in dataset_name:
        if omni3d50:
            det_map = omni3d_sun_rgbd_det_map
        else:
            det_map = get_sunrgbd_det_map(split)
    elif "Hypersim" in dataset_name:
        if omni3d50:
            det_map = omni3d_hypersim_det_map
        else:
            det_map = get_hypersim_det_map(split)
    elif "ARKitScenes" in dataset_name:
        det_map = (
            omni3d_arkitscenes_det_map if omni3d50 else arkitscenes_det_map
        )
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return det_map
