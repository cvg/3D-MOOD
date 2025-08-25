"""KITTI Object from Omni3D.

KITTI Object Labels:
Categories, -, -, alpha, x1, y1, x2, y2, h, w, l, x, botom_y, z, ry

KITTI Object Categories:
{
    "Pedestrian": "pedestrian",
    "Cyclist": "cyclist",
    "Car": "car",
    "Van": "car",
    "Truck": "truck",
    "Tram": "tram",
    "Person": "pedestrian",
    "Person_sitting": "pedestrian",
    "Misc": "misc",
    "DontCare": "dontcare",
}
"""

from __future__ import annotations

import os

from vis4d.common.typing import ArgsType, DictStrAny

from opendet3d.data.datasets.coco3d import COCO3DDataset

from .omni3d_classes import omni3d_class_map

kitti_train_det_map = kitti_test_det_map = {
    "car": 0,
    "cyclist": 1,
    "pedestrian": 2,
    "person": 3,
    "tram": 4,
    "truck": 5,
    "van": 6,
}

kitti_val_det_map = {
    "car": 0,
    "cyclist": 1,
    "pedestrian": 2,
    "tram": 3,
    "truck": 4,
}

# KITTI-Omni3D Mapping
omni3d_kitti_det_map = {
    "pedestrian": 0,
    "car": 1,
    "cyclist": 2,
    "van": 3,
    "truck": 4,
}


def get_kitti_det_map(split: str) -> dict[str, int]:
    """Get the KITTI detection map."""
    assert split in {"train", "val", "test"}, f"Invalid split: {split}"

    if split == "val":
        return kitti_val_det_map

    # Train and Test are the same
    return kitti_train_det_map


class KITTIObject(COCO3DDataset):
    """KITTI Object Dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = omni3d_class_map,
        max_depth: float = 80.0,
        depth_scale: float = 256.0,
        depth_data_root: str = "data/KITTI_object_depth",
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        self.depth_data_root = depth_data_root

        super().__init__(
            class_map=class_map,
            max_depth=max_depth,
            depth_scale=depth_scale,
            **kwargs,
        )

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Get the depth filenames.

        Since not every data has depth.
        """
        _, _, split, image_id, img_filename = img["file_path"].split("/")

        depth_filename = os.path.join(
            self.depth_data_root,
            split,
            image_id,
            img_filename.replace(".jpg", ".png"),
        )

        return depth_filename
