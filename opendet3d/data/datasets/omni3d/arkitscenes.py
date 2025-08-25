"""ARKitScenes from Omni3D."""

from __future__ import annotations

import os

from vis4d.common.typing import ArgsType, DictStrAny

from opendet3d.data.datasets.coco3d import COCO3DDataset

from .omni3d_classes import omni3d_class_map

arkitscenes_det_map = {
    "bathtub": 0,
    "bed": 1,
    "cabinet": 2,
    "chair": 3,
    "fireplace": 4,
    "machine": 5,
    "oven": 6,
    "refrigerator": 7,
    "shelves": 8,
    "sink": 9,
    "sofa": 10,
    "stove": 11,
    "table": 12,
    "television": 13,
    "toilet": 14,
}

omni3d_arkitscenes_det_map = {
    "table": 0,
    "bed": 1,
    "sofa": 2,
    "television": 3,
    "refrigerator": 4,
    "chair": 5,
    "oven": 6,
    "machine": 7,
    "stove": 8,
    "shelves": 9,
    "sink": 10,
    "cabinet": 11,
    "bathtub": 12,
    "toilet": 13,
}


class ARKitScenes(COCO3DDataset):
    """ARKitScenes Dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = omni3d_class_map,
        max_depth: float = 10.0,
        depth_scale: float = 1000.0,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
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
        _, _, split, video_id, image_name = img["file_path"].split("/")

        depth_filename = os.path.join(
            "data/ARKitScenes_depth",
            split,
            video_id,
            image_name.replace("jpg", "png"),
        )

        return depth_filename
