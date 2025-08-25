"""Objectron from Omni3D."""

from __future__ import annotations

import os

from vis4d.common.typing import ArgsType, DictStrAny

from opendet3d.data.datasets.coco3d import COCO3DDataset

from .omni3d_classes import omni3d_class_map

objectron_det_map = {
    "bicycle": 0,
    "books": 1,
    "bottle": 2,
    "camera": 3,
    "cereal box": 4,
    "chair": 5,
    "cup": 6,
    "laptop": 7,
    "shoes": 8,
}


class Objectron(COCO3DDataset):
    """Objectron dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = omni3d_class_map,
        max_depth: float = 12.0,
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
        _, _, split, img_name = img["file_path"].split("/")

        depth_filename = os.path.join(
            "data/objectron_depth",
            split,
            img_name.replace(".jpg", "_depth.png"),
        )
        return depth_filename
