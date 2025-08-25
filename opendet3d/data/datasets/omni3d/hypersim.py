"""Hypersim from Omni3D."""

from __future__ import annotations

import os

from vis4d.common.typing import ArgsType, DictStrAny

from opendet3d.data.datasets.coco3d import COCO3DDataset

from .omni3d_classes import omni3d_class_map

hypersim_train_det_map = {
    "bathtub": 0,
    "bed": 1,
    "blinds": 2,
    "bookcase": 3,
    "books": 4,
    "box": 5,
    "cabinet": 6,
    "chair": 7,
    "clothes": 8,
    "counter": 9,
    "curtain": 10,
    "desk": 11,
    "door": 12,
    "dresser": 13,
    "floor mat": 14,
    "lamp": 15,
    "mirror": 16,
    "night stand": 17,
    "person": 18,
    "picture": 19,
    "pillow": 20,
    "refrigerator": 21,
    "shelves": 22,
    "sink": 23,
    "sofa": 24,
    "stationery": 25,
    "table": 26,
    "television": 27,
    "toilet": 28,
    "towel": 29,
    "window": 30,
}

hypersim_val_det_map = {
    "bathtub": 0,
    "bed": 1,
    "blinds": 2,
    "bookcase": 3,
    "books": 4,
    "box": 5,
    "cabinet": 6,
    "chair": 7,
    "clothes": 8,
    "counter": 9,
    "curtain": 10,
    "desk": 11,
    "door": 12,
    "dresser": 13,
    "floor mat": 14,
    "lamp": 15,
    "mirror": 16,
    "night stand": 17,
    "picture": 18,
    "pillow": 19,
    "refrigerator": 20,
    "shelves": 21,
    "sink": 22,
    "sofa": 23,
    "stationery": 24,
    "table": 25,
    "television": 26,
    "toilet": 27,
    "towel": 28,
    "window": 29,
}

hypersim_test_det_map = {
    "bathtub": 0,
    "bed": 1,
    "blinds": 2,
    "board": 3,
    "bookcase": 4,
    "books": 5,
    "box": 6,
    "cabinet": 7,
    "chair": 8,
    "clothes": 9,
    "counter": 10,
    "curtain": 11,
    "desk": 12,
    "door": 13,
    "floor mat": 14,
    "lamp": 15,
    "mirror": 16,
    "night stand": 17,
    "picture": 18,
    "pillow": 19,
    "refrigerator": 20,
    "shelves": 21,
    "sink": 22,
    "sofa": 23,
    "stationery": 24,
    "table": 25,
    "television": 26,
    "towel": 27,
    "window": 28,
}


omni3d_hypersim_det_map = {
    "books": 0,
    "chair": 1,
    "towel": 2,
    "blinds": 3,
    "window": 4,
    "lamp": 5,
    "shelves": 6,
    "mirror": 7,
    "sink": 8,
    "cabinet": 9,
    "bathtub": 10,
    "door": 11,
    "desk": 12,
    "box": 13,
    "bookcase": 14,
    "picture": 15,
    "table": 16,
    "counter": 17,
    "bed": 18,
    "night stand": 19,
    "pillow": 20,
    "sofa": 21,
    "television": 22,
    "floor mat": 23,
    "curtain": 24,
    "clothes": 25,
    "stationery": 26,
    "refrigerator": 27,
}


def get_hypersim_det_map(split: str) -> dict[str, int]:
    """Get Hypersim detection map."""
    assert split in {"train", "val", "test"}, f"Invalid split: {split}"

    if split == "train":
        return hypersim_train_det_map
    elif split == "val":
        return hypersim_val_det_map
    elif split == "test":
        return hypersim_test_det_map


class Hypersim(COCO3DDataset):
    """Hypersim Dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = omni3d_class_map,
        max_depth: float = 50.0,
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
        _, _, scene, _, img_dir, img_name = img["file_path"].split("/")

        depth_filename = os.path.join(
            "data/hypersim_depth",
            scene,
            "images",
            img_dir,
            img_name.replace("jpg", "png"),
        )

        return depth_filename
