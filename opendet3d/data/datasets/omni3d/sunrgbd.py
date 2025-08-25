"""SUN RGB-D from Omni3D."""

from __future__ import annotations

import os

import numpy as np
from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.datasets.util import im_decode

from opendet3d.data.datasets.coco3d import COCO3DDataset

from .omni3d_classes import omni3d_class_map

# Train and Test are sharing the classes
sun_rgbd_train_det_map = sun_rgbd_test_det_map = {
    "air conditioner": 0,
    "bag": 1,
    "bathtub": 2,
    "bed": 3,
    "bicycle": 4,
    "bin": 5,
    "blanket": 6,
    "blinds": 7,
    "board": 8,
    "bookcase": 9,
    "books": 10,
    "bottle": 11,
    "bowl": 12,
    "box": 13,
    "cabinet": 14,
    "cart": 15,
    "chair": 16,
    "clock": 17,
    "closet": 18,
    "clothes": 19,
    "coffee maker": 20,
    "computer": 21,
    "counter": 22,
    "cup": 23,
    "curtain": 24,
    "desk": 25,
    "door": 26,
    "drawers": 27,
    "dresser": 28,
    "electronics": 29,
    "fan": 30,
    "faucet": 31,
    "fire extinguisher": 32,
    "fire place": 33,
    "floor mat": 34,
    "fume hood": 35,
    "glass": 36,
    "keyboard": 37,
    "kitchen pan": 38,
    "ladder": 39,
    "lamp": 40,
    "laptop": 41,
    "machine": 42,
    "microwave": 43,
    "mirror": 44,
    "monitor": 45,
    "mouse": 46,
    "night stand": 47,
    "oven": 48,
    "painting": 49,
    "pen": 50,
    "person": 51,
    "phone": 52,
    "picture": 53,
    "pillow": 54,
    "plates": 55,
    "podium": 56,
    "potted plant": 57,
    "printer": 58,
    "projector": 59,
    "rack": 60,
    "refrigerator": 61,
    "remote": 62,
    "shelves": 63,
    "shoes": 64,
    "shower curtain": 65,
    "sink": 66,
    "sofa": 67,
    "soundsystem": 68,
    "stationery": 69,
    "stove": 70,
    "table": 71,
    "television": 72,
    "tissues": 73,
    "toaster": 74,
    "toilet": 75,
    "toilet paper": 76,
    "towel": 77,
    "toys": 78,
    "tray": 79,
    "utensils": 80,
    "vase": 81,
    "window": 82,
}

sun_rgbd_val_det_map = {
    "air conditioner": 0,
    "bag": 1,
    "bathtub": 2,
    "bed": 3,
    "bin": 4,
    "blanket": 5,
    "blinds": 6,
    "board": 7,
    "bookcase": 8,
    "books": 9,
    "bottle": 10,
    "bowl": 11,
    "box": 12,
    "cabinet": 13,
    "cart": 14,
    "chair": 15,
    "closet": 16,
    "clothes": 17,
    "coffee maker": 18,
    "computer": 19,
    "counter": 20,
    "cup": 21,
    "curtain": 22,
    "desk": 23,
    "door": 24,
    "drawers": 25,
    "dresser": 26,
    "electronics": 27,
    "fan": 28,
    "faucet": 29,
    "fire extinguisher": 30,
    "fire place": 31,
    "fume hood": 32,
    "keyboard": 33,
    "kitchen pan": 34,
    "lamp": 35,
    "laptop": 36,
    "machine": 37,
    "microwave": 38,
    "mirror": 39,
    "monitor": 40,
    "night stand": 41,
    "oven": 42,
    "painting": 43,
    "pen": 44,
    "person": 45,
    "phone": 46,
    "picture": 47,
    "pillow": 48,
    "plates": 49,
    "potted plant": 50,
    "printer": 51,
    "projector": 52,
    "rack": 53,
    "refrigerator": 54,
    "shelves": 55,
    "sink": 56,
    "sofa": 57,
    "soundsystem": 58,
    "stationery": 59,
    "stove": 60,
    "table": 61,
    "television": 62,
    "tissues": 63,
    "toaster": 64,
    "toilet": 65,
    "towel": 66,
    "toys": 67,
    "tray": 68,
    "utensils": 69,
    "vase": 70,
    "window": 71,
}

omni3d_sun_rgbd_det_map = {
    "bicycle": 0,
    "books": 1,
    "bottle": 2,
    "chair": 3,
    "cup": 4,
    "laptop": 5,
    "shoes": 6,
    "towel": 7,
    "blinds": 8,
    "window": 9,
    "lamp": 10,
    "shelves": 11,
    "mirror": 12,
    "sink": 13,
    "cabinet": 14,
    "bathtub": 15,
    "door": 16,
    "toilet": 17,
    "desk": 18,
    "box": 19,
    "bookcase": 20,
    "picture": 21,
    "table": 22,
    "counter": 23,
    "bed": 24,
    "night stand": 25,
    "pillow": 26,
    "sofa": 27,
    "television": 28,
    "floor mat": 29,
    "curtain": 30,
    "clothes": 31,
    "stationery": 32,
    "refrigerator": 33,
    "bin": 34,
    "stove": 35,
    "oven": 36,
    "machine": 37,
}


def get_sunrgbd_det_map(split: str) -> dict[str, int]:
    """Get the SUN RGB-D detection map."""
    assert split in {"train", "val", "test"}, f"Invalid split: {split}"

    if split == "train":
        return sun_rgbd_train_det_map
    elif split == "val":
        return sun_rgbd_val_det_map
    else:
        return sun_rgbd_test_det_map


class SUNRGBD(COCO3DDataset):
    """SUN RGB-D Dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = omni3d_class_map,
        max_depth: float = 8.0,
        depth_scale: float = 1000.0,
        **kwargs: ArgsType,
    ) -> None:
        """Initialize SUN RGB-D dataset."""
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
        img["file_path"] = img["file_path"].replace("//", "/")

        data_dir = img["file_path"].split("/image")[0]

        depth_files = self.data_backend.listdir(
            os.path.join(data_dir, "depth")
        )
        assert len(depth_files) == 1

        depth_filename = os.path.join(data_dir, "depth", depth_files[0])

        return depth_filename

    def get_depth_map(self, sample: DictStrAny) -> np.ndarray:
        """Get the depth map."""
        depth_bytes = self.data_backend.get(sample["depth_filename"])
        depth_array = im_decode(depth_bytes)

        depth_array = depth_array >> 3 | depth_array << (16 - 3)

        depth = np.ascontiguousarray(depth_array, dtype=np.float32)

        depth = depth / self.depth_scale

        return depth
