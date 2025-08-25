"""Generate depth for Hypersim."""

import os
import json
import h5py
import shutil
from functools import partial

import numpy as np
import numpy.typing as npt

from PIL import Image

from vis4d.common.typing import NDArrayF32

from opendet3d.common.parallel import pmap


def convert_hypersim_depth(
    intWidth: int,
    intHeight: int,
    fltFocal: float,
    npyDistance: npt.NDArray[np.float16],
) -> NDArrayF32:
    """Convert camera distance to planar depth.

    Depth_meters images contain Euclidean distances (in meters) to the optical
    center of the camera (perhaps a better name for these images would be
    distance_from_camera_meters). In other words, these images do not contain
    planar depth values, i.e., negative z-coordinates in camera-space.
    """
    npyImageplaneX = (
        np.linspace((-0.5 * intWidth) + 0.5, (0.5 * intWidth) - 0.5, intWidth)
        .reshape(1, intWidth)
        .repeat(intHeight, 0)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneY = (
        np.linspace(
            (-0.5 * intHeight) + 0.5, (0.5 * intHeight) - 0.5, intHeight
        )
        .reshape(intHeight, 1)
        .repeat(intWidth, 1)
        .astype(np.float32)[:, :, None]
    )
    npyImageplaneZ = np.full([intHeight, intWidth, 1], fltFocal, np.float32)
    npyImageplane = np.concatenate(
        [npyImageplaneX, npyImageplaneY, npyImageplaneZ], 2
    )

    return npyDistance / np.linalg.norm(npyImageplane, 2, 2) * fltFocal


def parse_depth(
    images,
    hypersim_data_root: str = "data/hypersim",
    target_data_root: str = "data/hypersim_depth",
    depth_scale: float = 1000.0,
):
    """Parse sequence of images."""
    for img in images:
        _, scene, _, img_dir, img_name = img["file_path"].split("/")

        target_dir = os.path.join(target_data_root, scene, "images")
        os.makedirs(target_dir, exist_ok=True)

        cam = img_dir.split("_")[2]
        frame_id = img_name.split(".")[1]

        # Depth
        save_dir = os.path.join(target_data_root, scene, "images", img_dir)
        os.makedirs(save_dir, exist_ok=True)

        depth_file_path = os.path.join(
            save_dir, img_name.replace("jpg", "png")
        )

        if os.path.exists(depth_file_path):
            continue

        depth_filename = os.path.join(
            hypersim_data_root,
            scene,
            "images",
            f"scene_cam_{cam}_geometry_hdf5",
            f"frame.{frame_id}.depth_meters.hdf5",
        )

        hdf5_file = h5py.File(depth_filename, "r")

        distance_from_camera = np.array(hdf5_file["dataset"])

        depth = convert_hypersim_depth(
            img["width"],
            img["height"],
            img["K"][0][0],
            distance_from_camera,
        )

        numpy_image = np.clip(
            np.clip(depth, a_min=0.0, a_max=50.0) * depth_scale,
            a_min=0,
            a_max=2**16 - 1,
        ).astype(np.uint16)

        Image.fromarray(numpy_image).save(depth_file_path)


def convert_dataset(omni3d_data_root: str = "data/omni3d") -> None:
    """Convert Hypersim depth to planar depth."""
    for dataset in ["Hypersim_train", "Hypersim_val", "Hypersim_test"]:
        annotation = os.path.join(
            omni3d_data_root, "annotations", f"{dataset}.json"
        )

        print(f"Loading {dataset}...")

        with open(annotation, "r") as file:
            hypersim_samples = json.load(file)

        func = partial(parse_depth)

        _ = pmap(
            func,
            zip(hypersim_samples["images"]),
            max_len=len(hypersim_samples["images"]) // 4 + 1,
            nprocs=4,
        )

    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    convert_dataset()
