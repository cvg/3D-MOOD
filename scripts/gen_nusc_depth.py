"""Generate NuScenes depth map."""

import os
import numpy as np

from tqdm import tqdm
from PIL import Image

from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.nuscenes import NuScenes
from vis4d.data.io.hdf5 import HDF5Backend


def generate_depth_map(
    output_dir: str = "data/nuscenes_depth", depth_scale: float = 256.0
):
    """Generate NuScenes depth map."""
    depth_dir = os.path.join(output_dir, "samples", "CAM_FRONT")

    os.makedirs(depth_dir, exist_ok=True)

    for split in ["train", "val"]:
        nusc = NuScenes(
            data_root="data/nuscenes",
            version="v1.0-trainval",
            split=split,
            keys_to_load=[K.depth_maps],
            sensors=["CAM_FRONT", "LIDAR_TOP"],
            data_backend=HDF5Backend(),
            max_sweeps=1,
            cache_as_binary=True,
            cached_file_path=f"data/nuscenes/{split}.pkl",
        )

        for i, nusc_sample in enumerate(tqdm(nusc.samples)):
            image_file_path = nusc_sample["CAM_FRONT"]["image_path"]

            nusc_data = nusc[i]

            depth = nusc_data["CAM_FRONT"][K.depth_maps]

            numpy_image = (depth * depth_scale).astype(np.uint16)

            image_file_name = image_file_path.split("/")[-1]

            depth_file_path = os.path.join(
                depth_dir, image_file_name.replace("jpg", "png")
            )

            Image.fromarray(numpy_image).save(depth_file_path)

    print("Done.")


if __name__ == "__main__":
    """Generate NuScenes depth map."""
    generate_depth_map()
