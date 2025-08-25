"""Generate depth images from the ARKitScene dataset."""

import os
import json
import cv2
import pandas as pd
from tqdm import tqdm

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from download_data import download_data


def rotate_image(img, direction):
    if direction == "Up":
        pass
    elif direction == "Left":
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif direction == "Right":
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif direction == "Down":
        img = cv2.rotate(img, cv2.ROTATE_180)
    else:
        raise Exception(f"No such direction (={direction}) rotation")
    return img


def generate_depth(
    omni3d_data_root: str = "data/omni3d",
    data_dir: str = "data/ARKitScenes",
    target_data_dir: str = "data/ARKitScenes_depth",
    depth_scale: float = 1000.0,
) -> None:
    """Generate depth for ARKitScenes dataset."""
    os.makedirs(target_data_dir, exist_ok=True)

    meta_data = None
    for dataset in [
        "ARKitScenes_train",
        "ARKitScenes_val",
        "ARKitScenes_test",
    ]:
        print(f"Parsing {dataset}...")

        video_ids = []
        lowres_samples = {}
        not_found = 0
        not_found_depths = []

        annotation = os.path.join(
            omni3d_data_root, "annotations", f"{dataset}.json"
        )

        with open(annotation, "r") as file:
            samples = json.load(file)

        for img in tqdm(samples["images"]):
            _, split, video_id, img_name = img["file_path"].split("/")

            if not video_id in video_ids:
                download_data(
                    "3dod",
                    [video_id],
                    [split],
                    data_dir,
                    keep_zip=False,
                    raw_dataset_assets=None,
                    should_download_laser_scanner_point_cloud=None,
                )
                video_ids.append(video_id)

            if meta_data is None:
                meta_data = pd.read_csv(
                    os.path.join(data_dir, "3dod", "metadata.csv")
                )

                sky_directions = {}
                for vid, sky_direction in zip(
                    meta_data["video_id"], meta_data["sky_direction"]
                ):
                    sky_directions[vid] = sky_direction

            depth_dir = os.path.join(
                data_dir,
                "3dod",
                split,
                video_id,
                f"{video_id}_frames",
                "lowres_depth",
            )

            if not video_id in lowres_samples:
                lowres_samples[video_id] = [
                    f"{float(f.split('_')[1].replace('.png', '')):.3f}"
                    for f in os.listdir(depth_dir)
                    if f.endswith(".png")
                ]

            sample_time = img_name.split("_")[0]

            if not sample_time in lowres_samples[video_id]:
                if (
                    f"{float(sample_time) - 0.001:.3f}"
                    in lowres_samples[video_id]
                ):
                    sample_time = f"{float(sample_time) - 0.001:.3f}"
                elif (
                    f"{float(sample_time) + 0.001:.3f}"
                    in lowres_samples[video_id]
                ):
                    sample_time = f"{float(sample_time) + 0.001:.3f}"
                else:
                    not_found += 1
                    not_found_depths.append(img["file_path"])

            depth_image = cv2.imread(
                os.path.join(depth_dir, f"{video_id}_{sample_time}.png"),
                cv2.IMREAD_UNCHANGED,
            )

            sky_direction = sky_directions[int(video_id)]

            depth_image = rotate_image(depth_image, sky_direction)

            depth = depth_image.astype(np.float32) / depth_scale

            depth = F.interpolate(
                torch.from_numpy(depth)[None, None],
                (img["height"], img["width"]),
                mode="nearest",
                align_corners=None,
                antialias=False,
            ).numpy()[0, 0]

            if depth.max() > 10.0:
                print(f"Depth max: {depth.max()}")

            numpy_image = np.clip(
                np.clip(depth, a_min=0.0, a_max=10.0) * depth_scale,
                a_min=0,
                a_max=2**16 - 1,
            ).astype(np.uint16)

            depth_folder = os.path.join(target_data_dir, split, video_id)
            os.makedirs(depth_folder, exist_ok=True)

            depth_file_path = os.path.join(
                depth_folder, img_name.replace("jpg", "png")
            )

            Image.fromarray(numpy_image).save(depth_file_path)

        print(f"Samples not found: {not_found}")
        for f in not_found_depths:
            print(f)


if __name__ == "__main__":  # pragma: no cover
    generate_depth()
