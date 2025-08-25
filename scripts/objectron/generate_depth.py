"""Generate depth images from the Objectron dataset."""

from __future__ import annotations

import json
import os
import struct
from typing import Any, Dict

from functools import partial

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm

# The AR Metadata captured with each frame in the video
from a_r_capture_metadata_pb2 import ARFrame
from util import pmap

DictStrAny = Dict[str, Any]  # type: ignore


def get_geometry_data(geometry_filename):
    sequence_geometry = []
    with open(geometry_filename, "rb") as pb:
        proto_buf = pb.read()

        i = 0
        while i < len(proto_buf):
            # Read the first four Bytes in little endian '<' integers 'I' format
            # indicating the length of the current message.
            msg_len = struct.unpack("<I", proto_buf[i : i + 4])[0]
            i += 4
            message_buf = proto_buf[i : i + msg_len]
            i += msg_len
            frame_data = ARFrame()
            frame_data.ParseFromString(message_buf)

            transform = np.reshape(frame_data.camera.transform, (4, 4))
            projection = np.reshape(
                frame_data.camera.projection_matrix, (4, 4)
            )
            view = np.reshape(frame_data.camera.view_matrix, (4, 4))

            current_points = [
                np.array([v.x, v.y, v.z])
                for v in frame_data.raw_feature_points.point
            ]
            current_points = np.array(current_points)

            sequence_geometry.append(
                (transform, projection, view, current_points)
            )
    return sequence_geometry


def project_points(points, projection_matrix, view_matrix, width, height):
    p_3d = np.concatenate((points, np.ones_like(points[:, :1])), axis=-1).T
    p_3d_cam = np.matmul(view_matrix, p_3d)
    p_2d_proj = np.matmul(projection_matrix, p_3d_cam)

    # Project the points
    depth = -p_3d_cam.T[:, 2]

    p_2d_ndc = p_2d_proj[:-1, :] / p_2d_proj[-1, :]
    p_2d_ndc = p_2d_ndc.T

    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    x = p_2d_ndc[:, 1]
    y = p_2d_ndc[:, 0]
    pixels = np.copy(p_2d_ndc)
    pixels[:, 0] = ((1 + x) * 0.5) * width
    pixels[:, 1] = ((1 + y) * 0.5) * height
    pixels[:, 2] = depth
    return pixels


def parse_video(omni3d_data_root: str = "data/omni3d") -> DictStrAny:
    print("Parsing Omni3D Objectron Video...")
    videos = {}
    for dataset in ["Objectron_train", "Objectron_val", "Objectron_test"]:
        annotation = os.path.join(
            omni3d_data_root, "annotations", f"{dataset}.json"
        )

        print(f"Loading {dataset}...")

        with open(annotation, "r") as file:
            samples = json.load(file)

        for img in tqdm(samples["images"]):
            _, split, img_name = img["file_path"].split("/")

            obj_name, res = img_name.split("_batch_")
            batch_id, vid, frame_id = res.split("_")
            frame_id = int(frame_id.split(".")[0])

            video_id = f"{obj_name}/batch-{batch_id}/{vid}"
            video_name = f"{obj_name}_batch-{batch_id}_{vid}"

            if video_name not in videos:
                videos[video_name] = {
                    "video_id": video_id,
                    "splits": [split],
                    "img_names": [img_name],
                    "frame_ids": [frame_id],
                    "height": img["height"],
                    "width": img["width"],
                }
            else:
                videos[video_name]["splits"].append(split)
                videos[video_name]["img_names"].append(img_name)
                videos[video_name]["frame_ids"].append(frame_id)

    return videos


def download_videos(
    video,
    public_url: str = "https://storage.googleapis.com/objectron",
    output_dir: str = "data/objectron_video",
) -> None:
    """Download videos from the Objectron dataset."""
    video_name, video_info = video
    video_id = video_info["video_id"]

    metadata_path = os.path.join(output_dir, f"{video_name}.geometry.pbdata")

    if not os.path.exists(metadata_path):
        metadata_filename = os.path.join(
            public_url,
            "videos",
            video_id,
            "geometry.pbdata",
        )
        metadata = requests.get(metadata_filename)

        with open(metadata_path, "wb") as f:
            f.write(metadata.content)


def download(videos):
    print("Downloading Objectron Videos...")
    func = partial(download_videos)

    video_names = [video_name for video_name in videos]
    video_infos = [info for _, info in videos.items()]

    _ = pmap(
        func,
        zip(video_names, video_infos),
        max_len=len(video_names) // 4 + 1,
        nprocs=4,
    )

    print("Done.")


def generate_depth(
    videos: DictStrAny,
    objectron_data_root: str = "data/objectron_depth",
    max_depth: float = 12.0,
    depth_scale: float = 1000.0,
    output_dir: str = "data/objectron_video",
) -> None:
    """Generate Objectron depth."""
    print("Generating Depth Map...")
    for video_name, video_info in tqdm(videos.items()):
        frame_ids = video_info["frame_ids"]

        sequence_geometry = get_geometry_data(
            os.path.join(output_dir, f"{video_name}.geometry.pbdata")
        )

        for i, frame_id in enumerate(frame_ids):
            save_dir = os.path.join(
                objectron_data_root, video_info["splits"][i], "depth"
            )
            os.makedirs(save_dir, exist_ok=True)

            depth_file_path = os.path.join(
                save_dir,
                video_info["img_names"][i].replace(".jpg", "_depth.png"),
            )

            if os.path.exists(depth_file_path):
                continue

            height, width = video_info["height"], video_info["width"]

            # First, let's grab the point-cloud from the geometry metadata
            _, projection, view, scene_points_3d = sequence_geometry[frame_id]

            # Project the 3D points as 2D depth maps
            if len(scene_points_3d) == 0:
                print(f"Empty scene points for {video_info['img_names'][i]}.")
                continue

            scene_points_2d = project_points(
                scene_points_3d, projection, view, width, height
            )

            depth = np.zeros((height, width))

            invalid_points = 0
            for point in scene_points_2d:
                try:
                    # Swap X and Y.
                    depth[int(point[1]), int(point[0])] = point[2]
                except:
                    invalid_points += 1

            if invalid_points == len(scene_points_2d):
                print(
                    f"All points are invalid for {video_info['img_names'][i]}."
                )
                continue

            depth = np.clip(depth, a_min=0.0, a_max=max_depth)

            if (depth > 0).sum() == 0:
                print(
                    f"All points are invalid for {video_info['img_names'][i]}."
                )
                continue

            numpy_image = np.clip(
                depth * depth_scale,
                a_min=0,
                a_max=2**16 - 1,
            ).astype(np.uint16)

            Image.fromarray(numpy_image).save(depth_file_path)

    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    """Download Objectron videos and generate depth images."""
    videos = parse_video()
    download(videos=videos)
    generate_depth(videos=videos)
