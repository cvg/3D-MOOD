"""Convert Argoverse dataset for Vis4D.

There are two ego poses:
    1. Ego to City for lidar timestamp: ego_lidar_to_city
    2. Ego to City for camera timestamp: ego_cam_to_city

NOTE: The lidar sweep is provided in the ego reference frame, while they
provide up and down lidar ego pose as well.

Due to the sampling frequency of different sensors (10 Hz for LiDAR,
20 Hz for Cameara), we need to synchronize the data on our own by finding the
nearest timestamp.

The annotations are provided in the ego coordinate in lidar timestamp.
"""

import os
import json
import numpy as np
import pandas as pd
import shutil

from PIL import Image
from pyarrow import feather

import torch

from tqdm import tqdm

from vis4d.common.typing import NDArrayF32
from vis4d.data.const import AxisMode
from vis4d.op.box.box2d import bbox_intersection
from vis4d.op.box.box3d import (
    boxes3d_to_corners,
    boxes3d_in_image,
    transform_boxes3d,
)
from vis4d.op.geometry.rotation import quaternion_to_matrix
from vis4d.op.geometry.transform import (
    transform_points,
    inverse_rigid_transform,
)
from vis4d.op.geometry.projection import generate_depth_map, project_points

from opendet3d.data.datasets.argoverse import (
    av2_class_map,
    VAL_SAMPLE_RATE,
    ACC_FRAMES,
)
from opendet3d.op.box.box2d import compute_visibility_mask

from split import VAL


class_names = [
    "REGULAR_VEHICLE",
    "PEDESTRIAN",
    "BICYCLIST",
    "MOTORCYCLIST",
    "WHEELED_RIDER",
    "BOLLARD",
    "CONSTRUCTION_CONE",
    "SIGN",
    "CONSTRUCTION_BARREL",
    "STOP_SIGN",
    "MOBILE_PEDESTRIAN_CROSSING_SIGN",
    "LARGE_VEHICLE",
    "BUS",
    "BOX_TRUCK",
    "TRUCK",
    "VEHICULAR_TRAILER",
    "TRUCK_CAB",
    "SCHOOL_BUS",
    "ARTICULATED_BUS",
    "MESSAGE_BOARD_TRAILER",
    "BICYCLE",
    "MOTORCYCLE",
    "WHEELED_DEVICE",
    "WHEELCHAIR",
    "STROLLER",
    "DOG",
]


def read_feather(
    file_path: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """Read Apache Feather data from a .feather file.

    AV2 uses .feather to serialize much of its data. This function handles the deserialization
    process and returns a `pandas` DataFrame with rows corresponding to the records and the
    columns corresponding to the record attributes.

    Args:
        path: Source data file (e.g., 'lidar.feather', 'calibration.feather', etc.)
        columns: Tuple of columns to load for the given record. Defaults to None.

    Returns:
        (N,len(columns)) Apache Feather data represented as a `pandas` DataFrame.
    """
    with open(file_path, "rb") as file_handle:
        dataframe: pd.DataFrame = feather.read_feather(
            file_handle, columns=columns, memory_map=True
        )
    return dataframe


def get_intrinsics(intrinsics: pd.DataFrame, sensor_name: str) -> NDArrayF32:
    """Get camera intrinsics with given sensor name."""
    intrinsics = intrinsics[intrinsics["sensor_name"] == sensor_name]

    K = np.eye(3, dtype=np.float32)
    K[0, 0] = intrinsics["fx_px"].item()
    K[1, 1] = intrinsics["fy_px"].item()
    K[0, 2] = intrinsics["cx_px"].item()
    K[1, 2] = intrinsics["cy_px"].item()

    return K


def get_pose(ego_pose_data: pd.DataFrame) -> NDArrayF32:
    """Get pose."""
    pose = np.eye(4).astype(np.float32)

    rot_matrix = quaternion_to_matrix(
        torch.from_numpy(
            ego_pose_data[["qw", "qx", "qy", "qz"]]
            .to_numpy()
            .astype(np.float32)
        )
    ).numpy()[0]

    translation = torch.from_numpy(
        ego_pose_data[["tx_m", "ty_m", "tz_m"]]
        .to_numpy()
        .astype(np.float32)[0]
        .reshape(3, 1)
    )

    pose[:3, :3] = rot_matrix
    pose[:3, 3:] = translation

    return pose


def find_closest_target_fpath(
    target_timestamps: list[int],
    src_timestamp_ns: int,
) -> tuple[int | None, int | None]:
    """Find the file path to the target sensor from a source sensor.

    Args:
        split: Dataset split.
        log_id: Vehicle log uuid.
        src_sensor_name: Name of the source sensor.
        src_timestamp_ns: Nanosecond timestamp of the source sensor (vehicle time).
        target_sensor_name: Name of the target sensor.

    Returns:
        The target sensor file path if it exists, otherwise None.

    Raises:
        RuntimeError: if the synchronization database (sync_records) has not been created.
    """
    index = None
    target_timestamp = None

    # Grab the synchronization record.
    for i, t in enumerate(target_timestamps):
        if t == src_timestamp_ns:
            target_timestamp = t
            index = i
            break
        elif t > src_timestamp_ns:
            if (
                t - src_timestamp_ns
                < target_timestamps[i - 1] - src_timestamp_ns
            ):
                target_timestamp = t
                index = i
            else:
                target_timestamp = target_timestamps[i - 1]
                index = i - 1
            break

    return target_timestamp, index


def sync_sensor_data(
    split: str,
    sample_rate: int,
    log_list: list[str],
    data_root: str = "data/av2",
    target_data_root: str = "data/argoverse",
    camera: str = "ring_front_center",
):
    """Syncronize Argoverse sensor data."""
    data_dir = os.path.join(data_root, "sensor")

    lidar_to_cam = {}
    for log in tqdm(log_list):
        lidar_to_cam[log] = {}

        log_dir = os.path.join(data_dir, split, log)

        cam_dir = os.path.join(log_dir, "sensors", "cameras", camera)

        cam_filenames = sorted(
            [f for f in os.listdir(cam_dir) if f.endswith(".jpg")]
        )

        cam_timestamps = [
            int(filename.split(".")[0]) for filename in cam_filenames
        ]

        lidar_dir = os.path.join(log_dir, "sensors", "lidar")

        frame_id = 0
        for filename in sorted(os.listdir(lidar_dir)):
            if not filename.endswith(".feather"):
                continue

            lidar_timestamp_ns = int(filename.split(".")[0])

            cam_timestamp_ns, camera_idx = find_closest_target_fpath(
                cam_timestamps, lidar_timestamp_ns
            )

            cam_filename = cam_filenames[camera_idx]

            if cam_timestamp_ns is None:
                print(f"Cannot find the closest timestamp for {filename}")
                continue

            # Camera Image
            if frame_id % sample_rate == 0:
                target_dir = os.path.join(
                    target_data_root, split, log, "images"
                )

                os.makedirs(target_dir, exist_ok=True)

                target_image_path = os.path.join(target_dir, cam_filename)

                if not os.path.exists(target_image_path):
                    shutil.copy(
                        os.path.join(cam_dir, cam_filename),
                        target_image_path,
                    )

            lidar_to_cam[log][lidar_timestamp_ns] = cam_timestamp_ns
            frame_id += 1

    return lidar_to_cam


def convert_sensor_data(
    split: str,
    sample_rate: int,
    log_list: list[str],
    lidar_to_cam: dict[str, dict[int, int]],
    target_data_root: str = "data/argoverse",
    camera: str = "ring_front_center",
):
    """Convert Argoverse sensor dataset for Vis4D."""
    data_dir = os.path.join("data", "av2", "sensor")

    coco_annotations = {
        "info": {
            "name": "Argoverse 2",
            "url": "https://www.argoverse.org/av2.html",
        },
    }

    categories = []
    for cat_name in class_names:
        cat_name = cat_name.lower()
        cat_name = cat_name.replace("_", " ")

        class_id = av2_class_map[cat_name]

        categories.append({"id": class_id, "name": cat_name})

    coco_annotations["categories"] = categories

    images = []
    image_id = 0
    coco_anns = []
    coco_ann_id = 0

    for i, log in enumerate(tqdm(log_list)):
        log_dir = os.path.join(data_dir, split, log)

        annotations = read_feather(
            os.path.join(log_dir, "annotations.feather")
        )

        # Ego to City per timestamp
        city_SE3_ego = read_feather(
            os.path.join(log_dir, "city_SE3_egovehicle.feather")
        )

        # Sensor to Ego per sensor
        ego_SE3_sensor = read_feather(
            os.path.join(
                log_dir, "calibration", "egovehicle_SE3_sensor.feather"
            )
        )

        cam_to_ego_data = ego_SE3_sensor[
            ego_SE3_sensor["sensor_name"] == camera
        ]

        cam_to_ego_cam = torch.from_numpy(get_pose(cam_to_ego_data))

        ego_cam_to_cam = inverse_rigid_transform(cam_to_ego_cam)

        intrinsics_dataframe = read_feather(
            os.path.join(log_dir, "calibration", "intrinsics.feather")
        )

        height = intrinsics_dataframe[
            intrinsics_dataframe["sensor_name"] == camera
        ]["height_px"].item()

        width = intrinsics_dataframe[
            intrinsics_dataframe["sensor_name"] == camera
        ]["width_px"].item()

        intrinsics = torch.from_numpy(
            get_intrinsics(intrinsics_dataframe, camera)
        )

        for frame_id, (lidar_timestamp_ns, cam_timestamp_ns) in enumerate(
            lidar_to_cam[log].items()
        ):
            if frame_id % sample_rate != 0:
                continue

            cam_filename = f"{cam_timestamp_ns}.jpg"

            anns = annotations[
                annotations["timestamp_ns"] == lidar_timestamp_ns
            ]

            # Ego LiDAR to City
            ego_pose_data = city_SE3_ego[
                city_SE3_ego["timestamp_ns"] == lidar_timestamp_ns
            ]

            ego_lidar_to_city = torch.from_numpy(get_pose(ego_pose_data))

            # Ego camera to City
            ego_pose_data = city_SE3_ego[
                city_SE3_ego["timestamp_ns"] == cam_timestamp_ns
            ]

            ego_cam_to_city = torch.from_numpy(get_pose(ego_pose_data))

            city_to_ego_cam = inverse_rigid_transform(ego_cam_to_city)

            ego_lidar_to_cam = (
                ego_cam_to_cam @ city_to_ego_cam @ ego_lidar_to_city
            )

            # Camera Image
            target_dir = os.path.join(target_data_root, split, log, "images")

            target_image_path = os.path.join(target_dir, cam_filename)

            image_info = {
                "width": width,
                "height": height,
                "file_path": target_image_path,
                "K": intrinsics.numpy().tolist(),
                "src_90_rotate": 0,
                "src_flagged": False,
                "incomplete": False,
                "id": image_id,
            }

            images.append(image_info)

            # Camera Annotations
            anns_tmp = []
            for ann in anns.iterrows():
                coco_ann = {}
                category_name = ann[1]["category"]

                category_name = category_name.lower()
                category_name = category_name.replace("_", " ")

                if not category_name in av2_class_map:
                    continue

                category_id = av2_class_map[category_name]

                # track_uuid = ann[1]["track_uuid"]
                num_interior_pts = ann[1]["num_interior_pts"]

                translation = [
                    ann[1]["tx_m"],
                    ann[1]["ty_m"],
                    ann[1]["tz_m"],
                ]
                wlh = [
                    ann[1]["width_m"],
                    ann[1]["length_m"],
                    ann[1]["height_m"],
                ]
                quat = [
                    ann[1]["qw"],
                    ann[1]["qx"],
                    ann[1]["qy"],
                    ann[1]["qz"],
                ]

                box3d = torch.tensor(
                    [*translation, *wlh, *quat], dtype=torch.float32
                )

                box3d_cam = transform_boxes3d(
                    box3d[None],
                    ego_lidar_to_cam,
                    source_axis_mode=AxisMode.ROS,
                    target_axis_mode=AxisMode.OPENCV,
                )

                corners_cam = boxes3d_to_corners(box3d_cam, AxisMode.OPENCV)

                if not boxes3d_in_image(
                    corners_cam, intrinsics, (height, width)
                ).item():
                    continue

                corners_cam_list = corners_cam[0].numpy().tolist()

                coco_ann["behind_camera"] = False
                coco_ann["depth_error"] = -1

                corner_coords = project_points(corners_cam, intrinsics)[
                    0
                ].numpy()

                x1 = min(corner_coords[:, 0])
                y1 = min(corner_coords[:, 1])
                x2 = max(corner_coords[:, 0])
                y2 = max(corner_coords[:, 1])

                proj_area = (x2 - x1) * (y2 - y1)

                min_x = max(0, x1)
                min_y = max(0, y1)
                max_x = min(width - 1, x2)
                max_y = min(height - 1, y2)

                area = (max_x - min_x) * (max_y - min_y)

                coco_ann["bbox2D_proj"] = [
                    float(min_x),
                    float(min_y),
                    float(max_x),
                    float(max_y),
                ]

                # Truncation
                coco_ann["truncation"] = 1.0 - area / proj_area

                coco_ann["segmentation_pts"] = -1
                coco_ann["lidar_pts"] = num_interior_pts

                coco_ann["valid3D"] = True

                coco_ann["category_name"] = category_name
                coco_ann["category_id"] = category_id

                coco_ann["id"] = coco_ann_id
                coco_ann_id += 1

                coco_ann["image_id"] = image_id

                box3d_cam_list = box3d_cam[0].numpy().tolist()

                coco_ann["center_cam"] = [
                    box3d_cam_list[0],
                    box3d_cam_list[1],
                    box3d_cam_list[2],
                ]

                # wlh to whl
                coco_ann["dimensions"] = [
                    box3d_cam_list[3],
                    box3d_cam_list[5],
                    box3d_cam_list[4],
                ]

                # Rotation Matrix
                coco_ann["R_cam"] = (
                    quaternion_to_matrix(box3d_cam[0, 6:]).numpy().tolist()
                )

                # Map to Omni3D corners
                coco_ann["bbox3D_cam"] = [
                    corners_cam_list[6],
                    corners_cam_list[4],
                    corners_cam_list[0],
                    corners_cam_list[2],
                    corners_cam_list[7],
                    corners_cam_list[5],
                    corners_cam_list[1],
                    corners_cam_list[3],
                ]

                anns_tmp.append(coco_ann)

            # Compute Visibility
            if len(anns_tmp) == 0:
                pass
            elif len(anns_tmp) == 1:
                anns_tmp[0]["visibility"] = 1.0
            else:
                boxes = torch.tensor([a["bbox2D_proj"] for a in anns_tmp])

                ious = bbox_intersection(boxes, boxes)

                for i, coco_ann in enumerate(anns_tmp):
                    iou = ious[i]

                    if iou.sum().item() == 1.0:
                        coco_ann["visibility"] = 1.0
                    else:
                        depth = anns_tmp[i]["center_cam"][2]
                        occ_boxes = []
                        for j, iou_val in enumerate(iou):
                            if j == i:
                                continue

                            if iou_val.item() != 0.0:
                                if anns_tmp[j]["center_cam"][2] < depth:
                                    occ_boxes.append(
                                        anns_tmp[j]["bbox2D_proj"]
                                    )

                        coco_ann["visibility"] = compute_visibility_mask(
                            coco_ann["bbox2D_proj"], occ_boxes
                        )

            coco_anns.extend(anns_tmp)

            image_id += 1

    coco_annotations["images"] = images
    coco_annotations["annotations"] = coco_anns

    ann_file = os.path.join(
        target_data_root, "annotations", f"Argoverse_{split}.json"
    )

    with open(ann_file, "w") as file:
        json.dump(coco_annotations, file)

    print(f"Argoverse V2 {split} has {image_id} {camera} images.")


def generate_av2_depth(
    split: str,
    sample_rate: int,
    log_list: list[str],
    lidar_to_cam: dict[str, dict[int, int]],
    target_data_root: str = "data/argoverse",
    camera: str = "ring_front_center",
    depth_scale: float = 256.0,
):
    """Generate AV2 depth maps."""
    data_dir = os.path.join("data", "av2", "sensor")

    for log in tqdm(log_list):
        log_dir = os.path.join(data_dir, split, log)

        # Ego to City per timestamp
        city_SE3_ego = read_feather(
            os.path.join(log_dir, "city_SE3_egovehicle.feather")
        )

        # Sensor to Ego per sensor
        ego_SE3_sensor = read_feather(
            os.path.join(
                log_dir, "calibration", "egovehicle_SE3_sensor.feather"
            )
        )

        cam_to_ego_data = ego_SE3_sensor[
            ego_SE3_sensor["sensor_name"] == camera
        ]

        cam_to_ego_cam = torch.from_numpy(get_pose(cam_to_ego_data))

        ego_cam_to_cam = inverse_rigid_transform(cam_to_ego_cam)

        intrinsics_dataframe = read_feather(
            os.path.join(log_dir, "calibration", "intrinsics.feather")
        )

        height = intrinsics_dataframe[
            intrinsics_dataframe["sensor_name"] == camera
        ]["height_px"].item()

        width = intrinsics_dataframe[
            intrinsics_dataframe["sensor_name"] == camera
        ]["width_px"].item()

        intrinsics = torch.from_numpy(
            get_intrinsics(intrinsics_dataframe, camera)
        )

        sweeps = []
        for frame_id, (lidar_timestamp_ns, cam_timestamp_ns) in enumerate(
            lidar_to_cam[log].items()
        ):
            filename = f"{lidar_timestamp_ns}.feather"

            cam_filename = f"{cam_timestamp_ns}.jpg"

            lidar_path = os.path.join(log_dir, "sensors", "lidar", filename)

            sweep = read_feather(
                lidar_path,
                columns=["x", "y", "z", "intensity", "laser_number"],
            )

            points = (
                sweep.loc[:, ["x", "y", "z"]].to_numpy().astype(np.float32)
            )

            # Ego LiDAR to City
            ego_pose_data = city_SE3_ego[
                city_SE3_ego["timestamp_ns"] == lidar_timestamp_ns
            ]

            ego_lidar_to_city = torch.from_numpy(get_pose(ego_pose_data))

            city_to_ego_lidar = inverse_rigid_transform(ego_lidar_to_city)

            # Ego camera to City
            ego_pose_data = city_SE3_ego[
                city_SE3_ego["timestamp_ns"] == cam_timestamp_ns
            ]

            ego_cam_to_city = torch.from_numpy(get_pose(ego_pose_data))

            city_to_ego_cam = inverse_rigid_transform(ego_cam_to_city)

            ego_lidar_to_cam = (
                ego_cam_to_cam @ city_to_ego_cam @ ego_lidar_to_city
            )

            # Generate Depth Map
            if frame_id % sample_rate == 0:
                points_sweeps = [torch.from_numpy(points)]
                for city_points in sweeps:
                    cur_lidar_points = transform_points(
                        city_points, city_to_ego_lidar
                    )

                    points_sweeps.append(cur_lidar_points)

                points_cam = transform_points(
                    torch.cat(points_sweeps), ego_lidar_to_cam
                )

                depth_map = generate_depth_map(
                    points_cam, intrinsics, (height, width)
                )
                depth = depth_map.numpy()

                numpy_image = (depth * depth_scale).astype(np.uint16)

                depth_dir = os.path.join(target_data_root, split, log, "depth")

                os.makedirs(depth_dir, exist_ok=True)

                depth_file_path = os.path.join(
                    depth_dir, cam_filename.replace(".jpg", "_depth.png")
                )

                Image.fromarray(numpy_image).save(depth_file_path)

            lidar_global = transform_points(
                torch.from_numpy(points), ego_lidar_to_city
            )

            sweeps.append(lidar_global)

            if len(sweeps) > ACC_FRAMES:
                sweeps.pop(0)


if __name__ == "__main__":
    """Convert Argoverse dataset."""
    data_root = "data/av2"
    target_data_root = "data/argoverse"
    split = "val"
    sample_rate = VAL_SAMPLE_RATE
    log_list = VAL

    # Syncronize sensor data
    print(f"Syncronizing {split} sensor data...")
    lidar_to_cam = sync_sensor_data(
        split,
        sample_rate,
        log_list,
        data_root=data_root,
        target_data_root=target_data_root,
        camera="ring_front_center",
    )

    # Generate annoations in COCO format
    print(f"Converting {split} sensor data...")
    convert_sensor_data(
        split,
        sample_rate,
        log_list,
        lidar_to_cam=lidar_to_cam,
        target_data_root=target_data_root,
        camera="ring_front_center",
    )

    # Generate depth maps
    print(f"Generating {split} depth maps...")
    generate_av2_depth(
        split,
        sample_rate,
        log_list,
        lidar_to_cam=lidar_to_cam,
        target_data_root=target_data_root,
        camera="ring_front_center",
    )
