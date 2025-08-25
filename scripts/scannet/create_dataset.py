"""ScanNet dataset creation script."""

import argparse

import os
import json
import shutil

from tqdm import tqdm
from PIL import Image, ImageOps
from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion

import numpy as np
import torch

from vis4d.data.const import AxisMode
from vis4d.op.box.box2d import bbox_intersection
from vis4d.op.box.box3d import (
    boxes3d_to_corners,
    boxes3d_in_image,
    transform_boxes3d,
)
from vis4d.op.geometry.rotation import (
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from vis4d.op.geometry.transform import (
    transform_points,
    inverse_rigid_transform,
)
from vis4d.op.geometry.projection import generate_depth_map, project_points

from opendet3d.data.datasets.scannet import (
    scannet_class_map,
    scannet200_class_map,
)
from opendet3d.op.box.box2d import compute_visibility_mask


def convert_scannet(
    data_root: str,
    split: str,
    scannet200: bool = False,
    max_images_per_sequence: int = 20,
    camera_near_clip: float = 0.1,
    depth_scale: float = 1000.0,
) -> None:
    """Get information of the raw data and save the coco format annotations."""
    pose_images_dir = os.path.join(data_root, "posed_images")

    coco_annotations = {
        "info": {
            "name": "ScanNet V2",
            "url": "https://github.com/ScanNet/ScanNet/tree/master",
        },
    }

    if scannet200:
        instance_data_dir = os.path.join(data_root, "scannet200_instance_data")
        class_map = scannet200_class_map
    else:
        instance_data_dir = os.path.join(data_root, "scannet_instance_data")
        class_map = scannet_class_map

    cat_mapping = {v: k for k, v in class_map.items()}

    categories = []
    for cat_name, class_id in class_map.items():
        categories.append({"id": class_id, "name": cat_name})

    coco_annotations["categories"] = categories

    split_file = os.path.join(data_root, "meta_data", f"scannetv2_{split}.txt")

    with open(split_file, "r", encoding="utf-8") as f:
        sample_id_list = f.read().splitlines()

    images = []
    image_id = 0
    coco_anns = []
    coco_ann_id = 0
    for sample_idx in tqdm(sample_id_list):
        pose_image_dir = os.path.join(pose_images_dir, sample_idx)

        # RGB Images & Cameara Poses
        all_img_paths = []
        all_extrinsics = []
        for file in sorted(os.listdir(pose_image_dir)):
            if file.endswith(".jpg"):
                all_img_paths.append(os.path.join(pose_image_dir, file))

            if file.endswith(".txt") and not file == "intrinsic.txt":
                all_extrinsics.append(
                    np.loadtxt(os.path.join(pose_image_dir, file))
                )

        # Remove invalid poses
        extrinsics, img_paths = [], []
        for extrinsic, img_path in zip(all_extrinsics, all_img_paths):
            if np.all(np.isfinite(extrinsic)):
                img_paths.append(img_path)
                extrinsics.append(extrinsic)

        # Camera intrinsics
        intrinsics_np = np.loadtxt(
            os.path.join(pose_image_dir, "intrinsic.txt")
        ).astype(np.float32)
        intrinsics = torch.from_numpy(intrinsics_np[:3, :3])

        # NOTE: 3D Box is (x, y, z, delta_x, delta_y, delta_z, class)
        # Axis-aligned 3D bounding box
        aligned_box_label = np.load(
            os.path.join(instance_data_dir, f"{sample_idx}_aligned_bbox.npy")
        )

        # Unaligned 3D bounding box
        # unaligned_box_label = np.load(
        #     os.path.join(
        #         instance_data_dir,
        #         f"{sample_idx}_unaligned_bbox.npy",
        #     )
        # )

        # Axis align matrix
        axis_align_matrix = torch.from_numpy(
            np.load(
                os.path.join(
                    instance_data_dir, f"{sample_idx}_axis_align_matrix.npy"
                )
            ).astype(np.float32)
        )

        inv_align_matrix = inverse_rigid_transform(axis_align_matrix)

        if aligned_box_label.shape[0] != 0:
            aligned_box = aligned_box_label[:, :-1]  # k, 6

            classes = aligned_box_label[:, -1]  # k
            classes = torch.from_numpy(classes.astype(np.int64))

            x, y, z, w = R.from_euler("XYZ", [0, 0, 0]).as_quat()

            orientation = Quaternion([w, x, y, z])

            boxes3d = np.zeros((aligned_box.shape[0], 10)).astype(np.float32)

            boxes3d[:, :3] = aligned_box[:, :3]

            boxes3d[:, 3] = aligned_box[:, 4]  # w
            boxes3d[:, 4] = aligned_box[:, 3]  # l
            boxes3d[:, 5] = aligned_box[:, 5]  # h

            boxes3d[:, 6:10] = orientation.elements

            boxes3d = transform_boxes3d(
                torch.from_numpy(boxes3d),
                inv_align_matrix,
                AxisMode.LIDAR,
                AxisMode.LIDAR,
                only_yaw=False,
            )

        pts_filename = os.path.join(
            instance_data_dir, f"{sample_idx}_vert.npy"
        )
        points = np.load(pts_filename)

        # x, y, z
        points = points[:, :3]

        current_points = torch.from_numpy(points.astype(np.float32))

        # NOTE: Transform points to the axis-aligned frame
        # points_aligned = transform_points(
        #     current_points, axis_align_matrix
        # )

        num_images = 0

        sample_freq = len(img_paths) // max_images_per_sequence

        for image_i, image_path in enumerate(img_paths):
            if image_i % sample_freq != 0:
                continue

            if num_images >= max_images_per_sequence:
                break

            scene_name, image_name = image_path.split("/")[-2:]

            # Camera Image
            scene_dir = os.path.join(data_root, split, scene_name)
            os.makedirs(scene_dir, exist_ok=True)

            image_dir = os.path.join(scene_dir, "image")
            os.makedirs(image_dir, exist_ok=True)

            target_image_path = os.path.join(image_dir, image_name)

            if not os.path.exists(target_image_path):
                shutil.copy(image_path, target_image_path)

            pil_img_file = Image.open(target_image_path)
            pil_img = ImageOps.exif_transpose(pil_img_file)
            image = np.array(pil_img)

            num_images += 1

            img_height, img_width = image.shape[:2]

            image_info = {
                "width": img_width,
                "height": img_height,
                "file_path": target_image_path,
                "K": intrinsics.numpy().tolist(),
                "src_90_rotate": 0,
                "src_flagged": False,
                "incomplete": False,
                "id": image_id,
            }

            images.append(image_info)

            cam_pose = extrinsics[image_i]

            world_to_cam = inverse_rigid_transform(
                torch.from_numpy(cam_pose.astype(np.float32))
            )

            # Generate Depth Map
            points_cam = transform_points(current_points, world_to_cam)
            points_cam = points_cam[points_cam[:, 2] > 0]

            depth = generate_depth_map(
                points_cam, intrinsics, image.shape[:2]
            ).numpy()

            depth_dir = os.path.join(scene_dir, "depth")
            os.makedirs(depth_dir, exist_ok=True)

            numpy_image = (depth * depth_scale).astype(np.uint16)

            depth_file_path = os.path.join(
                depth_dir, image_name.replace(".jpg", ".png")
            )

            Image.fromarray(numpy_image).save(depth_file_path)

            # Camera Annotations
            boxes3d_cam = transform_boxes3d(
                boxes3d,
                world_to_cam,
                AxisMode.LIDAR,
                AxisMode.OPENCV,
                only_yaw=False,
            )

            # Depth to Camera
            rot = quaternion_to_matrix(boxes3d_cam[:, 6:])

            rt_mat = rot.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

            rot = rot @ rt_mat

            boxes3d_cam[:, 6:] = matrix_to_quaternion(rot)

            corners = boxes3d_to_corners(boxes3d_cam, AxisMode.OPENCV)

            mask = boxes3d_in_image(
                corners, intrinsics.numpy(), image.shape[:2]
            )

            boxes3d_cam = boxes3d_cam[mask]

            classes_cam = classes[mask]
            corners = corners[mask]

            anns_tmp = []
            for i, corner in enumerate(corners):
                coco_ann = {}

                category_id = int(classes_cam[i].item())

                if category_id not in cat_mapping:
                    continue

                category_name = cat_mapping[category_id]

                coco_ann["behind_camera"] = False

                corner_valid = corner[corner[:, 2] > camera_near_clip]

                corner_coords = project_points(
                    corner_valid, intrinsics
                ).numpy()

                x1 = min(corner_coords[:, 0])
                y1 = min(corner_coords[:, 1])
                x2 = max(corner_coords[:, 0])
                y2 = max(corner_coords[:, 1])

                proj_area = (x2 - x1) * (y2 - y1)

                min_x = max(0, x1)
                min_y = max(0, y1)
                max_x = min(img_width - 1, x2)
                max_y = min(img_height - 1, y2)

                area = (max_x - min_x) * (max_y - min_y)

                # Truncation
                coco_ann["truncation"] = 1.0 - area / proj_area

                coco_ann["bbox2D_proj"] = [
                    float(min_x),
                    float(min_y),
                    float(max_x),
                    float(max_y),
                ]

                coco_ann["segmentation_pts"] = -1
                coco_ann["lidar_pts"] = -1

                coco_ann["valid3D"] = True

                coco_ann["category_name"] = category_name
                coco_ann["category_id"] = category_id

                coco_ann["id"] = coco_ann_id
                coco_ann_id += 1
                coco_ann["image_id"] = image_id

                coco_ann["depth_error"] = -1

                box3d_cam_list = boxes3d_cam[i].numpy().tolist()

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
                    quaternion_to_matrix(
                        torch.tensor(box3d_cam_list[6:], dtype=torch.float32)
                    )
                    .numpy()
                    .tolist()
                )

                corners_cam_list = corner.numpy().tolist()

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

    print(f"Total images: {image_id}")

    coco_annotations["images"] = images
    coco_annotations["annotations"] = coco_anns

    if scannet200:
        ann_file = os.path.join(
            data_root, "annotations", f"ScanNet200_{split}.json"
        )
    else:
        ann_file = os.path.join(
            data_root, "annotations", f"ScanNet_{split}.json"
        )

    with open(ann_file, "w") as file:
        json.dump(coco_annotations, file)


if __name__ == "__main__":
    """Create ScanNet dataset."""
    parser = argparse.ArgumentParser(description="Create ScanNet dataset")
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/scannet",
        help="specify the root path of dataset",
    )
    parser.add_argument("--scannet200", default=False, action="store_true")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="specify the root path of dataset",
    )
    args = parser.parse_args()

    convert_scannet(args.data_root, args.split, scannet200=args.scannet200)
