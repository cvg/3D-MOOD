"""Generate depth for KITTI Object from Omni3D."""

import argparse
import os
import json

from PIL import Image
from tqdm import tqdm

from vis4d.data.io.file import FileBackend
from vis4d.data.datasets.util import im_decode


def get_kitti_mapping() -> tuple[list[str], list[str]]:
    """Get the KITTI object and raw data mapping."""
    with open(os.path.join("scripts", "kitti", "train_rand.txt"), "r") as f:
        kitti_train_rand = f.readlines()[0].split(",")

    with open(os.path.join("scripts", "kitti", "train_mapping.txt"), "r") as f:
        kitti_train_mapping = f.readlines()

    return kitti_train_rand, kitti_train_mapping


def parse_depth(
    images,
    kitti_depth_data_root: str = "data/kitti_depth",
    target_data_root: str = "data/KITTI_object_depth",
):
    """Parse sequence of images."""
    kitti_train_rand, kitti_train_mapping = get_kitti_mapping()

    depth_data_backend = FileBackend()

    num_valid_depth = 0
    for img in tqdm(images):
        _, split, image_id, img_filename = img["file_path"].split("/")

        img_name = img_filename.split(".")[0]

        raw = kitti_train_mapping[
            int(kitti_train_rand[int(img_name)]) - 1
        ].replace("\n", "")
        _, seq_name, raw_img_name = raw.split()

        depth_gt_file_path = os.path.join(
            kitti_depth_data_root,
            "gt_depth",
            seq_name,
            "proj_depth",
            "groundtruth",
            "image_02",
            f"{raw_img_name}.png",
        )

        if depth_data_backend.exists(depth_gt_file_path):
            if os.path.exists(depth_file_path):
                continue

            depth_bytes = depth_data_backend.get(depth_gt_file_path)
            depth_array = im_decode(depth_bytes)

            target_dir = os.path.join(target_data_root, split, image_id)
            os.makedirs(target_dir, exist_ok=True)

            depth_file_path = os.path.join(target_dir, f"{img_name}.png")

            Image.fromarray(depth_array).save(depth_file_path)

            num_valid_depth += 1

    print(f"Numbers of valid depth maps: {num_valid_depth}")


def convert_dataset(
    omni3d_data_root: str = "data/omni3d",
    kitti_depth_data_root: str = "data/kitti_depth",
    target_data_root: str = "data/KITTI_object_depth",
) -> None:
    """Convert KITTI depth."""
    for dataset in ["KITTI_train", "KITTI_val", "KITTI_test"]:
        annotation = os.path.join(
            omni3d_data_root, "annotations", f"{dataset}.json"
        )

        print(f"Generating depth for {dataset}...")

        with open(annotation, "r") as file:
            kitti_samples = json.load(file)

            parse_depth(
                kitti_samples["images"],
                kitti_depth_data_root,
                target_data_root,
            )

    print("Done.")


if __name__ == "__main__":
    """Generate KITTI depth."""
    parser = argparse.ArgumentParser(description="Generate KITTI depth.")
    parser.add_argument(
        "--omni3d_data_root",
        default="data/omni3d",
        help="Path to the Omni3D dataset",
    )
    parser.add_argument(
        "--kitti_depth_data_root",
        default="data/kitti_depth",
        help="Path to the KITTI depth dataset",
    )
    parser.add_argument(
        "--target_data_root",
        default="data/KITTI_object_depth",
        help="Path to the target dataset",
    )
    args = parser.parse_args()

    convert_dataset(
        omni3d_data_root=args.omni3d_data_root,
        kitti_depth_data_root=args.kitti_depth_data_root,
        target_data_root=args.target_data_root,
    )
