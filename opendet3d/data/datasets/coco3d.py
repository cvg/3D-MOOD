"""COCO 3D API."""

from __future__ import annotations

import contextlib
import io
import json
import os
import time
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from pycocotools.coco import COCO
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from vis4d.common.logging import rank_zero_info, rank_zero_warn
from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import AxisMode
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.base import Dataset
from vis4d.data.datasets.util import (
    CacheMappingMixin,
    im_decode,
    print_class_histogram,
)
from vis4d.data.typing import DictData


class COCO3DDataset(CacheMappingMixin, Dataset):
    """3D Object Detection Dataset using coco annotation files."""

    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        class_map: dict[str, int],
        det_map: dict[str, int],
        keys_to_load: Sequence[str] = (K.images, K.boxes2d, K.boxes3d),
        with_depth: bool = False,
        max_depth: float = 80.0,
        depth_scale: float = 256.0,
        remove_empty: bool = False,
        data_prefix: str | None = None,
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__(**kwargs)
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.annotation_file = f"{dataset_name}.json"

        self.keys_to_load = list(keys_to_load)
        self.remove_empty = remove_empty

        self.class_map = class_map  # Class mapping in the annotation file
        self.det_map = det_map  # Class mapping for detection
        self.categories = sorted(self.det_map, key=self.det_map.get)

        self.data_prefix = data_prefix
        self.text_prompt_mapping = text_prompt_mapping

        # Metric Depth
        if with_depth and not K.depth_maps in keys_to_load:
            self.keys_to_load.append(K.depth_maps)

        self.max_depth = max_depth
        self.depth_scale = depth_scale

        # Load annotations
        self.samples, _ = self._load_mapping(
            self._generate_data_mapping,
            self._filter_data,
            cache_as_binary=cache_as_binary,
            cached_file_path=cached_file_path,
        )

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return self.dataset_name

    def _filter_data(self, data: list[DictStrAny]) -> list[DictStrAny]:
        """Remove empty samples."""
        samples = []

        frequencies = {cat: 0 for cat in sorted(self.det_map)}

        empty_samples = 0
        no_depth_samples = 0
        for sample in data:
            if self.remove_empty and len(sample["anns"]) == 0:
                empty_samples += 1
                continue

            if (
                K.depth_maps in self.keys_to_load
                and "depth_filename" not in sample
            ):
                empty_samples += 1
                no_depth_samples += 1
                continue

            for ann in sample["anns"]:
                frequencies[ann["category_name"]] += 1

            samples.append(sample)

        rank_zero_info(
            f"Propocessing {self.dataset_name} with {len(samples)} samples."
        )
        rank_zero_info(f"No depth samples: {no_depth_samples}")
        rank_zero_info(f"Filtered {empty_samples} empty samples")
        print_class_histogram(frequencies)

        return samples

    def _get_cat_id(
        self, img: DictStrAny, ann: DictStrAny, cat_name: str
    ) -> None:
        """Get the category id from the category name."""
        ann["category_id"] = self.det_map[cat_name]

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generates the data mapping."""
        # Load annotations
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO3D(
                os.path.join(
                    self.data_root, "annotations", self.annotation_file
                ),
                self.categories,
            )

        cats_map = {v: k for k, v in self.class_map.items()}

        img_ids = sorted(coco_api.getImgIds())
        imgs = coco_api.loadImgs(img_ids)

        samples = []
        for img_id, img in zip(img_ids, imgs):
            # Fix file path for Omni3D
            if self.data_prefix is not None:
                img["file_path"] = os.path.join(
                    self.data_prefix, img["file_path"]
                )

            valid_anns = []
            anns = coco_api.imgToAnns[img_id]

            boxes = []
            boxes3d = np.empty((0, 10), dtype=np.float32)[1:]
            class_ids = np.empty((0,), dtype=np.int64)[1:]
            for ann in anns:
                cat_name = cats_map[ann["category_id"]]
                assert cat_name == ann["category_name"]

                if cat_name in {"dontcare", "ignore", "void"}:
                    continue

                if ann["ignore"]:
                    continue

                self._get_cat_id(img, ann, cat_name)

                # Box 2D
                x1, y1, width, height = ann["bbox"]
                x2, y2 = x1 + width, y1 + height
                boxes.append((x1, y1, x2, y2))

                # Class
                class_ids = np.concatenate(
                    [
                        class_ids,
                        np.array([ann["category_id"]], dtype=np.int64),
                    ]
                )

                # Box 3D
                center = ann["center_cam"]
                width, height, length = ann["dimensions"]

                # Check if the rotation matrix is valid
                try:
                    x, y, z, w = R.from_matrix(
                        np.array(ann["R_cam"])
                    ).as_quat()
                except Exception as e:
                    rank_zero_warn(
                        f"Error processing rotation matrix for annotation {ann['id']}: {e}"
                    )
                    continue

                orientation = Quaternion([w, x, y, z])

                boxes3d = np.concatenate(
                    [
                        boxes3d,
                        np.array(
                            [
                                [
                                    *center,
                                    width,
                                    length,
                                    height,
                                    *orientation.elements,
                                ]
                            ],
                            dtype=np.float32,
                        ),
                    ]
                )

                valid_anns.append(ann)

            boxes2d = (
                np.empty((0, 4), dtype=np.float32)
                if not boxes
                else np.array(boxes, dtype=np.float32)
            )

            depth_filename = self.get_depth_filenames(img)

            sample = {
                "img_id": img_id,
                "img": img,
                "anns": valid_anns,
                "boxes2d": boxes2d,
                "boxes3d": boxes3d,
                "class_ids": class_ids,
            }

            if depth_filename is not None and self.data_backend.exists(
                depth_filename
            ):
                sample["depth_filename"] = depth_filename

            samples.append(sample)

        return samples

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Get the depth filenames.

        Since not every data has depth.
        """
        return None

    def get_cat_ids(self, idx: int) -> list[int]:
        """Return the samples."""
        return self.samples[idx]["class_ids"].tolist()

    def __len__(self) -> int:
        """Total number of samples of data."""
        return len(self.samples)

    def get_depth_map(self, sample: DictStrAny) -> np.ndarray:
        """Get the depth map."""
        depth_bytes = self.data_backend.get(sample["depth_filename"])
        depth_array = im_decode(depth_bytes)

        depth = np.ascontiguousarray(depth_array, dtype=np.float32)

        depth = depth / self.depth_scale

        return depth

    def __getitem__(self, idx: int) -> DictData:
        """Get single sample.

        Args:
            idx (int): Index of sample.

        Returns:
            DictData: sample at index in Vis4D input format.
        """
        sample = self.samples[idx]
        data_dict: DictData = {}

        # Get image info
        data_dict[K.sample_names] = sample["img_id"]

        data_dict["dataset_name"] = self.dataset_name
        data_dict[K.boxes2d_names] = self.categories
        data_dict["text_prompt_mapping"] = self.text_prompt_mapping

        if K.images in self.keys_to_load:
            im_bytes = self.data_backend.get(sample["img"]["file_path"])
            image = np.ascontiguousarray(
                im_decode(im_bytes, mode=self.image_channel_mode),
                dtype=np.float32,
            )[None]

            data_dict[K.images] = image
            data_dict[K.input_hw] = (image.shape[1], image.shape[2])

            data_dict[K.original_images] = image
            data_dict[K.original_hw] = (image.shape[1], image.shape[2])

            # Get camera info
            intrinsics = np.array(sample["img"]["K"], dtype=np.float32)
            data_dict[K.intrinsics] = intrinsics
            data_dict["original_intrinsics"] = intrinsics

        data_dict[K.boxes2d] = sample["boxes2d"]
        data_dict[K.boxes2d_classes] = sample["class_ids"]
        data_dict[K.boxes3d] = sample["boxes3d"]
        data_dict[K.boxes3d_classes] = sample["class_ids"]
        data_dict[K.axis_mode] = AxisMode.OPENCV

        if K.depth_maps in self.keys_to_load:
            depth = self.get_depth_map(sample)

            depth[depth > self.max_depth] = 0

            data_dict[K.depth_maps] = depth

        data_dict["tokens_positive"] = None

        self.data_backend.close()

        return data_dict


class COCO3D(COCO):
    """COCO API with 3D annotations."""

    def __init__(
        self,
        annotation_files: Sequence[str] | str,
        category_names: Sequence[str] | None = None,
        ignore_names: Sequence[str] = ("dontcare", "ignore", "void"),
        truncation_thres: float = 0.33333333,
        visibility_thres: float = 0.33333333,
        min_height_thres: float = 0.0625,
        max_height_thres: float = 1.50,
        modal_2D_boxes: bool = False,
        trunc_2D_boxes: bool = True,
        max_depth: int = 1e8,
    ) -> None:
        """Creates an instance of the class."""
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        self.truncation_thres = truncation_thres
        self.visibility_thres = visibility_thres
        self.min_height_thres = min_height_thres
        self.max_height_thres = max_height_thres
        self.max_depth = max_depth

        if isinstance(annotation_files, str):
            annotation_files = [annotation_files]

        cats_ids_master = []
        cats_master = []

        for annotation_file in annotation_files:
            _, tail = os.path.split(annotation_file)
            name, _ = os.path.splitext(tail)

            print(f"loading {name} annotations into memory...")
            tic = time.time()

            with open(annotation_file, "r") as f:
                dataset = json.load(f)

            assert (
                type(dataset) == dict
            ), f"annotation file format {type(dataset)} not supported"
            print(f"Done (t={time.time() - tic:.2f}s)")

            if type(dataset["info"]) == list:
                dataset["info"] = dataset["info"][0]

            dataset["info"]["known_category_ids"] = [
                cat["id"] for cat in dataset["categories"]
            ]

            # first dataset
            if len(self.dataset) == 0:
                self.dataset = dataset
            # concatenate datasets
            else:
                if type(self.dataset["info"]) == dict:
                    self.dataset["info"] = [self.dataset["info"]]

                self.dataset["info"] += [dataset["info"]]
                self.dataset["annotations"] += dataset["annotations"]
                self.dataset["images"] += dataset["images"]

            # sort through categories
            for cat in dataset["categories"]:
                if not cat["id"] in cats_ids_master:
                    cats_ids_master.append(cat["id"])
                    cats_master.append(cat)

        # category names are provided to us
        if category_names is not None:
            self.dataset["categories"] = [
                cats_master[i]
                for i in np.argsort(cats_ids_master)
                if cats_master[i]["name"] in category_names
            ]
        # no categories are provided, so assume use ALL available.
        else:
            self.dataset["categories"] = [
                cats_master[i] for i in np.argsort(cats_ids_master)
            ]

            category_names = [
                cat["name"] for cat in self.dataset["categories"]
            ]

        # determine which categories we may actually use for filtering.
        trainable_cats = set(ignore_names) | set(category_names)

        valid_anns = []
        im_height_map = {}

        for im_obj in self.dataset["images"]:
            im_height_map[im_obj["id"]] = im_obj["height"]

        # Filter out annotations
        for anno_idx, anno in enumerate(self.dataset["annotations"]):

            im_height = im_height_map[anno["image_id"]]

            # tightly annotated 2D boxes are not always available.
            if (
                modal_2D_boxes
                and "bbox2D_tight" in anno
                and anno["bbox2D_tight"][0] != -1
            ):
                bbox2D = anno["bbox2D_tight"]
            elif (
                trunc_2D_boxes
                and "bbox2D_trunc" in anno
                and not np.all([val == -1 for val in anno["bbox2D_trunc"]])
            ):
                bbox2D = anno["bbox2D_trunc"]
            elif anno["bbox2D_proj"][0] != -1:
                bbox2D = anno["bbox2D_proj"]
            elif anno["bbox2D_tight"][0] != -1:
                bbox2D = anno["bbox2D_tight"]
            else:
                continue

            # convert to xywh
            bbox2D[2] = bbox2D[2] - bbox2D[0]
            bbox2D[3] = bbox2D[3] - bbox2D[1]

            ignore = self.is_ignore(anno, bbox2D, ignore_names, im_height)

            width = bbox2D[2]
            height = bbox2D[3]

            self.dataset["annotations"][anno_idx]["area"] = width * height
            self.dataset["annotations"][anno_idx]["iscrowd"] = False
            self.dataset["annotations"][anno_idx]["ignore"] = ignore
            self.dataset["annotations"][anno_idx]["ignore2D"] = ignore
            self.dataset["annotations"][anno_idx]["ignore3D"] = ignore

            self.dataset["annotations"][anno_idx]["bbox"] = bbox2D
            self.dataset["annotations"][anno_idx]["bbox3D"] = anno[
                "bbox3D_cam"
            ]
            self.dataset["annotations"][anno_idx]["depth"] = anno[
                "center_cam"
            ][2]

            category_name = anno["category_name"]

            if category_name in trainable_cats:
                valid_anns.append(self.dataset["annotations"][anno_idx])

        self.dataset["annotations"] = valid_anns

        self.createIndex()

    def is_ignore(
        self,
        anno,
        bbox2D: list[float, float, float, float],
        ignore_names: Sequence[str] | None,
        image_height: int,
    ) -> bool:
        ignore = anno["behind_camera"]
        ignore |= not bool(anno["valid3D"])

        if ignore:
            return ignore

        ignore |= anno["dimensions"][0] <= 0
        ignore |= anno["dimensions"][1] <= 0
        ignore |= anno["dimensions"][2] <= 0
        ignore |= anno["center_cam"][2] > self.max_depth
        ignore |= anno["lidar_pts"] == 0
        ignore |= anno["segmentation_pts"] == 0
        ignore |= anno["depth_error"] > 0.5

        ignore |= bbox2D[3] <= self.min_height_thres * image_height
        ignore |= bbox2D[3] >= self.max_height_thres * image_height

        ignore |= (
            anno["truncation"] >= 0
            and anno["truncation"] >= self.truncation_thres
        )
        ignore |= (
            anno["visibility"] >= 0
            and anno["visibility"] <= self.visibility_thres
        )

        if ignore_names is not None:
            ignore |= anno["category_name"] in ignore_names

        return ignore
