"""Object detection and visual grounding dataset."""

from __future__ import annotations

import json
import os.path as osp

import numpy as np
from tqdm import tqdm
from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import ArgsType, DictStrAny
from vis4d.data.const import CommonKeys as K
from vis4d.data.datasets.base import Dataset
from vis4d.data.datasets.util import (
    CacheMappingMixin,
    im_decode,
    print_class_histogram,
)
from vis4d.data.typing import DictData


class ODVGDataset(CacheMappingMixin, Dataset):
    """Object detection and visual grounding dataset."""

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        label_map_file: str | None = None,
        dataset_type: str = "VG",
        dataset_prefix: str | None = None,
        remove_empty: bool = False,
        cache_as_binary: bool = False,
        cached_file_path: str | None = None,
        **kwargs: ArgsType,
    ) -> None:
        """Create an object detection and visual grounding dataset."""
        super().__init__(**kwargs)

        self.data_root = data_root
        self.ann_file = ann_file
        self.dataset_type = dataset_type
        self.dataset_prefix = dataset_prefix
        self.remove_empty = remove_empty

        if label_map_file is not None:
            label_map_file = osp.join(self.data_root, label_map_file)

            with open(label_map_file, "r") as file:
                # dict[class_id (str): class_name (str)]
                self.label_map = json.load(file)

            self.dataset_type = "OD"

            self.det_map = {v: int(k) for k, v in self.label_map.items()}
            self.categories = sorted(self.det_map, key=self.det_map.get)
        else:
            self.label_map = None
            self.dataset_type = "VG"

        # Load annotations
        self.samples, _ = self._load_mapping(
            self._generate_data_mapping,
            self._filter_data,
            cache_as_binary=cache_as_binary,
            cached_file_path=cached_file_path,
        )

    def __repr__(self) -> str:
        """Concise representation of the dataset."""
        return f"ODVGDataset({self.ann_file})"

    def _filter_data(self, data: list[DictStrAny]) -> list[DictStrAny]:
        """Remove empty samples."""
        samples = []

        if self.dataset_type == "OD":
            frequencies = {cat: 0 for _, cat in self.label_map.items()}

        empty_samples = 0
        for sample in data:
            if self.remove_empty and len(sample["anns"]) == 0:
                empty_samples += 1
                continue

            if self.dataset_type == "OD":
                for ann in sample["anns"]:
                    frequencies[ann["category"]] += 1

            samples.append(sample)

        rank_zero_info(f"Propocessing {self} with {len(samples)} samples.")
        rank_zero_info(f"Filtered {empty_samples} empty samples")

        if self.dataset_type == "OD":
            frequencies = dict(sorted(frequencies.items()))

            print_class_histogram(frequencies)

        return samples

    def _generate_data_mapping(self) -> list[DictStrAny]:
        """Generates the data mapping."""
        with open(osp.join(self.data_root, self.ann_file), "r") as f:
            data_list = [json.loads(line) for line in f]

        if self.with_camera:
            with open(osp.join(self.data_root, "cam_info.json"), "r") as f:
                cameras = json.load(f)

        samples = []
        for data in tqdm(data_list):
            data_info = {}

            if self.dataset_prefix is not None:
                img_path = osp.join(
                    self.data_root, self.dataset_prefix, data["filename"]
                )
            else:
                img_path = osp.join(self.data_root, data["filename"])

            data_info["img_path"] = img_path

            # Pseudo K
            if self.with_camera:
                data_info["K"] = cameras[img_path][0]

            # Pseudo Depth Path
            if self.dataset_prefix is not None:
                depth_path = osp.join(
                    self.data_root,
                    f"{self.dataset_prefix}_depth",
                    data["filename"].replace(".jpg", "_depth.png"),
                )
            else:
                depth_path = osp.join(
                    self.data_root,
                    data["filename"].replace(".jpg", "_depth.png"),
                )
            data_info["depth_path"] = depth_path

            data_info["height"] = data["height"]
            data_info["width"] = data["width"]

            valid_anns = []
            boxes = []
            class_ids = np.empty((0,), dtype=np.int64)[1:]
            if self.dataset_type == "OD":
                instances = data.get("detection", {}).get("instances", [])

                for ann in instances:
                    bbox = ann["bbox"]

                    # Box 2D
                    x1, y1, x2, y2 = bbox
                    inter_w = max(0, min(x2, data["width"]) - max(x1, 0))
                    inter_h = max(0, min(y2, data["height"]) - max(y1, 0))

                    if inter_w * inter_h == 0:
                        continue
                    if (x2 - x1) < 1 or (y2 - y1) < 1:
                        continue

                    boxes.append(bbox)

                    # Class
                    class_ids = np.concatenate(
                        [class_ids, np.array([ann["label"]], dtype=np.int64)]
                    )

                    valid_anns.append(ann)
            else:
                anno = data["grounding"]

                caption = anno["caption"].lower().strip()
                if not caption.endswith("."):
                    caption = caption + ". "

                data_info["caption"] = caption

                regions = anno["regions"]
                phrases = []
                positive_positions = []
                for i, region in enumerate(regions):
                    bboxes = region["bbox"]

                    if not isinstance(bboxes[0], list):
                        bboxes = [bboxes]

                    for bbox in bboxes:
                        x1, y1, x2, y2 = bbox
                        inter_w = max(0, min(x2, data["width"]) - max(x1, 0))
                        inter_h = max(0, min(y2, data["height"]) - max(y1, 0))

                        if inter_w * inter_h == 0:
                            continue
                        if (x2 - x1) < 1 or (y2 - y1) < 1:
                            continue

                        boxes.append(bbox)
                        phrases.append(region["phrase"])
                        positive_positions.append(region["tokens_positive"])
                        valid_anns.append(region)

                        class_ids = np.concatenate(
                            [class_ids, np.array([i], dtype=np.int64)]
                        )

                data_info["phrases"] = phrases
                data_info["positive_positions"] = positive_positions

            boxes2d = (
                np.empty((0, 4), dtype=np.float32)
                if not boxes
                else np.array(boxes, dtype=np.float32)
            )

            data_info["boxes2d"] = boxes2d
            data_info["class_ids"] = class_ids
            data_info["anns"] = valid_anns

            samples.append(data_info)

        del data_list
        return samples

    def get_cat_ids(self, idx: int) -> list[int]:
        """Return the samples."""
        return self.samples[idx]["class_ids"].tolist()

    def __len__(self) -> int:
        """Total number of samples of data."""
        return len(self.samples)

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
        sample_name = sample["img_path"].split("/")[-1]
        data_dict[K.sample_names] = sample_name

        im_bytes = self.data_backend.get(sample["img_path"])
        image = np.ascontiguousarray(
            im_decode(im_bytes, mode=self.image_channel_mode),
            dtype=np.float32,
        )[None]

        data_dict[K.images] = image
        data_dict[K.input_hw] = (image.shape[1], image.shape[2])

        data_dict[K.original_images] = image
        data_dict[K.original_hw] = (image.shape[1], image.shape[2])

        data_dict[K.boxes2d] = sample["boxes2d"]
        data_dict[K.boxes2d_classes] = sample["class_ids"]

        if self.dataset_type == "OD":
            data_dict[K.boxes2d_names] = self.categories
            data_dict["phrases"] = None
            data_dict["positive_positions"] = None
        else:
            data_dict[K.boxes2d_names] = sample["caption"]
            data_dict["phrases"] = sample["phrases"]
            data_dict["positive_positions"] = sample["positive_positions"]

        data_dict["dataset_type"] = self.dataset_type
        data_dict["label_map"] = self.label_map

        self.data_backend.close()

        return data_dict
