"""Language related transforms."""

from __future__ import annotations

import random
import re

import numpy as np
from transformers import AutoTokenizer
from vis4d.common.logging import rank_zero_warn
from vis4d.common.typing import NDArrayF32, NDArrayI64
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.base import Transform


def clean_name(name: str) -> str:
    """Clean the name."""
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    name = name.lower()
    return name


def generate_senetence_given_labels(
    positive_label_list: list[int],
    negative_label_list: list[str],
    label_map: dict[str, str],
) -> tuple[dict[int, list[list[int]]], str, dict[int, int]]:
    """Generate a sentence given positive and negative labels."""
    label_to_positions = {}

    label_list = negative_label_list + positive_label_list

    random.shuffle(label_list)

    pheso_caption = ""

    label_remap_dict = {}
    for index, label in enumerate(label_list):
        start_index = len(pheso_caption)

        pheso_caption += clean_name(label_map[str(label)])

        end_index = len(pheso_caption)

        if label in positive_label_list:
            label_to_positions[index] = [[start_index, end_index]]
            label_remap_dict[int(label)] = index

        pheso_caption += ". "

    return label_to_positions, pheso_caption, label_remap_dict


@Transform(
    [
        "dataset_type",
        K.boxes2d,
        K.boxes2d_classes,
        K.boxes2d_names,
        "label_map",
        "positive_positions",
    ],
    [K.boxes2d, K.boxes2d_classes, K.boxes2d_names, "tokens_positive"],
)
class RandomSamplingNegPos:
    """Randomly sample negative and positive labels for object detection."""

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        num_sample_negative: int = 85,
        max_tokens: int = 256,
        full_sampling_prob: float = 0.5,
    ) -> None:
        """Creates an instance of RandomSamplingNegPos."""
        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers is not installed, please install it by: "
                "pip install transformers."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.num_sample_negative = num_sample_negative
        self.full_sampling_prob = full_sampling_prob
        self.max_tokens = max_tokens

    def __call__(
        self,
        dataset_type_list: list[str],
        boxes_list: list[NDArrayF32],
        class_ids_list: list[NDArrayI64],
        texts_list: list[str] | None = None,
        label_map_list: dict | None = None,
        positive_positions_list: list[dict] | None = None,
    ) -> tuple[
        list[NDArrayF32],
        list[NDArrayI64],
        list[str],
        list[dict[int, list[list[int]]]],
    ]:
        """Randomly sample negative and positive labels."""
        new_texts_list = []
        tokens_positive_list = []
        for i, (boxes, class_ids) in enumerate(
            zip(boxes_list, class_ids_list)
        ):
            if dataset_type_list[i] == "OD":
                assert (
                    label_map_list[i] is not None
                ), "label_map should not be None"
                boxes_list[i], class_ids_list[i], text, tokens_positive = (
                    self.od_aug(boxes, class_ids, label_map_list[i])
                )
                new_texts_list.append(text)
                tokens_positive_list.append(tokens_positive)
            else:
                assert (
                    positive_positions_list[i] is not None
                ), "positive_positions should not be None"
                tokens_positive = self.vg_aug(
                    class_ids, positive_positions_list[i]
                )
                new_texts_list.append(texts_list[i])
                tokens_positive_list.append(tokens_positive)

        return boxes_list, class_ids_list, new_texts_list, tokens_positive_list

    def vg_aug(self, class_ids: NDArrayI64, positive_positions):
        """Visual Genome data augmentation."""
        positive_label_list = np.unique(class_ids).tolist()

        label_to_positions = {}
        for label in positive_label_list:
            label_to_positions[label] = positive_positions[label]

        return label_to_positions

    def od_aug(
        self,
        boxes: NDArrayF32,
        class_ids: NDArrayI64,
        label_map: dict,
    ) -> tuple[NDArrayF32, NDArrayI64, str, dict[int, list[list[int]]]]:
        """Object detection data augmentation."""
        original_box_num = len(class_ids)

        # If the category name is in the format of 'a/b' (in object365),
        # we randomly select one of them.
        for key, value in label_map.items():
            if "/" in value:
                label_map[key] = random.choice(value.split("/")).strip()

        keep_box_index, class_ids, positive_caption_length = (
            self.check_for_positive_overflow(class_ids, label_map)
        )

        boxes = boxes[keep_box_index]

        if len(boxes) < original_box_num:
            rank_zero_warn(
                f"Remove {original_box_num - len(boxes)} boxes due to "
                "positive caption overflow."
            )

        valid_negative_indexes = list(label_map.keys())

        positive_label_list = np.unique(class_ids).tolist()

        full_negative = self.num_sample_negative
        if full_negative > len(valid_negative_indexes):
            full_negative = len(valid_negative_indexes)

        outer_prob = random.random()

        if outer_prob < self.full_sampling_prob:
            # c. probability_full: add both all positive and all negatives
            num_negatives = full_negative
        else:
            if random.random() < 1.0:
                num_negatives = np.random.choice(max(1, full_negative)) + 1
            else:
                num_negatives = full_negative

        # Keep some negatives
        negative_label_list = set()
        if num_negatives != -1:
            if num_negatives > len(valid_negative_indexes):
                num_negatives = len(valid_negative_indexes)

            for i in np.random.choice(
                valid_negative_indexes, size=num_negatives, replace=False
            ):
                if int(i) not in positive_label_list:
                    negative_label_list.add(i)

        random.shuffle(positive_label_list)

        negative_label_list = list(negative_label_list)
        random.shuffle(negative_label_list)

        negative_max_length = self.max_tokens - positive_caption_length
        screened_negative_label_list = []

        for negative_label in negative_label_list:
            label_text = clean_name(label_map[str(negative_label)]) + ". "

            tokenized = self.tokenizer.tokenize(label_text)

            negative_max_length -= len(tokenized)

            if negative_max_length > 0:
                screened_negative_label_list.append(negative_label)
            else:
                break

        negative_label_list = screened_negative_label_list
        label_to_positions, pheso_caption, label_remap_dict = (
            generate_senetence_given_labels(
                positive_label_list, negative_label_list, label_map
            )
        )

        # label remap
        if len(class_ids) > 0:
            class_ids = np.vectorize(lambda x: label_remap_dict[x])(class_ids)

        return boxes, class_ids, pheso_caption, label_to_positions

    def check_for_positive_overflow(
        self, class_ids: NDArrayI64, label_map: dict[str, str]
    ) -> tuple[list[int], NDArrayI64, int]:
        """Check if having too many positive labels."""
        # generate a caption by appending the positive labels
        positive_label_list = np.unique(class_ids).tolist()

        # random shuffule so we can sample different annotations
        # at different epochs
        random.shuffle(positive_label_list)

        kept_lables = []
        length = 0
        for _, label in enumerate(positive_label_list):
            label_text = clean_name(label_map[str(label)]) + ". "

            tokenized = self.tokenizer.tokenize(label_text)

            length += len(tokenized)

            if length > self.max_tokens:
                break
            else:
                kept_lables.append(label)

        keep_box_index = []
        keep_gt_labels = []
        for i, class_id in enumerate(class_ids):
            if class_id in kept_lables:
                keep_box_index.append(i)
                keep_gt_labels.append(class_id)

        return (
            keep_box_index,
            np.array(keep_gt_labels, dtype=np.int64),
            length,
        )
