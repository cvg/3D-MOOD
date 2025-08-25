"""Grounding DINO data config."""

from .coco import get_coco_detection_test_cfg, get_coco_detection_train_cfg
from .o365 import get_objects365v1_detection_train_cfg

__all__ = [
    "get_coco_detection_train_cfg",
    "get_coco_detection_test_cfg",
    "get_objects365v1_detection_train_cfg",
]
