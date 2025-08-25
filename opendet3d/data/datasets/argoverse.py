"""Argoverse V2 Sensor dataset."""

from __future__ import annotations

from vis4d.common.typing import ArgsType, DictStrAny

from .coco3d import COCO3DDataset

TRAIN_SAMPLE_RATE = 10
VAL_SAMPLE_RATE = 5
ACC_FRAMES = 5


av2_class_map = {
    "regular vehicle": 0,
    "pedestrian": 1,
    "bicyclist": 2,
    "motorcyclist": 3,
    "wheeled rider": 4,
    "bollard": 5,
    "construction cone": 6,
    "sign": 7,
    "construction barrel": 8,
    "stop sign": 9,
    "mobile pedestrian crossing sign": 10,
    "large vehicle": 11,
    "bus": 12,
    "box truck": 13,
    "truck": 14,
    "vehicular trailer": 15,
    "truck cab": 16,
    "school bus": 17,
    "articulated bus": 18,
    "message board trailer": 19,
    "bicycle": 20,
    "motorcycle": 21,
    "wheeled device": 22,
    "wheelchair": 23,
    "stroller": 24,
    "dog": 25,
}

av2_det_map = {
    "regular vehicle": 0,
    "pedestrian": 1,
    "bicyclist": 2,
    "motorcyclist": 3,
    "wheeled rider": 4,
    "bollard": 5,
    "construction cone": 6,
    "sign": 7,
    "construction barrel": 8,
    "stop sign": 9,
    "mobile pedestrian crossing sign": 10,
    "large vehicle": 11,
    "bus": 12,
    "box truck": 13,
    "truck": 14,
    "vehicular trailer": 15,
    "truck cab": 16,
    "school bus": 17,
    "articulated bus": 18,
    "bicycle": 19,
    "motorcycle": 20,
    "wheeled device": 21,
    "stroller": 22,
}


class AV2SensorDataset(COCO3DDataset):
    """Argoverse V2 Sensor dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = av2_class_map,
        max_depth: float = 80.0,
        depth_scale: float = 256.0,
        **kwargs: ArgsType,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__(
            class_map=class_map,
            max_depth=max_depth,
            depth_scale=depth_scale,
            **kwargs,
        )

    def get_depth_filenames(self, img: DictStrAny) -> str | None:
        """Get the depth filenames."""
        return (
            img["file_path"]
            .replace("images", "depth")
            .replace(".jpg", "_depth.png")
        )
