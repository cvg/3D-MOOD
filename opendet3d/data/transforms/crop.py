"""Crop transforms."""

from __future__ import annotations

from vis4d.common.typing import (
    NDArrayBool,
    NDArrayF32,
    NDArrayI64,
)
from vis4d.data.const import CommonKeys as K
from vis4d.data.transforms.base import Transform


@Transform(
    in_keys=[
        K.boxes3d,
        K.boxes3d_classes,
        K.boxes3d_track_ids,
        "transforms.crop.keep_mask",
    ],
    out_keys=[K.boxes3d, K.boxes3d_classes, K.boxes3d_track_ids],
)
class CropBoxes3D:
    """Crop 3D bounding boxes."""

    def __call__(
        self,
        boxes_list: list[NDArrayF32],
        classes_list: list[NDArrayI64],
        track_ids_list: list[NDArrayI64] | None,
        keep_mask_list: list[NDArrayBool],
    ) -> tuple[list[NDArrayF32], list[NDArrayI64], list[NDArrayI64] | None]:
        """Crop 3D bounding boxes."""
        for i, (boxes, classes, keep_mask) in enumerate(
            zip(boxes_list, classes_list, keep_mask_list)
        ):
            boxes_list[i] = boxes[keep_mask]
            classes_list[i] = classes[keep_mask]

            if track_ids_list is not None:
                track_ids_list[i] = track_ids_list[i][keep_mask]

        return boxes_list, classes_list, track_ids_list
