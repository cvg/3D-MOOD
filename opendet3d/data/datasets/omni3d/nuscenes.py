"""nuScenes from Omni3D."""

from __future__ import annotations

from vis4d.common.typing import ArgsType, DictStrAny

from opendet3d.data.datasets.coco3d import COCO3DDataset

from .omni3d_classes import omni3d_class_map

nusc_det_map = {
    "bicycle": 0,
    "motorcycle": 1,
    "pedestrian": 2,
    "bus": 3,
    "car": 4,
    "trailer": 5,
    "truck": 6,
    "traffic cone": 7,
    "barrier": 8,
}


class nuScenes(COCO3DDataset):
    """nuScenes dataset."""

    def __init__(
        self,
        class_map: dict[str, int] = omni3d_class_map,
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
        """Get the depth filenames.

        Since not every data has depth.
        """
        img["file_path"] = img["file_path"].replace("nuScenes", "nuscenes")

        depth_filename = (
            img["file_path"]
            .replace("nuscenes", "nuscenes_depth")
            .replace("jpg", "png")
        )
        return depth_filename

    def get_cat_ids(self, idx: int) -> list[int]:
        """Return the samples."""
        return self.samples[idx]["class_ids"].tolist()

    def __len__(self) -> int:
        """Total number of samples of data."""
        return len(self.samples)
