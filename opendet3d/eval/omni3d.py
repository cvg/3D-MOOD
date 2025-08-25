"""Omni3D 3D detection evaluation."""

import contextlib
import copy
import io
import itertools
import os
from collections.abc import Sequence

import numpy as np
from terminaltables import AsciiTable
from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import GenericFunc, MetricLogs, NDArrayNumber
from vis4d.eval.base import Evaluator

from opendet3d.data.datasets.omni3d.omni3d_classes import omni3d_class_map
from opendet3d.data.datasets.omni3d.util import get_dataset_det_map

from .detect3d import Detect3Deval, Detect3DEvaluator

omni3d_in = {
    "stationery",
    "sink",
    "table",
    "floor mat",
    "bottle",
    "bookcase",
    "bin",
    "blinds",
    "pillow",
    "bicycle",
    "refrigerator",
    "night stand",
    "chair",
    "sofa",
    "books",
    "oven",
    "towel",
    "cabinet",
    "window",
    "curtain",
    "bathtub",
    "laptop",
    "desk",
    "television",
    "clothes",
    "stove",
    "cup",
    "shelves",
    "box",
    "shoes",
    "mirror",
    "door",
    "picture",
    "lamp",
    "machine",
    "counter",
    "bed",
    "toilet",
}

omni3d_out = {
    "cyclist",
    "pedestrian",
    "trailer",
    "bus",
    "motorcycle",
    "car",
    "barrier",
    "truck",
    "van",
    "traffic cone",
    "bicycle",
}


class Omni3DEvaluator(Evaluator):
    """Omni3D 3D detection evaluator."""

    def __init__(
        self,
        data_root: str = "data/omni3d",
        omni3d50: bool = True,
        datasets: Sequence[str] = (
            "KITTI_test",
            "nuScenes_test",
            "SUNRGBD_test",
            "Hypersim_test",
            "ARKitScenes_test",
            "Objectron_test",
        ),
        per_class_eval: bool = True,
    ) -> None:
        """Initialize the evaluator."""
        super().__init__()
        self.id_to_name = {v: k for k, v in omni3d_class_map.items()}
        self.dataset_names = datasets
        self.per_class_eval = per_class_eval

        # Each dataset evaluator is stored here
        self.evaluators: dict[str, Detect3DEvaluator] = {}

        # These store the evaluations for each category and area,
        # concatenated from ALL evaluated datasets. Doing so avoids
        # the need to re-compute them when accumulating results.
        self.evals_per_cat_area2D = {}
        self.evals_per_cat_area3D = {}

        self.overall_imgIds = set()
        self.overall_catIds = set()

        for dataset_name in self.dataset_names:
            annotation = os.path.join(
                data_root, "annotations", f"{dataset_name}.json"
            )

            det_map = get_dataset_det_map(
                dataset_name=dataset_name, omni3d50=omni3d50
            )

            # create an individual dataset evaluator
            self.evaluators[dataset_name] = Detect3DEvaluator(
                det_map,
                cat_map=omni3d_class_map,
                annotation=annotation,
                eval_prox=(
                    "Objectron" in dataset_name or "SUNRGBD" in dataset_name
                ),
            )

            self.overall_imgIds.update(
                set(self.evaluators[dataset_name]._coco_gt.getImgIds())
            )
            self.overall_catIds.update(
                set(self.evaluators[dataset_name]._coco_gt.getCatIds())
            )

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        datasets_str = ", ".join(self.dataset_names)
        return f"Omni3DEvaluator ({datasets_str})"

    @property
    def metrics(self) -> list[str]:
        """Supported metrics.

        Returns:
            list[str]: Metrics to evaluate.
        """
        return ["2D", "3D"]

    def reset(self) -> None:
        """Reset the saved predictions to start new round of evaluation."""
        for dataset_name in self.dataset_names:
            self.evaluators[dataset_name].reset()
        self.evals_per_cat_area2D.clear()
        self.evals_per_cat_area3D.clear()

    def gather(self, gather_func: GenericFunc) -> None:
        """Accumulate predictions across processes."""
        for dataset_name in self.dataset_names:
            self.evaluators[dataset_name].gather(gather_func)

    def process_batch(
        self,
        coco_image_id: list[int],
        dataset_names: list[str],
        pred_boxes: list[NDArrayNumber],
        pred_scores: list[NDArrayNumber],
        pred_classes: list[NDArrayNumber],
        pred_boxes3d: list[NDArrayNumber] | None = None,
    ) -> None:
        """Process sample and convert detections to coco format."""
        for i, dataset_name in enumerate(dataset_names):
            self.evaluators[dataset_name].process_batch(
                [coco_image_id[i]],
                [pred_boxes[i]],
                [pred_scores[i]],
                [pred_classes[i]],
                pred_boxes3d=[pred_boxes3d[i]] if pred_boxes3d else None,
            )

    def evaluate(self, metric: str) -> tuple[MetricLogs, str]:
        """Evaluate predictions and return the results."""
        assert metric in self.metrics, f"Unsupported metric: {metric}"

        log_dict = {}

        for dataset_name in self.dataset_names:
            rank_zero_info(f"Evaluating {dataset_name}...")
            per_dataset_log_dict, dataset_log_str = self.evaluators[
                dataset_name
            ].evaluate(metric)

            log_dict[f"AP_{dataset_name}"] = per_dataset_log_dict["AP"]

            rank_zero_info(dataset_log_str + "\n")

            # store the partially accumulated evaluations per category per area
            if metric == "2D":
                for key, item in self.evaluators[
                    dataset_name
                ].bbox_2D_evals_per_cat_area.items():
                    if not key in self.evals_per_cat_area2D:
                        self.evals_per_cat_area2D[key] = []
                    self.evals_per_cat_area2D[key] += item
            else:
                for key, item in self.evaluators[
                    dataset_name
                ].bbox_3D_evals_per_cat_area.items():
                    if not key in self.evals_per_cat_area3D:
                        self.evals_per_cat_area3D[key] = []
                    self.evals_per_cat_area3D[key] += item

        results_per_category_dict = {}
        results_per_category = []

        rank_zero_info(f"Evaluating Omni3D for {metric} Detection...")

        evaluator = Detect3Deval(mode=metric)
        evaluator.params.catIds = list(self.overall_catIds)
        evaluator.params.imgIds = list(self.overall_imgIds)
        evaluator.evalImgs = True

        if metric == "2D":
            evaluator.evals_per_cat_area = self.evals_per_cat_area2D
            metrics = ["AP", "AP50", "AP75", "AP95", "APs", "APm", "APl"]
        else:
            evaluator.evals_per_cat_area = self.evals_per_cat_area3D
            metrics = ["AP", "AP15", "AP25", "AP50", "APn", "APm", "APf"]

        evaluator._paramsEval = copy.deepcopy(evaluator.params)

        with contextlib.redirect_stdout(io.StringIO()):
            evaluator.accumulate()
            log_str = "\n" + evaluator.summarize()

        log_dict.update(dict(zip(metrics, evaluator.stats)))

        if self.per_class_eval:
            precisions = evaluator.eval["precision"]
            for idx, cat_id in enumerate(self.overall_catIds):
                cat_name = self.id_to_name[cat_id]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = float(np.mean(precision).item())
                else:
                    ap = float("nan")

                results_per_category_dict[cat_name] = ap
                results_per_category.append((f"{cat_name}", f"{ap:0.3f}"))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(itertools.chain(*results_per_category))
            headers = ["category", "AP"] * (num_columns // 2)
            results_2d = itertools.zip_longest(
                *[results_flatten[i::num_columns] for i in range(num_columns)]
            )
            table_data = [headers] + list(results_2d)
            table = AsciiTable(table_data)
            log_str = f"\n{table.table}\n{log_str}"

        # Omni3D Outdoor performance
        ap_out_lst = []
        for cat in omni3d_out:
            ap_out_lst.append(results_per_category_dict.get(cat, 0.0))

        log_dict["Omni3D_Out"] = np.mean(ap_out_lst).item()

        # Omni3D Indoor performance
        ap_in_lst = []
        for cat in omni3d_in:
            ap_in_lst.append(results_per_category_dict.get(cat, 0.0))

        log_dict["Omni3D_In"] = np.mean(ap_in_lst).item()

        return log_dict, log_str

    def save(self, metric: str, output_dir: str) -> None:
        """Save the results to json files."""
        for dataset_name in self.dataset_names:
            self.evaluators[dataset_name].save(
                metric, output_dir, prefix=dataset_name
            )
