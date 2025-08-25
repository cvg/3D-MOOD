"""Multi-data 3D detection evaluation."""

from collections.abc import Sequence

from vis4d.common.logging import rank_zero_info
from vis4d.common.typing import GenericFunc, MetricLogs, NDArrayNumber
from vis4d.eval.base import Evaluator

from .detect3d import Detect3DEvaluator
from .omni3d import Omni3DEvaluator


class OpenDetect3DEvaluator(Evaluator):
    """Multi-data 3D detection evaluator."""

    def __init__(
        self,
        datasets: Sequence[str],
        evaluators: Sequence[Detect3DEvaluator],
        omni3d_evaluator: Omni3DEvaluator | None = None,
    ) -> None:
        """Initialize the evaluator."""
        super().__init__()
        self.dataset_names = datasets
        self.evaluators = {
            name: evaluator for name, evaluator in zip(datasets, evaluators)
        }

        self.omni3d_evaluator = omni3d_evaluator

    def __repr__(self) -> str:
        """Returns the string representation of the object."""
        datasets_str = ", ".join(self.dataset_names)
        return f"Open 3D Object Detection Evaluator ({datasets_str})"

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

        if self.omni3d_evaluator is not None:
            self.omni3d_evaluator.reset()

    def gather(self, gather_func: GenericFunc) -> None:
        """Accumulate predictions across processes."""
        for dataset_name in self.dataset_names:
            self.evaluators[dataset_name].gather(gather_func)

        if self.omni3d_evaluator is not None:
            self.omni3d_evaluator.gather(gather_func)

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
            if (
                self.omni3d_evaluator is not None
                and dataset_name in self.omni3d_evaluator.dataset_names
            ):
                self.omni3d_evaluator.process_batch(
                    [coco_image_id[i]],
                    [dataset_name],
                    [pred_boxes[i]],
                    [pred_scores[i]],
                    [pred_classes[i]],
                    pred_boxes3d=[pred_boxes3d[i]] if pred_boxes3d else None,
                )
            else:
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
        log_str = ""

        if self.omni3d_evaluator is not None:
            log_dict_omni3d, omni3d_log_str = self.omni3d_evaluator.evaluate(
                metric
            )

            log_dict.update(log_dict_omni3d)
            log_str += omni3d_log_str

        for dataset_name in self.dataset_names:
            rank_zero_info(f"Evaluating {dataset_name}...")
            per_dataset_log_dict, dataset_log_str = self.evaluators[
                dataset_name
            ].evaluate(metric)

            if "ODS" in per_dataset_log_dict:
                score = "ODS"
            else:
                score = "AP"

            log_dict[f"{score}_{dataset_name}"] = per_dataset_log_dict[score]

            if self.evaluators[dataset_name].base_classes is not None:
                log_dict[f"{score}_Base_{dataset_name}"] = (
                    per_dataset_log_dict[f"{score}_Base"]
                )
                log_dict[f"{score}_Novel_{dataset_name}"] = (
                    per_dataset_log_dict[f"{score}_Novel"]
                )

            log_str += f"\nCheck {dataset_name} results in log dict."

            rank_zero_info(dataset_log_str + "\n")

        return log_dict, log_str

    def save(self, metric: str, output_dir: str) -> None:
        """Save the results to json files."""
        for dataset_name in self.dataset_names:
            self.evaluators[dataset_name].save(
                metric, output_dir, prefix=dataset_name
            )
