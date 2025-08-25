"""Run evaluation results."""

import argparse
import json

from opendet3d.eval.omni3d import Omni3DEvaluator
from opendet3d.eval.detect3d import Detect3DEvaluator
from opendet3d.data.datasets.argoverse import av2_class_map, av2_det_map
from opendet3d.data.datasets.scannet import (
    scannet_class_map,
    scannet_det_map,
)


def eval_omni3d(output_dir: str):
    """Evaluate Omni3D."""
    evaluator = Omni3DEvaluator()

    for dataset in evaluator.dataset_names:
        with open(f"{output_dir}/{dataset}/detect_3D_results.json", "r") as f:
            results = json.load(f)

        evaluator.evaluators[dataset]._predictions = results

    log_dict, log_str = evaluator.evaluate("3D")

    for k, v in log_dict.items():
        print(f"{k}: {v}")

    print(log_str)


def eval_av2(prediction_file_path: str) -> None:
    """Evaluate Argoverse 2."""
    evaluator = Detect3DEvaluator(
        det_map=av2_det_map,
        cat_map=av2_class_map,
        annotation="data/argoverse/annotations/Argoverse_val.json",
        eval_prox=True,
        iou_type="dist",
        num_columns=2,
    )

    with open(prediction_file_path, "r") as f:
        results = json.load(f)

    evaluator._predictions = results

    log_dict, log_str = evaluator.evaluate("3D")

    for k, v in log_dict.items():
        print(f"{k}: {v}")

    print(log_str)


def eval_scannet(prediction_file_path: str) -> None:
    """Evaluate ScanNet."""
    evaluator = Detect3DEvaluator(
        det_map=scannet_det_map,
        cat_map=scannet_class_map,
        annotation="data/scannet/annotations/ScanNet_val.json",
        iou_type="dist",
        num_columns=2,
    )

    with open(prediction_file_path, "r") as f:
        results = json.load(f)

    evaluator._predictions = results

    log_dict, log_str = evaluator.evaluate("3D")

    for k, v in log_dict.items():
        print(f"{k}: {v}")

    print(log_str)


if __name__ == "__main__":
    """Run evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate the prediction results."
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help="Dataset to evaluate"
    )
    parser.add_argument(
        "-p", "--path", required=True, help="Path to the prediction"
    )
    args = parser.parse_args()

    if args.dataset == "omni3d":
        eval_omni3d(args.path)
    elif args.dataset == "av2":
        eval_av2(args.path)
    elif args.dataset == "scannet":
        eval_scannet(args.path)
