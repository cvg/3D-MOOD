"""Callbacks for 3D-MOOD."""

from __future__ import annotations

import os

from ml_collections import ConfigDict, FieldReference
from vis4d.config import class_config
from vis4d.data.const import AxisMode
from vis4d.engine.callbacks import EvaluatorCallback, VisualizerCallback
from vis4d.engine.connectors import CallbackConnector
from vis4d.vis.image.bbox3d_visualizer import BoundingBox3DVisualizer
from vis4d.vis.image.canvas import PillowCanvasBackend
from vis4d.zoo.base import get_default_callbacks_cfg

from opendet3d.data.datasets.argoverse import av2_class_map, av2_det_map
from opendet3d.data.datasets.scannet import (
    scannet200_class_map,
    scannet200_det_map,
    scannet_class_map,
    scannet_det_map,
)
from opendet3d.eval.detect3d import Detect3DEvaluator
from opendet3d.eval.omni3d import Omni3DEvaluator
from opendet3d.eval.open import OpenDetect3DEvaluator
from opendet3d.vis.image.depth_visualizer import DepthVisualizer
from opendet3d.zoo.gdino3d.base.connector import (
    CONN_BBOX_3D_VIS,
    CONN_COCO_DET3D_EVAL,
    CONN_DEPTH_VIS,
    CONN_OMNI3D_DET3D_EVAL,
)


def get_callback_cfg(
    output_dir: str | FieldReference,
    open_test_datasets: list[str] | None,
    omni3d_evaluator: ConfigDict | None = None,
    visualize_depth: bool = True,
) -> list[ConfigDict]:
    """Get callbacks for 3D-MOOD."""
    # Logger
    callbacks = get_default_callbacks_cfg()

    # Evaluator
    if "ScanNet200_val" in open_test_datasets:
        assert (
            len(open_test_datasets) == 1 and omni3d_evaluator is None
        ), "ScanNet200_val should be evaluated alone."

        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=get_scannet_evaluator_cfg(scannet200=True),
                metrics_to_eval=["3D"],
                save_predictions=True,
                output_dir=output_dir,
                save_prefix="detection",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_COCO_DET3D_EVAL
                ),
            )
        )
    elif len(open_test_datasets) > 0:
        evaluators = []
        for dataset in open_test_datasets:
            if dataset == "Argoverse_val":
                evaluators.append(get_av2_evaluator_cfg())
            elif dataset == "ScanNet_val":
                evaluators.append(get_scannet_evaluator_cfg())
            else:
                raise ValueError(
                    f"Unknown dataset {dataset} for open evaluation."
                )

        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=class_config(
                    OpenDetect3DEvaluator,
                    datasets=open_test_datasets,
                    evaluators=evaluators,
                    omni3d_evaluator=omni3d_evaluator,
                ),
                metrics_to_eval=["3D"],
                save_predictions=True,
                output_dir=output_dir,
                save_prefix="detection",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_OMNI3D_DET3D_EVAL
                ),
            )
        )
    else:
        assert omni3d_evaluator is not None, "No evaluator provided."
        callbacks.append(
            class_config(
                EvaluatorCallback,
                evaluator=omni3d_evaluator,
                metrics_to_eval=["3D"],
                save_predictions=True,
                output_dir=output_dir,
                save_prefix="detection",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_OMNI3D_DET3D_EVAL
                ),
            )
        )

    # Visualizer
    callbacks.extend(
        get_visualizer_callback_cfg(
            output_dir, visualize_depth=visualize_depth
        )
    )

    return callbacks


def get_omni3d_evaluator_cfg(
    data_root: str,
    omni3d50: bool,
    test_datasets: list[str],
) -> ConfigDict:
    """Get Omni3D evaluator config."""
    return class_config(
        Omni3DEvaluator,
        data_root=data_root,
        omni3d50=omni3d50,
        datasets=test_datasets,
    )


def get_av2_evaluator_cfg(data_root: str = "data/argoverse") -> ConfigDict:
    """Get Argoverse 2 evaluator config."""
    return class_config(
        Detect3DEvaluator,
        det_map=av2_det_map,
        cat_map=av2_class_map,
        eval_prox=True,
        iou_type="dist",
        num_columns=2,
        annotation=os.path.join(data_root, "annotations/Argoverse_val.json"),
        base_classes=[
            "regular vehicle",
            "pedestrian",
            "bicyclist",
            "construction cone",
            "construction barrel",
            "large vehicle",
            "bus",
            "truck",
            "vehicular trailer",
            "bicycle",
            "motorcycle",
        ],
    )


def get_scannet_evaluator_cfg(
    data_root: str = "data/scannet", scannet200: bool = False
) -> ConfigDict:
    """Get ScanNet evaluator config."""
    if scannet200:
        s_det_map = scannet200_det_map
        s_class_map = scannet200_class_map
        annotation = os.path.join(data_root, "annotations/ScanNet200_val.json")
        base_classes = None
    else:
        s_det_map = scannet_det_map
        s_class_map = scannet_class_map
        annotation = os.path.join(data_root, "annotations/ScanNet_val.json")
        base_classes = [
            "cabinet",
            "bed",
            "chair",
            "sofa",
            "table",
            "door",
            "window",
            "picture",
            "counter",
            "desk",
            "curtain",
            "refrigerator",
            "toilet",
            "sink",
            "bathtub",
        ]

    return class_config(
        Detect3DEvaluator,
        det_map=s_det_map,
        cat_map=s_class_map,
        iou_type="dist",
        num_columns=2,
        annotation=annotation,
        base_classes=base_classes,
    )


def get_visualizer_callback_cfg(
    output_dir: str | FieldReference,
    visualize_depth: bool = False,
    vis_freq: int = 50,
    width: int = 4,
    font_size: int = 16,
    save_boxes3d: bool = True,
) -> list[ConfigDict]:
    """Get basic callbacks."""
    callbacks = []

    callbacks.append(
        class_config(
            VisualizerCallback,
            visualizer=class_config(
                BoundingBox3DVisualizer,
                axis_mode=AxisMode.OPENCV,
                width=width,
                camera_near_clip=0.01,
                plot_heading=False,
                vis_freq=vis_freq,
                plot_trajectory=False,
                canvas=class_config(PillowCanvasBackend, font_size=font_size),
                save_boxes3d=save_boxes3d,
            ),
            output_dir=output_dir,
            save_prefix="box3d",
            test_connector=class_config(
                CallbackConnector, key_mapping=CONN_BBOX_3D_VIS
            ),
        )
    )

    if visualize_depth:
        callbacks.append(
            class_config(
                VisualizerCallback,
                visualizer=class_config(
                    DepthVisualizer,
                    plot_error=False,
                    lift=True,
                    vis_freq=vis_freq,
                ),
                output_dir=output_dir,
                save_prefix="depth",
                test_connector=class_config(
                    CallbackConnector, key_mapping=CONN_DEPTH_VIS
                ),
            )
        )

    return callbacks
