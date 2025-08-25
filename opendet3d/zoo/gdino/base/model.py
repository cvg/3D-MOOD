"""Grounding DINO model config."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference
from vis4d.config import class_config

from opendet3d.model.detect.grounding_dino import GroundingDINO
from opendet3d.op.detect.grounding_dino import RoI2Det

GDINO_MODEL_WEIGHTS = {
    "mm_gdino_swin_tiny_obj365_goldg_grit9m_v3det": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth",
    "mm_gdino_swin_base_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-b_pretrain_all/grounding_dino_swin-b_pretrain_all-f9818a7c.pth",
    "mm_gdino_swin_large_all": "https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_all/grounding_dino_swin-l_pretrain_all-56d69e78.pth",
    "gdino_swin_tiny_obj365_goldg_cap4m": "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swint_ogc_mmdet-822d7e9d.pth",
    "gdino_swin_base_mixeddata": "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth",
}


def get_gdino_cfg(
    basemodel: ConfigDict,
    neck: ConfigDict,
    num_feature_levels: int = 4,
    nms: bool | FieldReference = False,
    max_per_img: int | FieldReference = 300,
    class_agnostic_nms: bool | FieldReference = False,
    score_threshold: float | FieldReference = 0.0,
    iou_threshold: float | FieldReference = 0.5,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
) -> ConfigDict:
    """Get the Grounding DINO with Swin-B model config."""
    roi2det = class_config(
        RoI2Det,
        nms=nms,
        max_per_img=max_per_img,
        class_agnostic_nms=class_agnostic_nms,
        score_threshold=score_threshold,
        iou_threshold=iou_threshold,
    )

    if pretrained is not None:
        weights = GDINO_MODEL_WEIGHTS[pretrained]
    else:
        weights = None

    model = class_config(
        GroundingDINO,
        basemodel=basemodel,
        neck=neck,
        num_feature_levels=num_feature_levels,
        roi2det=roi2det,
        use_checkpoint=use_checkpoint,
        weights=weights,
    )

    return model
