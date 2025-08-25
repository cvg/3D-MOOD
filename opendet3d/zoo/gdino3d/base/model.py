"""3D-MOOD model config."""

from __future__ import annotations

from ml_collections import ConfigDict, FieldReference
from vis4d.config import class_config
from vis4d.config.typing import ExperimentParameters
from vis4d.op.fpp.fpn import FPN

from opendet3d.model.detect3d.grounding_dino_3d import GroundingDINO3D
from opendet3d.op.base.swin import SwinTransformer
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DCoder,
    GroundingDINO3DHead,
    RoI2Det3D,
    UniDepthHead,
)
from opendet3d.op.fpp.channel_mapper import ChannelMapper
from opendet3d.zoo.gdino.base.model import GDINO_MODEL_WEIGHTS


def get_gdino3d_hyperparams_cfg() -> ExperimentParameters:
    """Get the hyperparameters for 3D-MOOD."""
    params = ExperimentParameters()

    # Training
    params.samples_per_gpu = 2
    params.workers_per_gpu = 4
    params.accumulate_grad_batches = 1
    params.lr = 0.0004  # bs=128, lr=0.0004
    params.weight_decay = 0.0001

    # Learning rate schedule
    params.num_epochs = 120
    params.step_1 = 80
    params.step_2 = 110
    params.check_val_every_n_epoch = 1

    # Grounding DINO 3D Coder
    params.center_scale = 10.0
    params.depth_scale = 2.0
    params.dim_scale = 2.0
    params.orientation = "rotation_6d"

    # Grounding DINO 3D Loss
    params.loss_center_weight = 1.0
    params.loss_depth_weight = 1.0
    params.loss_dim_weight = 1.0
    params.loss_rot_weight = 1.0

    # Aux Depth Loss
    params.si_log_weight = 10.0

    # RoI2Det3D
    params.nms = False
    params.class_agnostic_nms = False
    params.max_per_img = 100
    params.score_threshold = 0.0
    params.iou_threshold = 0.5

    # Depth Head
    params.depth_output_scales = 1

    return params


def get_gdino3d_head_cfg(params: ExperimentParameters) -> ConfigDict:
    """Get the G-DINO 3D head config."""
    box_coder = class_config(
        GroundingDINO3DCoder,
        center_scale=params.center_scale,
        depth_scale=params.depth_scale,
        dim_scale=params.dim_scale,
        orientation=params.orientation,
    )

    bbox3d_head = class_config(
        GroundingDINO3DHead,
        box_coder=box_coder,
        depth_output_scales=params.depth_output_scales,
    )

    roi2det3d = class_config(
        RoI2Det3D,
        nms=params.nms,
        max_per_img=params.max_per_img,
        class_agnostic_nms=params.class_agnostic_nms,
        score_threshold=params.score_threshold,
        iou_threshold=params.iou_threshold,
        box_coder=box_coder,
    )

    return bbox3d_head, roi2det3d, box_coder


def get_gdino3d_cfg(
    params: ExperimentParameters,
    basemodel: ConfigDict,
    neck: ConfigDict,
    depth_fpn: ConfigDict,
    num_feature_levels: int = 4,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
) -> ConfigDict:
    """Get the Grounding DINO with Swin-B model config."""
    # UniDepth Head
    depth_head = class_config(
        UniDepthHead,
        depth_scale=params.depth_scale,
        input_dims=[256, 256, 256, 256],
        output_scales=params.depth_output_scales,
    )

    bbox3d_head, roi2det3d, box_coder = get_gdino3d_head_cfg(params=params)

    if pretrained is not None:
        weights = GDINO_MODEL_WEIGHTS[pretrained]
    else:
        weights = None

    model = class_config(
        GroundingDINO3D,
        basemodel=basemodel,
        neck=neck,
        num_feature_levels=num_feature_levels,
        bbox3d_head=bbox3d_head,
        roi2det3d=roi2det3d,
        fpn=depth_fpn,
        depth_head=depth_head,
        use_checkpoint=use_checkpoint,
        weights=weights,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
    )

    return model, box_coder


def get_gdino3d_swin_tiny_cfg(
    params: ExperimentParameters,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
) -> ConfigDict:
    """Get the config of Swin-Tiny."""
    basemodel = class_config(
        SwinTransformer,
        convert_weights=True,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,
        out_indices=(0, 1, 2, 3),
        with_cp=use_checkpoint,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
    )

    neck = class_config(
        ChannelMapper,
        in_channels=[192, 384, 768],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm="GroupNorm",
        num_groups=32,
        activation=None,
        bias=True,
    )

    depth_fpn = class_config(
        FPN,
        in_channels_list=[96, 192, 384, 768],
        out_channels=256,
        extra_blocks=None,
        start_index=0,
    )

    return get_gdino3d_cfg(
        params,
        basemodel=basemodel,
        neck=neck,
        depth_fpn=depth_fpn,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
    )


def get_gdino3d_swin_base_cfg(
    params: ExperimentParameters,
    chunked_size: int = -1,
    cat_mapping: dict[str, int] | None = None,
    pretrained: str | None = None,
    use_checkpoint: bool | FieldReference = False,
) -> ConfigDict:
    """Get the config of Swin-Base."""
    basemodel = class_config(
        SwinTransformer,
        convert_weights=True,
        pretrain_img_size=384,
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.3,
        out_indices=(0, 1, 2, 3),
        with_cp=use_checkpoint,
        pretrained="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384_22k.pth",
    )

    neck = class_config(
        ChannelMapper,
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_outs=4,
        kernel_size=1,
        norm="GroupNorm",
        num_groups=32,
        activation=None,
        bias=True,
    )

    depth_fpn = class_config(
        FPN,
        in_channels_list=[128, 256, 512, 1024],
        out_channels=256,
        extra_blocks=None,
        start_index=0,
    )

    return get_gdino3d_cfg(
        params,
        basemodel=basemodel,
        neck=neck,
        depth_fpn=depth_fpn,
        chunked_size=chunked_size,
        cat_mapping=cat_mapping,
        pretrained=pretrained,
        use_checkpoint=use_checkpoint,
    )
