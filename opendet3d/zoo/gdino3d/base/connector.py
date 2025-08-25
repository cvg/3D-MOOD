"""3D G-DINO data connector."""

from ml_collections import ConfigDict
from vis4d.config import class_config
from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import DataConnector, data_key, pred_key

CONN_GDINO3D_LOSS = {
    "all_layers_cls_scores": pred_key("all_layers_cls_scores"),
    "all_layers_bbox_preds": pred_key("all_layers_bbox_preds"),
    "all_layers_bbox_3d_preds": pred_key("all_layers_bbox_3d_preds"),
    "text_token_mask": pred_key("text_token_mask"),
    "enc_cls_scores": pred_key("enc_outputs_class"),
    "enc_bbox_preds": pred_key("enc_outputs_coord"),
    "enc_outputs_3d": pred_key("enc_outputs_3d"),
    "dn_meta": pred_key("dn_meta"),
    "positive_maps": pred_key("positive_maps"),
    "input_hw": data_key(K.input_hw),
    "batch_gt_boxes": data_key(K.boxes2d),
    "batch_gt_boxes_3d": data_key(K.boxes3d),
    "batch_gt_boxes_classes": data_key(K.boxes2d_classes),
    "batch_gt_intrinsics": data_key(K.intrinsics),
}

CONN_BBOX_3D_TRAIN = {
    "images": K.images,
    "input_texts": K.boxes2d_names,
    "input_hw": K.input_hw,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
    "intrinsics": K.intrinsics,
}

CONN_BBOX_3D_TEST = {
    "images": K.images,
    "input_texts": K.boxes2d_names,
    "input_hw": K.input_hw,
    "original_hw": K.original_hw,
    "intrinsics": K.intrinsics,
    "padding": "padding",
    "text_prompt_mapping": "text_prompt_mapping",
}

CONN_COCO_DET3D_EVAL = {
    "coco_image_id": data_key(K.sample_names),
    "pred_boxes": pred_key("boxes"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
    "pred_boxes3d": pred_key("boxes3d"),
}

CONN_OMNI3D_DET3D_EVAL = {
    **CONN_COCO_DET3D_EVAL,
    "dataset_names": data_key("dataset_name"),
}

CONN_BBOX_3D_VIS = {
    "images": data_key(K.original_images),
    "image_names": data_key(K.sample_names),
    "intrinsics": data_key("original_intrinsics"),
    "boxes3d": pred_key("boxes3d"),
    "class_ids": pred_key("class_ids"),
    "scores": pred_key("scores"),
    "categories": pred_key("categories"),
}

CONN_DEPTH_LOSS = {
    "depths": pred_key("depth_maps"),
    "target_depths": data_key(K.depth_maps),
}

CONN_DEPTH_VIS = {
    "images": data_key(K.original_images),
    "image_names": data_key(K.sample_names),
    "depths": pred_key("depth_maps"),
    # "depth_gts": data_key(K.depth_maps),
    "intrinsics": data_key("original_intrinsics"),
}


def get_data_connector_cfg(
    center_padding: bool = True,
    with_text_prompt_mapping: bool = True,
) -> tuple[ConfigDict, ConfigDict]:
    """Get the data connector configuration."""
    train_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_3D_TRAIN
    )

    test_data_connector = class_config(
        DataConnector, key_mapping=CONN_BBOX_3D_TEST
    )

    return train_data_connector, test_data_connector
