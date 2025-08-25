"""Grounding DINO data connector."""

from vis4d.data.const import CommonKeys as K
from vis4d.engine.connectors import data_key, pred_key

CONN_GDINO_LOSS = {
    "all_layers_cls_scores": pred_key("all_layers_cls_scores"),
    "all_layers_bbox_preds": pred_key("all_layers_bbox_preds"),
    "text_token_mask": pred_key("text_token_mask"),
    "enc_cls_scores": pred_key("enc_outputs_class"),
    "enc_bbox_preds": pred_key("enc_outputs_coord"),
    "dn_meta": pred_key("dn_meta"),
    "positive_maps": pred_key("positive_maps"),
    "input_hw": data_key(K.input_hw),
    "batch_gt_boxes": data_key(K.boxes2d),
    "batch_gt_boxes_classes": data_key(K.boxes2d_classes),
}

CONN_BBOX_2D_TRAIN = {
    "images": K.images,
    "input_texts": K.boxes2d_names,
    "input_hw": K.input_hw,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
}

CONN_BBOX_2D_TRAIN_WITH_TOKENS = {
    "images": K.images,
    "input_texts": K.boxes2d_names,
    "input_hw": K.input_hw,
    "boxes2d": K.boxes2d,
    "boxes2d_classes": K.boxes2d_classes,
    "input_tokens_positive": "tokens_positive",
}

CONN_BBOX_2D_TEST = {
    "images": K.images,
    "input_texts": K.boxes2d_names,
    "input_hw": K.input_hw,
    "original_hw": K.original_hw,
}

CONN_BBOX_2D_VIS = {
    "images": data_key(K.original_images),
    "image_names": data_key(K.sample_names),
    "boxes": pred_key("boxes"),
    "scores": pred_key("scores"),
    "class_ids": pred_key("class_ids"),
    "categories": pred_key("categories"),
}

CONN_OMNI3D_DET2D_EVAL = {
    "coco_image_id": data_key(K.sample_names),
    "pred_boxes": pred_key("boxes"),
    "dataset_names": data_key("dataset_name"),
    "pred_scores": pred_key("scores"),
    "pred_classes": pred_key("class_ids"),
}
