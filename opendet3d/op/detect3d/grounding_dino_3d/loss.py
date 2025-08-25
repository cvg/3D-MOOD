"""3D-MOOD loss."""

from __future__ import annotations

import torch
from torch import Tensor
from vis4d.common.distributed import reduce_mean
from vis4d.common.typing import ArgsType
from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss

from opendet3d.op.box.box2d import bbox_cxcywh_to_xyxy
from opendet3d.op.detect.grounding_dino.loss import (
    GroundingDINOLoss,
    split_outputs,
)
from opendet3d.op.util import multi_apply

from .coder import GroundingDINO3DCoder


class GroundingDINO3DLoss(GroundingDINOLoss):
    """Grounding DINO with 3D loss."""

    def __init__(
        self,
        *args: ArgsType,
        box_coder: GroundingDINO3DCoder | None = None,
        loss_center_weight: float = 1.0,
        loss_depth_weight: float = 1.0,
        loss_dim_weight: float = 1.0,
        loss_rot_weight: float = 1.0,
        **kwargs: ArgsType,
    ):
        """Init."""
        super().__init__(*args, **kwargs)
        self.box_coder = box_coder or GroundingDINO3DCoder()

        self.reg_dims = self.box_coder.reg_dims

        self.loss_center_weight = loss_center_weight
        self.loss_depth_weight = loss_depth_weight
        self.loss_dim_weight = loss_dim_weight
        self.loss_rot_weight = loss_rot_weight

    def get_targets_3d(
        self,
        cls_scores_list: list[Tensor],
        bbox_preds_list: list[Tensor],
        bbox_preds_3d_list: list[Tensor],
        input_hw: list[tuple[int, int]],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_3d: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        batch_gt_intrinsics: list[Tensor],
        positive_maps: list[Tensor],
        text_token_mask: Tensor,
    ) -> tuple:
        """Compute regression and classification targets for a batch image."""
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            bbox_targets_3d_list,
            bbox_weights_3d_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_targets_3d_single,
            cls_scores_list,
            bbox_preds_list,
            bbox_preds_3d_list,
            input_hw,
            batch_gt_boxes,
            batch_gt_boxes_3d,
            batch_gt_boxes_classes,
            batch_gt_intrinsics,
            positive_maps,
            text_token_mask,
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            bbox_targets_3d_list,
            bbox_weights_3d_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_targets_3d_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        bbox_pred_3d: Tensor,
        input_hw: tuple[int, int],
        gt_boxes: Tensor,
        gt_boxes_3d: Tensor,
        gt_classes: Tensor,
        gt_intrinsics: Tensor,
        positive_map: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for one image."""
        # 2D Target
        with torch.no_grad():
            (
                labels,
                label_weights,
                bbox_targets,
                bbox_weights,
                pos_pred_boxes2d,
                pos_inds,
                neg_inds,
                pos_assigned_gt_inds,
            ) = self._get_targets_2d_single(
                cls_score,
                bbox_pred,
                input_hw,
                gt_boxes,
                gt_classes,
                positive_map,
                text_token_mask,
            )

        # 3D Target
        pos_gt_boxes3d = gt_boxes_3d[pos_assigned_gt_inds.long(), :]

        pos_gt_bboxes_3d, pos_gt_bboxes_3d_weights = self.box_coder.encode(
            pos_pred_boxes2d, pos_gt_boxes3d, gt_intrinsics
        )

        bbox_targets_3d = torch.zeros_like(bbox_pred_3d)
        bbox_targets_3d[pos_inds] = pos_gt_bboxes_3d

        bbox_weights_3d = torch.zeros_like(bbox_pred_3d)
        bbox_weights_3d[pos_inds] = pos_gt_bboxes_3d_weights

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            bbox_targets_3d,
            bbox_weights_3d,
            pos_inds,
            neg_inds,
        )

    def loss_3d_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        bbox_3d_preds: Tensor,
        text_token_mask: Tensor,
        input_hw: list[tuple[int, int]],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_3d: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        batch_gt_intrinsics: list[Tensor],
        positive_maps: list[Tensor],
    ):
        """Loss function for outputs from a single decoder layer."""
        num_imgs = cls_scores.size(0)

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        bbox_preds_3d_list = [bbox_3d_preds[i] for i in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            bbox_targets_3d_list,
            bbox_weights_3d_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_targets_3d(
            cls_scores_list,
            bbox_preds_list,
            bbox_preds_3d_list,
            input_hw,
            batch_gt_boxes,
            batch_gt_boxes_3d,
            batch_gt_boxes_classes,
            batch_gt_intrinsics,
            positive_maps,
            text_token_mask,
        )

        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_targets_3d = torch.cat(bbox_targets_3d_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        bbox_weights_3d = torch.cat(bbox_weights_3d_list, 0)

        # Loss is not computed for the padded regions of the text.
        assert text_token_mask.dim() == 2
        text_masks = text_token_mask.new_zeros(
            (text_token_mask.size(0), self.max_text_len)
        )
        text_masks[:, : text_token_mask.size(1)] = text_token_mask
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, cls_scores.size(1), 1)
        cls_scores = torch.masked_select(cls_scores, text_mask).contiguous()

        labels = torch.masked_select(labels, text_mask)
        label_weights = label_weights[..., None].repeat(
            1, 1, text_mask.size(-1)
        )
        label_weights = torch.masked_select(label_weights, text_mask)

        # classification loss
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = (
            num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        )
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor])
            )
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.cls_loss_weight * self.loss_cls(
            cls_scores,
            labels,
            reducer=SumWeightedLoss(
                weight=label_weights, avg_factor=cls_avg_factor
            ),
        )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_hw, bbox_pred in zip(input_hw, bbox_preds):
            img_h, img_w = img_hw
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression L1 loss
        loss_bbox = self.bbox_loss_weight * self.loss_bbox(
            bbox_preds,
            bbox_targets,
            reducer=SumWeightedLoss(
                weight=bbox_weights, avg_factor=num_total_pos
            ),
        )

        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.iou_loss_weight * self.loss_iou(
            bboxes,
            bboxes_gt,
            reducer=SumWeightedLoss(
                weight=bbox_weights.mean(-1), avg_factor=num_total_pos
            ),
        )

        # 3D Loss
        bbox_3d_preds = bbox_3d_preds.reshape(-1, self.reg_dims)

        # Delta 2D center Loss
        loss_cen = self.loss_center_weight * l1_loss(
            bbox_3d_preds[:, :2],
            bbox_targets_3d[:, :2],
            reducer=SumWeightedLoss(
                weight=bbox_weights_3d[:, :2], avg_factor=num_total_pos
            ),
        )

        # Depth Loss
        loss_depth = self.loss_depth_weight * l1_loss(
            bbox_3d_preds[:, 2],
            bbox_targets_3d[:, 2],
            reducer=SumWeightedLoss(
                weight=bbox_weights_3d[:, 2], avg_factor=num_total_pos
            ),
        )

        # Dimension Loss
        loss_dim = self.loss_dim_weight * l1_loss(
            bbox_3d_preds[:, 3:6],
            bbox_targets_3d[:, 3:6],
            reducer=SumWeightedLoss(
                weight=bbox_weights_3d[:, 3:6], avg_factor=num_total_pos
            ),
        )

        # Rotation Loss
        loss_rot = self.loss_rot_weight * l1_loss(
            bbox_3d_preds[:, 6:],
            bbox_targets_3d[:, 6:],
            reducer=SumWeightedLoss(
                weight=bbox_weights_3d[:, 6:], avg_factor=num_total_pos
            ),
        )

        return (
            loss_cls,
            loss_bbox,
            loss_iou,
            loss_cen,
            loss_depth,
            loss_dim,
            loss_rot,
        )

    def forward(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        all_layers_bbox_3d_preds: Tensor,
        text_token_mask: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        enc_outputs_3d: Tensor,
        dn_meta: dict[str, int],
        positive_maps: list[Tensor],
        input_hw: list[tuple[int, int]],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_3d: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        batch_gt_intrinsics: list[Tensor],
    ) -> dict[str, Tensor]:
        """Forward pass of the 3D Grounding DINO loss."""
        (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
        ) = split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta
        )

        (
            losses_cls,
            losses_bbox,
            losses_iou,
            losses_cen,
            losses_depth,
            losses_dim,
            losses_rot,
        ) = multi_apply(
            self.loss_3d_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_bbox_3d_preds,
            text_token_mask=text_token_mask,
            input_hw=input_hw,
            batch_gt_boxes=batch_gt_boxes,
            batch_gt_boxes_3d=batch_gt_boxes_3d,
            batch_gt_boxes_classes=batch_gt_boxes_classes,
            batch_gt_intrinsics=batch_gt_intrinsics,
            positive_maps=positive_maps,
        )

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]
        loss_dict["loss_delta_2d"] = losses_cen[-1]
        loss_dict["loss_depth"] = losses_depth[-1]
        loss_dict["loss_dim"] = losses_dim[-1]
        loss_dict["loss_rot"] = losses_rot[-1]

        # loss from other decoder layers
        for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in enumerate(
            zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1])
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_delta_2d"] = losses_cen[
                num_dec_layer
            ]
            loss_dict[f"d{num_dec_layer}.loss_depth"] = losses_depth[
                num_dec_layer
            ]
            loss_dict[f"d{num_dec_layer}.loss_dim"] = losses_dim[num_dec_layer]
            loss_dict[f"d{num_dec_layer}.loss_rot"] = losses_rot[num_dec_layer]

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            if enc_outputs_3d is None:
                # NOTE The enc_loss calculation of the DINO is
                # different from that of Deformable DETR.
                enc_loss_cls, enc_losses_bbox, enc_losses_iou = (
                    self.loss_by_feat_single(
                        enc_cls_scores,
                        enc_bbox_preds,
                        text_token_mask=text_token_mask,
                        input_hw=input_hw,
                        batch_gt_boxes=batch_gt_boxes,
                        batch_gt_boxes_classes=batch_gt_boxes_classes,
                        positive_maps=positive_maps,
                    )
                )
                loss_dict["enc_loss_cls"] = enc_loss_cls
                loss_dict["enc_loss_bbox"] = enc_losses_bbox
                loss_dict["enc_loss_iou"] = enc_losses_iou
            else:
                (
                    enc_loss_cls,
                    enc_losses_bbox,
                    enc_losses_iou,
                    enc_losses_cen,
                    enc_losses_depth,
                    enc_losses_dim,
                    enc_losses_rot,
                ) = self.loss_3d_by_feat_single(
                    enc_cls_scores,
                    enc_bbox_preds,
                    enc_outputs_3d,
                    text_token_mask=text_token_mask,
                    input_hw=input_hw,
                    batch_gt_boxes=batch_gt_boxes,
                    batch_gt_boxes_3d=batch_gt_boxes_3d,
                    batch_gt_boxes_classes=batch_gt_boxes_classes,
                    batch_gt_intrinsics=batch_gt_intrinsics,
                    positive_maps=positive_maps,
                )
                loss_dict["enc_loss_cls"] = enc_loss_cls
                loss_dict["enc_loss_bbox"] = enc_losses_bbox
                loss_dict["enc_loss_iou"] = enc_losses_iou
                loss_dict["enc_loss_delta_2d"] = enc_losses_cen
                loss_dict["enc_loss_depth"] = enc_losses_depth
                loss_dict["enc_loss_dim"] = enc_losses_dim
                loss_dict["enc_loss_rot"] = enc_losses_rot

        if all_layers_denoising_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds,
                boxes2d=batch_gt_boxes,
                boxes2d_classes=batch_gt_boxes_classes,
                positive_maps=positive_maps,
                input_hw=input_hw,
                text_token_mask=text_token_mask,
                dn_meta=dn_meta,
            )

            # collate denoising loss
            loss_dict["dn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["dn_loss_bbox"] = dn_losses_bbox[-1]
            loss_dict["dn_loss_iou"] = dn_losses_iou[-1]

            for num_dec_layer, (
                loss_cls_i,
                loss_bbox_i,
                loss_iou_i,
            ) in enumerate(
                zip(
                    dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1]
                )
            ):
                loss_dict[f"d{num_dec_layer}.dn_loss_cls"] = loss_cls_i
                loss_dict[f"d{num_dec_layer}.dn_loss_bbox"] = loss_bbox_i
                loss_dict[f"d{num_dec_layer}.dn_loss_iou"] = loss_iou_i

        return loss_dict
