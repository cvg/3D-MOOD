"""G-DINO Loss."""

import torch
from torch import Tensor, nn
from vis4d.common.distributed import reduce_mean
from vis4d.op.loss.common import l1_loss
from vis4d.op.loss.reducer import SumWeightedLoss

from opendet3d.op.box.box2d import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from opendet3d.op.box.matchers.hungarian import HungarianMatcher
from opendet3d.op.loss.focal_loss import FocalLoss
from opendet3d.op.loss.iou_loss import GIoULoss
from opendet3d.op.match_cost import (
    BBoxL1Cost,
    BinaryFocalLossCost,
    IoUCost,
)
from opendet3d.op.util import multi_apply


class GroundingDINOLoss(nn.Module):
    """Grounding DINO loss module."""

    def __init__(
        self, max_text_len: int = 256, sync_cls_avg_factor: bool = True
    ):
        super().__init__()
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.max_text_len = max_text_len

        # Matcher
        self.cls_cost = BinaryFocalLossCost(weight=2.0)
        self.reg_cost = BBoxL1Cost(weight=5.0, box_format="xywh")
        self.iou_cost = IoUCost(weight=2.0, iou_mode="giou")

        self.assigner = HungarianMatcher()

        # Losses
        self.loss_cls = FocalLoss(alpha=0.25, gamma=2.0)
        self.bg_cls_weight = 0.0
        self.cls_loss_weight = 1.0

        self.loss_bbox = l1_loss
        self.bbox_loss_weight = 5.0

        self.loss_iou = GIoULoss()
        self.iou_loss_weight = 2.0

    def get_targets(
        self,
        cls_scores_list: list[Tensor],
        bbox_preds_list: list[Tensor],
        input_hw: list[tuple[int, int]],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        positive_maps: list[Tensor],
        text_token_mask: Tensor,
    ) -> tuple:
        """Compute regression and classification targets for a batch image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image, has shape [num_queries,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_queries, 4].
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_targets_single,
            cls_scores_list,
            bbox_preds_list,
            input_hw,
            batch_gt_boxes,
            batch_gt_boxes_classes,
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
            num_total_pos,
            num_total_neg,
        )

    def _get_cost(
        self,
        cls_score,
        bbox_pred,
        gt_boxes,
        input_hw,
        text_token_mask,
        positive_map,
    ):
        """Compute regression and classification cost for one image."""
        if self.cls_cost.weight != 0:
            cls_cost = self.cls_cost(cls_score, text_token_mask, positive_map)
        else:
            cls_cost = 0

        if self.reg_cost.weight != 0:
            reg_cost = self.reg_cost(
                bbox_pred, gt_boxes, input_hw[0], input_hw[1]
            )
        else:
            reg_cost = 0

        if self.iou_cost.weight != 0:
            iou_cost = self.iou_cost(bbox_pred, gt_boxes)
        else:
            iou_cost = 0

        return cls_cost + reg_cost + iou_cost

    def _get_targets_2d_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        input_hw: tuple[int, int],
        gt_boxes: Tensor,
        gt_classes: Tensor,
        positive_map: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for one image."""
        img_h, img_w = input_hw
        num_bboxes = bbox_pred.size(0)
        factor = bbox_pred.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(
            0
        )

        # convert bbox_pred from xywh, normalized to xyxy, unnormalized
        bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_pred = bbox_pred * factor

        # assigner and sampler
        cost = self._get_cost(
            cls_score,
            bbox_pred,
            gt_boxes,
            input_hw,
            text_token_mask,
            positive_map,
        )

        assign_result = self.assigner(cost, bbox_pred, gt_boxes, gt_classes)

        pos_inds = (
            torch.nonzero(
                assign_result.assigned_gt_indices > 0, as_tuple=False
            )
            .squeeze(-1)
            .unique()
        )
        neg_inds = (
            torch.nonzero(
                assign_result.assigned_gt_indices == 0, as_tuple=False
            )
            .squeeze(-1)
            .unique()
        )
        pos_assigned_gt_inds = assign_result.assigned_gt_indices[pos_inds] - 1
        pos_gt_bboxes = gt_boxes[pos_assigned_gt_inds.long(), :]

        # Major changes. The labels are 0-1 binary labels for each bbox
        # and text tokens.
        labels = gt_boxes.new_full(
            (num_bboxes, self.max_text_len), 0, dtype=torch.float32
        )
        labels[pos_inds] = positive_map[pos_assigned_gt_inds]
        label_weights = gt_boxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred, dtype=gt_boxes.dtype)
        bbox_weights = torch.zeros_like(bbox_pred, dtype=gt_boxes.dtype)
        bbox_weights[pos_inds] = 1.0

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        pos_gt_bboxes_normalized = pos_gt_bboxes / factor
        pos_gt_bboxes_targets = bbox_xyxy_to_cxcywh(pos_gt_bboxes_normalized)
        bbox_targets[pos_inds] = pos_gt_bboxes_targets

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_gt_bboxes,
            pos_inds,
            neg_inds,
            pos_assigned_gt_inds,
        )

    def _get_targets_single(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        input_hw: tuple[int, int],
        gt_boxes: Tensor,
        gt_classes: Tensor,
        positive_map: Tensor,
        text_token_mask: Tensor,
    ) -> tuple:
        """Compute regression and classification targets for one image.

        Outputs from a single decoder layer of a single feature level are used.

        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_queries, 4].
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            _,
            pos_inds,
            neg_inds,
            _,
        ) = self._get_targets_2d_single(
            cls_score,
            bbox_pred,
            input_hw,
            gt_boxes,
            gt_classes,
            positive_map,
            text_token_mask,
        )

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        text_token_mask: Tensor,
        input_hw: list[tuple[int, int]],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
        positive_maps: list[Tensor],
    ) -> tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)

        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]

        with torch.no_grad():
            (
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                num_total_pos,
                num_total_neg,
            ) = self.get_targets(
                cls_scores_list,
                bbox_preds_list,
                input_hw,
                batch_gt_boxes,
                batch_gt_boxes_classes,
                positive_maps,
                text_token_mask,
            )

        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

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

        return loss_cls, loss_bbox, loss_iou

    def forward(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        text_token_mask: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        dn_meta: dict[str, int],
        positive_maps: list[Tensor],
        input_hw: list[tuple[int, int]],
        batch_gt_boxes: list[Tensor],
        batch_gt_boxes_classes: list[Tensor],
    ) -> dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
        ) = split_outputs(
            all_layers_cls_scores, all_layers_bbox_preds, dn_meta
        )

        # DETRHead loss_by_feat
        losses_cls, losses_bbox, losses_iou = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            text_token_mask=text_token_mask,
            input_hw=input_hw,
            batch_gt_boxes=batch_gt_boxes,
            batch_gt_boxes_classes=batch_gt_boxes_classes,
            positive_maps=positive_maps,
        )

        loss_dict = dict()

        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_iou"] = losses_iou[-1]

        # loss from other decoder layers
        for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in enumerate(
            zip(losses_cls[:-1], losses_bbox[:-1], losses_iou[:-1])
        ):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_iou"] = loss_iou_i

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
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

    def _get_dn_targets_single(
        self,
        gt_bboxes: Tensor,
        gt_labels: Tensor,
        positive_maps: Tensor,
        img_shape: tuple[int, int],
        num_groups: int,
        num_denoising_queries: int,
    ) -> tuple:
        """Get targets in denoising part for one image.

        Args:
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It should includes ``bboxes`` and ``labels``
                attributes.
            img_meta (dict): Meta information for one image.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

            - labels (Tensor): Labels of each image.
            - label_weights (Tensor]): Label weights of each image.
            - bbox_targets (Tensor): BBox targets of each image.
            - bbox_weights (Tensor): BBox weights of each image.
            - pos_inds (Tensor): Sampled positive indices for each image.
            - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_queries_each_group = int(num_denoising_queries / num_groups)
        device = gt_bboxes.device

        if len(gt_labels) > 0:
            t = torch.arange(len(gt_labels), dtype=torch.long, device=device)
            t = t.unsqueeze(0).repeat(num_groups, 1)
            pos_assigned_gt_inds = t.flatten()
            pos_inds = torch.arange(
                num_groups, dtype=torch.long, device=device
            )
            pos_inds = pos_inds.unsqueeze(1) * num_queries_each_group + t
            pos_inds = pos_inds.flatten()
        else:
            pos_inds = pos_assigned_gt_inds = gt_bboxes.new_tensor(
                [], dtype=torch.long
            )

        neg_inds = pos_inds + num_queries_each_group // 2
        # label targets
        # this change
        labels = gt_bboxes.new_full(
            (num_denoising_queries, self.max_text_len), 0, dtype=torch.float32
        )
        labels[pos_inds] = positive_maps[pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_denoising_queries)

        # bbox targets
        bbox_targets = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights = torch.zeros(num_denoising_queries, 4, device=device)
        bbox_weights[pos_inds] = 1.0

        img_h, img_w = img_shape

        # DETR regress the relative position of boxes (cxcywh) in the image.
        # Thus the learning target should be normalized by the image size, also
        # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(
            0
        )
        gt_bboxes_normalized = gt_bboxes / factor
        gt_bboxes_targets = bbox_xyxy_to_cxcywh(gt_bboxes_normalized)
        bbox_targets[pos_inds] = gt_bboxes_targets.repeat([num_groups, 1])

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def get_dn_targets(
        self,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        positive_maps: list[Tensor],
        input_hw: list[tuple[int, int]],
        dn_meta: dict[str, int],
    ) -> tuple:
        """Get targets in denoising part for a batch of images.

        Args:
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            tuple: a tuple containing the following targets.

            - labels_list (list[Tensor]): Labels for all images.
            - label_weights_list (list[Tensor]): Label weights for all images.
            - bbox_targets_list (list[Tensor]): BBox targets for all images.
            - bbox_weights_list (list[Tensor]): BBox weights for all images.
            - num_total_pos (int): Number of positive samples in all images.
            - num_total_neg (int): Number of negative samples in all images.
        """
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_dn_targets_single,
            boxes2d,
            boxes2d_classes,
            positive_maps,
            input_hw,
            num_groups=dn_meta["num_denoising_groups"],
            num_denoising_queries=dn_meta["num_denoising_queries"],
        )

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _loss_dn_single(
        self,
        dn_cls_scores: Tensor,
        dn_bbox_preds: Tensor,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        positive_maps: list[Tensor],
        input_hw: list[tuple[int, int]],
        text_token_mask: Tensor,
        dn_meta,
    ):
        """Denoising loss for outputs from a single decoder layer.

        Args:
            dn_cls_scores (Tensor): Classification scores of a single decoder
                layer in denoising part, has shape (bs, num_denoising_queries,
                cls_out_channels).
            dn_bbox_preds (Tensor): Regression outputs of a single decoder
                layer in denoising part. Each is a 4D-tensor with normalized
                coordinate format (cx, cy, w, h) and has shape
                (bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = self.get_dn_targets(
            boxes2d, boxes2d_classes, positive_maps, input_hw, dn_meta
        )

        labels = torch.stack(labels_list, 0)
        label_weights = torch.stack(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # Loss is not computed for the padded regions of the text.
        assert text_token_mask.dim() == 2
        text_masks = text_token_mask.new_zeros(
            (text_token_mask.size(0), self.max_text_len)
        )
        text_masks[:, : text_token_mask.size(1)] = text_token_mask
        text_mask = (text_masks > 0).unsqueeze(1)
        text_mask = text_mask.repeat(1, dn_cls_scores.size(1), 1)
        cls_scores = torch.masked_select(dn_cls_scores, text_mask).contiguous()
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

        if len(cls_scores) > 0:
            loss_cls = self.cls_loss_weight * self.loss_cls(
                cls_scores,
                labels,
                reducer=SumWeightedLoss(
                    weight=label_weights, avg_factor=cls_avg_factor
                ),
            )
        else:
            loss_cls = torch.zeros(
                1, dtype=cls_scores.dtype, device=cls_scores.device
            )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_hw, bbox_pred in zip(input_hw, dn_bbox_preds):
            img_h, img_w = img_hw
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = dn_bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        if bbox_targets.shape[0] == 0:
            loss_bbox = bbox_preds.sum()
            loss_iou = bbox_preds.sum()
            return loss_cls, loss_bbox, loss_iou

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

        return loss_cls, loss_bbox, loss_iou

    def loss_dn(
        self,
        all_layers_denoising_cls_scores: Tensor,
        all_layers_denoising_bbox_preds: Tensor,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        positive_maps: list[Tensor],
        input_hw: list[tuple[int, int]],
        text_token_mask: Tensor,
        dn_meta: dict[str, int],
    ):
        """Calculate denoising loss.

        Args:
            all_layers_denoising_cls_scores (Tensor): Classification scores of
                all decoder layers in denoising part, has shape (
                num_decoder_layers, bs, num_denoising_queries,
                cls_out_channels).
            all_layers_denoising_bbox_preds (Tensor): Regression outputs of all
                decoder layers in denoising part. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h) and has shape
                (num_decoder_layers, bs, num_denoising_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            Tuple[List[Tensor]]: The loss_dn_cls, loss_dn_bbox, and loss_dn_iou
            of each decoder layers.
        """
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_bbox_preds,
            boxes2d=boxes2d,
            boxes2d_classes=boxes2d_classes,
            positive_maps=positive_maps,
            input_hw=input_hw,
            text_token_mask=text_token_mask,
            dn_meta=dn_meta,
        )


# TODO: Move to DINO ops
def split_outputs(
    all_layers_cls_scores: Tensor,
    all_layers_bbox_preds: Tensor,
    dn_meta: dict[str, int] | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split outputs of the denoising part and the matching part.

    For the total outputs of `num_queries_total` length, the former
    `num_denoising_queries` outputs are from denoising queries, and
    the rest `num_matching_queries` ones are from matching queries,
    where `num_queries_total` is the sum of `num_denoising_queries` and
    `num_matching_queries`.

    Args:
        all_layers_cls_scores (Tensor): Classification scores of all
            decoder layers, has shape (num_decoder_layers, bs,
            num_queries_total, cls_out_channels).
        all_layers_bbox_preds (Tensor): Regression outputs of all decoder
            layers. Each is a 4D-tensor with normalized coordinate format
            (cx, cy, w, h) and has shape (num_decoder_layers, bs,
            num_queries_total, 4).
        dn_meta (Dict[str, int]): The dictionary saves information about
            group collation, including 'num_denoising_queries' and
            'num_denoising_groups'.

    Returns:
        Tuple[Tensor]: a tuple containing the following outputs.

        - all_layers_matching_cls_scores (Tensor): Classification scores
            of all decoder layers in matching part, has shape
            (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
        - all_layers_matching_bbox_preds (Tensor): Regression outputs of
            all decoder layers in matching part. Each is a 4D-tensor with
            normalized coordinate format (cx, cy, w, h) and has shape
            (num_decoder_layers, bs, num_matching_queries, 4).
        - all_layers_denoising_cls_scores (Tensor): Classification scores
            of all decoder layers in denoising part, has shape
            (num_decoder_layers, bs, num_denoising_queries,
            cls_out_channels).
        - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
            all decoder layers in denoising part. Each is a 4D-tensor with
            normalized coordinate format (cx, cy, w, h) and has shape
            (num_decoder_layers, bs, num_denoising_queries, 4).
    """
    # FIXME: Can dn_meta be None?
    num_denoising_queries = dn_meta["num_denoising_queries"]

    if dn_meta is not None:
        all_layers_denoising_cls_scores = all_layers_cls_scores[
            :, :, :num_denoising_queries, :
        ]
        all_layers_denoising_bbox_preds = all_layers_bbox_preds[
            :, :, :num_denoising_queries, :
        ]
        all_layers_matching_cls_scores = all_layers_cls_scores[
            :, :, num_denoising_queries:, :
        ]
        all_layers_matching_bbox_preds = all_layers_bbox_preds[
            :, :, num_denoising_queries:, :
        ]
    else:
        all_layers_denoising_cls_scores = None
        all_layers_denoising_bbox_preds = None
        all_layers_matching_cls_scores = all_layers_cls_scores
        all_layers_matching_bbox_preds = all_layers_bbox_preds

    return (
        all_layers_matching_cls_scores,
        all_layers_matching_bbox_preds,
        all_layers_denoising_cls_scores,
        all_layers_denoising_bbox_preds,
    )
