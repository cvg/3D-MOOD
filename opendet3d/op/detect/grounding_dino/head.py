"""Groudning DINO bbox head."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torchvision.ops import batched_nms, nms
from vis4d.op.layer.transformer import get_clones, inverse_sigmoid
from vis4d.op.layer.weight_init import constant_init

from opendet3d.op.box.box2d import bbox_cxcywh_to_xyxy


class GroundingDINOHead(nn.Module):
    """Head of the Grounding DINO."""

    def __init__(
        self,
        num_classes: int = 256,
        embed_dims: int = 256,
        num_decoder_layer: int = 6,
        fc_cls: ContrastiveEmbed | None = None,
        num_reg_fcs: int = 2,
        as_two_stage: bool = True,
        with_box_refine: bool = True,
    ) -> None:
        """Create an instance of GroundingDINOHead."""
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims

        self.num_pred_layer = (
            num_decoder_layer + 1 if as_two_stage else num_decoder_layer
        )
        self.as_two_stage = as_two_stage

        fc_cls = fc_cls or ContrastiveEmbed(
            max_text_len=256, log_scale="auto", bias=True
        )

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, 4))
        reg_branch = nn.Sequential(*reg_branch)

        # NOTE: due to the fc_cls is a contrastive embedding and don't
        # have any trainable parameters, we do not need to copy it.
        if not with_box_refine:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(self.num_pred_layer)]
            )
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(self.num_pred_layer)]
            )
        else:
            self.cls_branches = get_clones(fc_cls, self.num_pred_layer)
            self.reg_branches = get_clones(reg_branch, self.num_pred_layer)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights of the Deformable DETR head."""
        for m in self.reg_branches:
            constant_init(m[-1], 0, bias=0)
        nn.init.constant_(self.reg_branches[0][-1].bias.data[2:], -2.0)
        if self.as_two_stage:
            for m in self.reg_branches:
                nn.init.constant_(m[-1].bias.data[2:], 0.0)

    def forward(
        self,
        hidden_states: Tensor,
        references: list[Tensor],
        memory_text: Tensor,
        text_token_mask: Tensor,
    ) -> tuple[Tensor]:
        """Forward function.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries, dim).
            references (List[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries, 4) when `as_two_stage` of the detector is `True`,
                otherwise (bs, num_queries, 2). Each `inter_reference` has
                shape (bs, num_queries, 4) when `with_box_refine` of the
                detector is `True`, otherwise (bs, num_queries, 2). The
                coordinates are arranged as (cx, cy) when the last dimension is
                2, and (cx, cy, w, h) when it is 4.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_token_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            tuple[Tensor]: results of head containing the following tensor.

            - all_layers_outputs_classes (Tensor): Outputs from the
              classification head, has shape (num_decoder_layers, bs,
              num_queries, cls_out_channels).
            - all_layers_outputs_coords (Tensor): Sigmoid outputs from the
              regression head with normalized coordinate format (cx, cy, w,
              h), has shape (num_decoder_layers, bs, num_queries, 4) with the
              last dimension arranged as (cx, cy, w, h).
        """
        all_layers_outputs_classes = []
        all_layers_outputs_coords = []

        for layer_id in range(hidden_states.shape[0]):
            reference = inverse_sigmoid(references[layer_id])
            # NOTE The last reference will not be used.
            hidden_state = hidden_states[layer_id]
            outputs_class = self.cls_branches[layer_id](
                hidden_state, memory_text, text_token_mask
            )
            tmp_reg_preds = self.reg_branches[layer_id](hidden_state)
            if reference.shape[-1] == 4:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `True`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `True`.
                tmp_reg_preds += reference
            else:
                # When `layer` is 0 and `as_two_stage` of the detector
                # is `False`, or when `layer` is greater than 0 and
                # `with_box_refine` of the detector is `False`.
                assert reference.shape[-1] == 2
                tmp_reg_preds[..., :2] += reference
            outputs_coord = tmp_reg_preds.sigmoid()
            all_layers_outputs_classes.append(outputs_class)
            all_layers_outputs_coords.append(outputs_coord)

        all_layers_outputs_classes = torch.stack(all_layers_outputs_classes)
        all_layers_outputs_coords = torch.stack(all_layers_outputs_coords)

        return all_layers_outputs_classes, all_layers_outputs_coords


class RoI2Det:
    """Convert RoI to Detection."""

    def __init__(
        self,
        nms: bool = False,
        max_per_img: int = 300,
        class_agnostic_nms: bool = False,
        score_threshold: float = 0.0,
        iou_threshold: float = 0.5,
    ) -> None:
        """Create an instance of RoI2Det."""
        self.nms = nms
        self.max_per_img = max_per_img
        self.class_agnostic_nms = class_agnostic_nms
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def __call__(
        self,
        cls_score: Tensor,
        bbox_pred: Tensor,
        token_positive_maps: dict[int, list[int]] | None,
        img_shape: tuple[int, int],
        ori_shape: tuple[int, int],
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_queries, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h) and
                shape [num_queries, 4].
            token_positive_maps (dict): Token positive map.
            img_meta (dict): Image meta info.
            rescale (bool, optional): If True, return boxes in original image
                space. Default True.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_score) == len(bbox_pred)  # num_queries

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])

        if token_positive_maps is not None:
            cls_score = convert_grounding_to_cls_scores(
                logits=cls_score.sigmoid()[None],
                positive_maps=[token_positive_maps],
            )[0]

            scores, indexes = cls_score.view(-1).topk(self.max_per_img)
            num_classes = cls_score.shape[-1]
            det_labels = indexes % num_classes
            bbox_index = indexes // num_classes
            det_bboxes = det_bboxes[bbox_index]
        else:
            cls_score = cls_score.sigmoid()
            scores, _ = cls_score.max(-1)
            scores, indexes = scores.topk(self.max_per_img)
            det_bboxes = det_bboxes[indexes]
            det_labels = scores.new_zeros(scores.shape, dtype=torch.long)

        # Remove low scoring boxes
        if self.score_threshold > 0.0:
            mask = scores > self.score_threshold
            det_bboxes = det_bboxes[mask]
            det_labels = det_labels[mask]
            scores = scores[mask]

        if self.nms:
            if self.class_agnostic_nms:
                keep = nms(det_bboxes, scores, self.iou_threshold)
            else:
                keep = batched_nms(
                    det_bboxes, scores, det_labels, self.iou_threshold
                )

            det_bboxes = det_bboxes[keep]
            det_labels = det_labels[keep]
            scores = scores[keep]

        # rescale to original shape
        det_bboxes /= det_bboxes.new_tensor(
            [img_shape[1] / ori_shape[1], img_shape[0] / ori_shape[0]]
        ).repeat((1, 2))

        return det_bboxes, scores, det_labels


class ContrastiveEmbed(nn.Module):
    """Text visual ContrastiveEmbed layer."""

    def __init__(
        self,
        max_text_len: int = 256,
        log_scale: str | float | None = None,
        bias: bool = False,
    ) -> None:
        """Create an instance of `ContrastiveEmbed`.

        Args:
            max_text_len (int, optional): Maximum length of text.
            log_scale (Optional[Union[str, float]]):  The initial value of a
                learnable parameter to multiply with the similarity matrix to
                normalize the output.  Defaults to 0.0.
                If set to 'auto', the similarity matrix will be normalized by
                a fixed value ``sqrt(d_c)`` where ``d_c`` is the channel
                number. If set to ``None``, there is no normalization applied.
                If set to a float number, the similarity matrix will be
                    multiplied by ``exp(log_scale)``, where ``log_scale`` is
                    learnable.
            bias (bool, optional): Whether to add bias to the output. If set to
                ``True``, a learnable bias that is initialized as -4.6 will be
                added to the output. Useful when training from scratch.
                Defaults to False.
        """
        super().__init__()
        self.max_text_len = max_text_len
        self.log_scale = log_scale

        if isinstance(log_scale, float):
            self.log_scale = nn.Parameter(
                torch.Tensor([float(log_scale)]), requires_grad=True
            )
        elif log_scale != "auto" and log_scale is not None:
            raise ValueError(
                "log_scale should be float, 'auto' or None, "
                f"but got {log_scale}"
            )

        if bias:
            bias_value = -math.log((1 - 0.01) / 0.01)
            self.bias = nn.Parameter(
                torch.Tensor([bias_value]), requires_grad=True
            )
        else:
            self.bias = None

    def forward(
        self, visual_feat: Tensor, text_feat: Tensor, text_token_mask: Tensor
    ) -> Tensor:
        """Forward function.

        Args:
            visual_feat (Tensor): Visual features.
            text_feat (Tensor): Text features.
            text_token_mask (Tensor): A mask used for text feats.

        Returns:
            Tensor: Classification score.
        """
        res = visual_feat @ text_feat.transpose(-1, -2)

        if isinstance(self.log_scale, nn.Parameter):
            res = res * self.log_scale.exp()
        elif self.log_scale == "auto":
            # NOTE: similar to the normalizer in self-attention
            res = res / math.sqrt(visual_feat.shape[-1])

        if self.bias is not None:
            res = res + self.bias
        res.masked_fill_(~text_token_mask[:, None, :], float("-inf"))

        new_res = torch.full(
            (*res.shape[:-1], self.max_text_len),
            float("-inf"),
            device=res.device,
        )
        new_res[..., : res.shape[-1]] = res

        return new_res


def convert_grounding_to_cls_scores(
    logits: Tensor, positive_maps: dict[int, list[int, int]]
) -> Tensor:
    """Convert logits to class scores."""
    assert len(positive_maps) == logits.shape[0]  # batch size

    scores = torch.zeros(
        logits.shape[0], logits.shape[1], len(positive_maps[0])
    ).to(logits.device)
    if positive_maps is not None:
        if all(x == positive_maps[0] for x in positive_maps):
            # only need to compute once
            positive_map = positive_maps[0]
            for label_j in positive_map:
                scores[:, :, label_j - 1] = logits[
                    :, :, torch.LongTensor(positive_map[label_j])
                ].mean(-1)
        else:
            for i, positive_map in enumerate(positive_maps):
                for label_j in positive_map:
                    scores[i, :, label_j - 1] = logits[
                        i, :, torch.LongTensor(positive_map[label_j])
                    ].mean(-1)
    return scores
