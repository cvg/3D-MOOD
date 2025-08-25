"""3D-MOOD."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from vis4d.op.base import BaseModel
from vis4d.op.fpp.fpn import FPN

from opendet3d.model.detect.grounding_dino import GroundingDINO
from opendet3d.model.language.mm_bert import BertModel
from opendet3d.op.detect3d.grounding_dino_3d import (
    GroundingDINO3DHead,
    RoI2Det3D,
)
from opendet3d.op.detect.grounding_dino import GroundingDINOHead, RoI2Det
from opendet3d.op.fpp.channel_mapper import ChannelMapper


class Det3DOut(NamedTuple):
    """Output of the detection model.

    boxes (list[Tensor]): 2D bounding boxes of shape [N, 4] in xyxy format.
    boxes3d (list[Tensor]): 3D bounding boxes of shape [N, 10].
    scores (list[Tensor]): confidence scores of shape [N,].
    class_ids (list[Tensor]): class ids of shape [N,].
    """

    boxes: list[Tensor]
    boxes3d: list[Tensor]
    scores: list[Tensor]
    class_ids: list[Tensor]
    depth_maps: list[Tensor] | None
    categories: list[list[str]] | None = None


class GroundingDINO3DOut(NamedTuple):
    """Output of the Grounding DINO model."""

    all_layers_cls_scores: list[Tensor]
    all_layers_bbox_preds: list[Tensor]
    all_layers_bbox_3d_preds: list[Tensor]
    enc_outputs_class: Tensor
    enc_outputs_coord: Tensor
    enc_outputs_3d: Tensor
    text_token_mask: Tensor
    dn_meta: dict[str, Tensor]
    positive_maps: list[Tensor]
    depth_maps: Tensor | None


class GroundingDINO3D(GroundingDINO):
    """Grounding DINO."""

    def __init__(
        self,
        basemodel: BaseModel,
        neck: ChannelMapper,
        texts: list[str] | None = None,
        custom_entities: bool = True,
        chunked_size: int = -1,
        num_queries: int = 900,
        num_feature_levels: int = 4,
        use_checkpoint: bool = False,
        bbox_head: GroundingDINOHead | None = None,
        language_model: BertModel | None = None,
        roi2det: RoI2Det | None = None,
        bbox3d_head: GroundingDINO3DHead | None = None,
        roi2det3d: RoI2Det3D | None = None,
        depth_head: nn.Module | None = None,
        fpn: FPN | None = None,
        freeze_detector: bool = False,
        weights: str | None = None,
        cat_mapping: dict[str, int] | None = None,
    ) -> None:
        """Create the Grounding DINO model."""
        super().__init__(
            basemodel=basemodel,
            neck=neck,
            texts=texts,
            custom_entities=custom_entities,
            chunked_size=chunked_size,
            num_queries=num_queries,
            num_feature_levels=num_feature_levels,
            use_checkpoint=use_checkpoint,
            bbox_head=bbox_head,
            roi2det=roi2det,
            language_model=language_model,
            weights=weights,
        )

        self.bbox3d_head = bbox3d_head or GroundingDINO3DHead()
        self.roi2det3d = roi2det3d or RoI2Det3D()

        # Depth Head
        self.fpn = fpn
        self.depth_head = depth_head

        if freeze_detector:
            self._freeze_detector()

        self.cat_mapping = cat_mapping

    def _freeze_detector(self):
        """Freeze the detector."""
        for model in [
            self.backbone,
            self.neck,
            self.encoder,
            self.positional_encoding,
            self.memory_trans_fc,
            self.memory_trans_norm,
            self.decoder,
            self.bbox_head,
            self.dn_query_generator,
            self.language_model,
            self.text_feat_map,
        ]:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        self.level_embed.requires_grad = False
        self.query_embedding.requires_grad = False

    def forward_transformer(
        self,
        feat_flatten: Tensor,
        lvl_pos_embed_flatten: Tensor,
        memory_mask: Tensor | None,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        text_dict: dict[str, Tensor],
        boxes: Tensor | None = None,
        class_ids: Tensor | None = None,
        input_hw: list[tuple[int, int]] | None = None,
        ray_embeddings: Tensor | None = None,
        depth_latents: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, list[Tensor]]:
        """Forward function for the transformer."""
        text_token_mask = text_dict["text_token_mask"]

        memory, memory_text = self.encoder(
            query=feat_flatten,
            query_pos=lvl_pos_embed_flatten,
            key_padding_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            memory_text=text_dict["embedded"],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict["position_ids"],
            text_self_attention_masks=text_dict["masks"],
        )

        bs = memory.shape[0]

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ](output_memory, memory_text, text_token_mask)
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].max_text_len
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1
        )[1]

        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4),
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        # Top-k 3D proposals
        topk_output_memory = torch.gather(
            output_memory, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 256)
        )

        topk_output_3d = self.bbox3d_head.single_forward(
            self.decoder.num_layers,
            topk_output_memory,
            ray_embeddings,
            depth_latents,
        )

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)

        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = (
                self.dn_query_generator(boxes, class_ids, input_hw)
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat(
                [dn_bbox_query, topk_coords_unact], dim=1
            )
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None

        reference_points = reference_points.sigmoid()

        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if self.training:
            enc_outputs_class = topk_score
            enc_outputs_coord = topk_coords

        hidden_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            hidden_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        if self.training:
            return (
                memory_text,
                text_token_mask,
                hidden_states,
                list(references),
                enc_outputs_class,
                enc_outputs_coord,
                topk_output_3d,
                dn_meta,
            )

        return (
            memory_text,
            text_token_mask,
            hidden_states,
            list(references),
        )

    def _extract_image_features(
        self, images: Tensor
    ) -> tuple[list[Tensor], list[Tensor] | None]:
        """Extract image features."""
        visual_feats = self.backbone(images)[2:]

        if self.fpn is not None:
            depth_feats = self.fpn(visual_feats)

        if len(visual_feats) > len(self.neck.in_channels):
            start_index = len(visual_feats) - len(self.neck.in_channels)
            # NOTE: This is to make sure the number of input channels is the
            # same as the number of input channels of the neck.
            visual_feats = visual_feats[start_index:]

        visual_feats = self.neck(visual_feats)

        if self.fpn is None:
            depth_feats = visual_feats

        return visual_feats, depth_feats

    def _forward_train(
        self,
        images: Tensor,
        input_texts: list[list[str]] | None,
        boxes2d: list[Tensor],
        boxes2d_classes: list[Tensor],
        input_hw: list[tuple[int, int]],
        intrinsics: Tensor,
        input_tokens_positive: list[list[int, int]] | list[None] | None = None,
        padding: list[list[int]] | None = None,
    ) -> GroundingDINO3DOut:
        """Forward function for training."""
        batch_size = images.shape[0]

        new_text_prompts = []
        positive_maps = []
        if input_tokens_positive is not None:
            for tokens_positive_dict, text_prompt, gt_label in zip(
                input_tokens_positive, input_texts, boxes2d_classes
            ):
                if tokens_positive_dict is not None:
                    tokenized = self.language_model.tokenizer(
                        [text_prompt], padding="longest", return_tensors="pt"
                    )

                    new_text_prompts.append(text_prompt)

                    new_tokens_positive = [
                        tokens_positive_dict[label.item()]
                        for label in gt_label
                    ]
                else:
                    tokenized, caption_string, tokens_positive, _ = (
                        self.get_tokens_and_prompts(text_prompt)
                    )

                    new_text_prompts.append(caption_string)

                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]

                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive
                )
                positive_maps.append(positive_map)
        else:
            # All the text prompts are the same or using the self.texts,
            # so there is no need to calculate them multiple times.
            if (
                input_texts is None
                or len(set(["".join(t) for t in input_texts])) == 1
            ):
                if input_texts is None:
                    assert self.texts is not None, "Texts should be provided."
                    text_prompt = self.texts
                else:
                    text_prompt = input_texts[0]

                tokenized, caption_string, tokens_positive, _ = (
                    self.get_tokens_and_prompts(text_prompt)
                )
                new_text_prompts = [caption_string] * batch_size
                for gt_label in boxes2d_classes:
                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )
                    positive_maps.append(positive_map)
            else:
                for text_prompt, gt_label in zip(input_texts, boxes2d_classes):
                    tokenized, caption_string, tokens_positive, _ = (
                        self.get_tokens_and_prompts(text_prompt)
                    )

                    new_text_prompts.append(caption_string)

                    new_tokens_positive = [
                        tokens_positive[label] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive
                    )

                    positive_maps.append(positive_map)

        for i in range(batch_size):
            positive_maps[i] = (
                positive_maps[i].to(images.device).bool().float()
            )

        text_dict = self.language_model(new_text_prompts)

        if self.text_feat_map is not None:
            text_dict["embedded"] = self.text_feat_map(text_dict["embedded"])

        text_token_masks = []
        for i in range(batch_size):
            text_token_masks.append(
                text_dict["text_token_mask"][i]
                .unsqueeze(0)
                .repeat(len(positive_map), 1)
            )

        visual_feats, depth_feats = self._extract_image_features(images)

        batch_input_img_h, batch_input_img_w = images.shape[-2:]
        batch_input_shape = (batch_input_img_h, batch_input_img_w)

        (
            feat_flatten,
            lvl_pos_embed_flatten,
            memory_mask,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        ) = self.pre_transformer(
            visual_feats, input_hw, batch_input_shape, padding=padding
        )

        ray_embeddings = self.bbox3d_head.get_camera_embeddings(
            intrinsics, batch_input_shape
        )

        # Depth Head
        depth_preds, depth_latents = self.depth_head(
            depth_feats, intrinsics, batch_input_shape
        )

        (
            memory_text,
            text_token_mask,
            hidden_states,
            references,
            enc_outputs_class,
            enc_outputs_coord,
            enc_outputs_3d,
            dn_meta,
        ) = self.forward_transformer(
            feat_flatten,
            lvl_pos_embed_flatten,
            memory_mask,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            text_dict,
            boxes2d,
            boxes2d_classes,
            input_hw,
            ray_embeddings,
            depth_latents,
        )

        all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
            hidden_states, references, memory_text, text_token_mask
        )

        # Not using denoising for 3D
        hidden_states_3d = hidden_states[
            :, :, dn_meta["num_denoising_queries"] :, :
        ]

        all_layers_outputs_3d = self.bbox3d_head(
            hidden_states_3d, ray_embeddings, depth_latents
        )

        return GroundingDINO3DOut(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            all_layers_outputs_3d,
            enc_outputs_class,
            enc_outputs_coord,
            enc_outputs_3d,
            text_token_mask,
            dn_meta,
            positive_maps,
            depth_maps=depth_preds,
        )

    def _forward_test(
        self,
        images: Tensor,
        input_texts: list[str] | None,
        text_prompt_mapping: list[dict[str, dict[str, str]]] | None,
        input_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
        intrinsics: list[Tensor] | None,
        padding: list[list[int]] | None,
    ) -> Det3DOut:
        """Forward."""
        batch_size = images.shape[0]

        token_positive_maps = []
        text_prompts = []
        entities = []
        for i in range(batch_size):
            if self.texts is not None:
                text_prompt = self.texts
            else:
                text_prompt = input_texts[i]

            if text_prompt_mapping is not None:
                prompt_mapping = text_prompt_mapping[i]
            else:
                prompt_mapping = None

            token_positive_map, captions, _, _entities = (
                self.get_tokens_positive_and_prompts(
                    text_prompt,
                    self.custom_entities,
                    text_prompt_mapping=prompt_mapping,
                )
            )

            token_positive_maps.append(token_positive_map)
            text_prompts.append(captions)
            entities.append(_entities)

        # image feature extraction
        batch_input_img_h, batch_input_img_w = images.shape[-2:]

        visual_feats, depth_feats = self._extract_image_features(images)

        batch_input_shape = (batch_input_img_h, batch_input_img_w)

        if isinstance(text_prompts[0], list):
            assert batch_size == 1, "Batch size should be 1 for chunked text."
            assert (
                self.cat_mapping is not None
            ), "Category mapping should be provided."

            boxes = []
            boxes3d = []
            scores = []
            class_ids = []
            categories = []
            for i, captions in enumerate(text_prompts[0]):
                token_positive_map = token_positive_maps[0][i]
                cur_entities = entities[0][i]

                text_dict = self.language_model([captions])

                # text feature map layer
                if self.text_feat_map is not None:
                    text_dict["embedded"] = self.text_feat_map(
                        text_dict["embedded"]
                    )

                (
                    feat_flatten,
                    lvl_pos_embed_flatten,
                    memory_mask,
                    spatial_shapes,
                    level_start_index,
                    valid_ratios,
                ) = self.pre_transformer(
                    copy.deepcopy(visual_feats), input_hw, batch_input_shape
                )

                ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                    intrinsics, batch_input_shape
                )

                # Depth Head
                depth_preds, depth_latents = self.depth_head(
                    depth_feats, intrinsics, batch_input_shape
                )

                depth_maps = []
                for i, depth_pred in enumerate(depth_preds):
                    if padding is not None:
                        pad_left, pad_right, pad_top, pad_bottom = padding[i]

                        depth_pred = depth_pred[
                            pad_top : batch_input_img_h - pad_bottom,
                            pad_left : batch_input_img_w - pad_right,
                        ]

                    depth_maps.append(
                        F.interpolate(
                            depth_pred.unsqueeze(0).unsqueeze(0),
                            size=original_hw[i],
                            mode="bilinear",
                            align_corners=False,
                            antialias=True,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )

                (
                    memory_text,
                    text_token_mask,
                    hidden_states,
                    references,
                ) = self.forward_transformer(
                    feat_flatten,
                    lvl_pos_embed_flatten,
                    memory_mask,
                    spatial_shapes,
                    level_start_index,
                    valid_ratios,
                    text_dict,
                    ray_embeddings=ray_embeddings,
                    depth_latents=depth_latents,
                )

                all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
                    hidden_states, references, memory_text, text_token_mask
                )

                all_layers_outputs_3d = self.bbox3d_head(
                    hidden_states, ray_embeddings, depth_latents=depth_latents
                )

                cls_scores = all_layers_cls_scores[-1]
                bbox_preds = all_layers_bbox_preds[-1]
                boxes3d_preds = all_layers_outputs_3d[-1]

                cls_score = cls_scores[0]
                det_bboxes, det_scores, det_labels, det_bboxes3d = (
                    self.roi2det3d(
                        cls_score,
                        bbox_preds[0],
                        token_positive_map,
                        input_hw[0],
                        original_hw[0],
                        boxes3d_preds[0],
                        intrinsics[0],
                        padding[0] if padding is not None else None,
                    )
                )

                boxes.append(det_bboxes)
                scores.append(det_scores)
                boxes3d.append(det_bboxes3d)

                # Get the categories text and class ids
                cur_class_ids = []
                cur_categories = []
                for label in det_labels:
                    cur_class_ids.append(self.cat_mapping[cur_entities[label]])
                    cur_categories.append(cur_entities[label])

                class_ids.append(cur_class_ids)
                categories.append(cur_categories)

            boxes = [torch.cat([b for b in boxes])]
            boxes3d = [torch.cat([b for b in boxes3d])]
            scores = [torch.cat([s for s in scores])]
            class_ids = [sum(class_ids, [])]
            categories = [sum(categories, [])]
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))

            # text feature map layer
            if self.text_feat_map is not None:
                text_dict["embedded"] = self.text_feat_map(
                    text_dict["embedded"]
                )

            (
                feat_flatten,
                lvl_pos_embed_flatten,
                memory_mask,
                spatial_shapes,
                level_start_index,
                valid_ratios,
            ) = self.pre_transformer(visual_feats, input_hw, batch_input_shape)

            ray_embeddings = self.bbox3d_head.get_camera_embeddings(
                intrinsics, batch_input_shape
            )

            # Depth Head
            depth_preds, depth_latents = self.depth_head(
                depth_feats, intrinsics, batch_input_shape
            )

            depth_maps = []
            for i, depth_pred in enumerate(depth_preds):
                if padding is not None:
                    pad_left, pad_right, pad_top, pad_bottom = padding[i]

                    depth_pred = depth_pred[
                        pad_top : batch_input_img_h - pad_bottom,
                        pad_left : batch_input_img_w - pad_right,
                    ]

                depth_maps.append(
                    F.interpolate(
                        depth_pred.unsqueeze(0).unsqueeze(0),
                        size=original_hw[i],
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )

            (
                memory_text,
                text_token_mask,
                hidden_states,
                references,
            ) = self.forward_transformer(
                feat_flatten,
                lvl_pos_embed_flatten,
                memory_mask,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                text_dict,
                ray_embeddings=ray_embeddings,
                depth_latents=depth_latents,
            )

            all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
                hidden_states, references, memory_text, text_token_mask
            )

            all_layers_outputs_3d = self.bbox3d_head(
                hidden_states, ray_embeddings, depth_latents=depth_latents
            )

            cls_scores = all_layers_cls_scores[-1]
            bbox_preds = all_layers_bbox_preds[-1]
            boxes3d_preds = all_layers_outputs_3d[-1]

            boxes = []
            boxes3d = []
            scores = []
            class_ids = []
            categories = []
            for i, bbox_pred in enumerate(bbox_preds):
                cls_score = cls_scores[i]
                det_bboxes, det_scores, det_labels, det_bboxes3d = (
                    self.roi2det3d(
                        cls_score,
                        bbox_pred,
                        token_positive_maps[i],
                        input_hw[i],
                        original_hw[i],
                        boxes3d_preds[i],
                        intrinsics[i],
                        padding[i] if padding is not None else None,
                    )
                )

                boxes.append(det_bboxes)
                scores.append(det_scores)
                class_ids.append(det_labels)
                boxes3d.append(det_bboxes3d)

                # Get the categories text
                cur_categories = []
                for label in det_labels:
                    cur_categories.append(entities[i][label])

                categories.append(cur_categories)

        return Det3DOut(
            boxes,
            boxes3d,
            scores,
            class_ids,
            depth_maps=depth_maps,
            categories=categories,
        )

    def forward(
        self,
        images: Tensor,
        input_hw: list[tuple[int, int]],
        intrinsics: Tensor,
        boxes2d: Tensor | None = None,
        boxes2d_classes: Tensor | None = None,
        original_hw: list[tuple[int, int]] | None = None,
        input_texts: Sequence[str] | str | None = None,
        input_tokens_positive: list[dict[int, list[int, int]]] | None = None,
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
        padding: list[list[int]] | None = None,
        **kwargs,
    ) -> GroundingDINO3DOut | Det3DOut:
        """Forward function."""
        if self.training:
            assert boxes2d is not None and boxes2d_classes is not None
            return self._forward_train(
                images,
                input_texts,
                boxes2d,
                boxes2d_classes,
                input_hw,
                intrinsics,
                input_tokens_positive=input_tokens_positive,
                padding=padding,
                **kwargs,
            )

        assert original_hw is not None
        return self._forward_test(
            images,
            input_texts,
            text_prompt_mapping,
            input_hw,
            original_hw,
            intrinsics,
            padding,
        )
