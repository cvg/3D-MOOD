"""Grounding DINO model.

modified from mmdetection.
"""

from collections.abc import Sequence
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import BatchEncoding
from vis4d.common.ckpt import load_model_checkpoint
from vis4d.common.logging import rank_zero_warn
from vis4d.op.base import BaseModel
from vis4d.op.layer.positional_encoding import SinePositionalEncoding

from opendet3d.model.language.mm_bert import BertModel
from opendet3d.op.detect.deformable_detr import get_valid_ratio
from opendet3d.op.detect.dino import CdnQueryGenerator
from opendet3d.op.detect.grounding_dino import (
    GroundingDINOHead,
    GroundingDinoTransformerDecoder,
    GroundingDinoTransformerEncoder,
    RoI2Det,
)
from opendet3d.op.fpp.channel_mapper import ChannelMapper
from opendet3d.op.language.grounding import (
    chunks,
    clean_label_name,
    create_positive_map,
    create_positive_map_label_to_token,
    run_ner,
)

REV_KEYS = [
    (r"\.conv.weight", ".weight"),
    (r"\.conv.bias", ".bias"),
    (r"\.gn", ".norm"),
]


class GroundingDINOOut(NamedTuple):
    """Output of the Grounding DINO model."""

    all_layers_cls_scores: list[Tensor]
    all_layers_bbox_preds: list[Tensor]
    enc_outputs_class: Tensor
    enc_outputs_coord: Tensor
    text_token_mask: Tensor
    dn_meta: dict[str, Tensor]
    positive_maps: list[Tensor]


class DetOut(NamedTuple):
    """Output of the Grounding DINO model."""

    boxes: list[Tensor]
    scores: list[Tensor]
    class_ids: list[Tensor]
    categories: list[list[str]] | None = None


class GroundingDINO(nn.Module):
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
        weights: str | None = None,
    ) -> None:
        """Create the Grounding DINO model."""
        super().__init__()
        self.texts = texts
        self.custom_entities = custom_entities
        self.chunked_size = chunked_size

        self.num_queries = num_queries

        self.backbone = basemodel
        self.neck = neck

        # Encoder
        self.encoder = GroundingDinoTransformerEncoder(
            num_levels=num_feature_levels, use_checkpoint=use_checkpoint
        )

        self.embed_dims = self.encoder.embed_dims
        self.positional_encoding = SinePositionalEncoding(
            num_feats=128, normalize=True, offset=0.0, temperature=20
        )

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f"embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(num_feature_levels, self.embed_dims)
        )

        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # Decoder
        self.decoder = GroundingDinoTransformerDecoder(
            num_levels=num_feature_levels
        )

        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)

        # Grounding DINO head
        self.bbox_head = bbox_head or GroundingDINOHead(
            num_classes=256, num_decoder_layer=self.decoder.num_layers
        )

        self.roi2det = roi2det or RoI2Det()

        self.dn_query_generator = CdnQueryGenerator(
            num_classes=self.bbox_head.num_classes,
            embed_dims=self.embed_dims,
            num_matching_queries=self.num_queries,
            label_noise_scale=0.5,
            box_noise_scale=1.0,  # 0.4 for DN-DETR
            dynamic=True,
            num_groups=None,
            num_dn_queries=100,
        )

        # Language model configuration
        self._special_tokens = ". "

        # text modules
        self.language_model = language_model or BertModel(
            name="bert-base-uncased",
            max_tokens=256,
            pad_to_max=False,
            use_sub_sentence_represent=True,
            special_tokens_list=["[CLS]", "[SEP]", ".", "?"],
            add_pooling_layer=False,
            use_checkpoint=use_checkpoint,
        )

        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True,
        )

        self._init_weights()

        if weights is not None:
            load_model_checkpoint(self, weights, rev_keys=REV_KEYS)

    def _init_weights(self) -> None:
        """Initialize weights."""
        # DINO
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        nn.init.normal_(self.level_embed)

        # G-DINO
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def get_captions_and_tokens_positive(
        self,
        text_prompt: list[str],
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
    ) -> tuple[str, list[list[int]]]:
        """Enhance the text prompts with the text mapping."""
        captions = ""
        tokens_positive = []
        for word in text_prompt:
            if text_prompt_mapping is not None and word in text_prompt_mapping:
                enhanced_text_dict = text_prompt_mapping[word]
                if "prefix" in enhanced_text_dict:
                    captions += enhanced_text_dict["prefix"]

                start_i = len(captions)
                if "name" in enhanced_text_dict:
                    captions += enhanced_text_dict["name"]
                else:
                    captions += word
                end_i = len(captions)

                tokens_positive.append([[start_i, end_i]])

                if "suffix" in enhanced_text_dict:
                    captions += enhanced_text_dict["suffix"]
            else:
                tokens_positive.append(
                    [[len(captions), len(captions) + len(word)]]
                )
                captions += word
            captions += self._special_tokens
        return captions, tokens_positive

    def get_tokens_and_prompts(
        self,
        text_prompt: str | list[str],
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
    ) -> tuple[BatchEncoding, str, list[list[int]], list[str]]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(text_prompt, list):
            captions, tokens_positive = self.get_captions_and_tokens_positive(
                text_prompt, text_prompt_mapping
            )

            tokenized = self.language_model.tokenizer(
                [captions], padding="longest", return_tensors="pt"
            )
            entities = text_prompt
        else:
            if not text_prompt.endswith("."):
                captions = text_prompt + self._special_tokens
            else:
                captions = text_prompt

            tokenized = self.language_model.tokenizer(
                [captions], padding="longest", return_tensors="pt"
            )
            tokens_positive, entities = run_ner(captions)

        return tokenized, captions, tokens_positive, entities

    def get_positive_map(
        self, tokenized: BatchEncoding, tokens_positive: list[list[int]]
    ) -> tuple[dict, Tensor]:
        """Get the positive map and label to token."""
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.bbox_head.cls_branches[
                self.decoder.num_layers
            ].max_text_len,
        )
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1
        )
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        text_prompt: str | list[str],
        custom_entities: bool = False,
        tokens_positive: list[list[int, int]] | int | None = None,
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
    ) -> tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            assert isinstance(
                text_prompt, str
            ), "Text prompt should be a string with given positive tokens."

            if not text_prompt.endswith("."):
                captions = text_prompt + self._special_tokens
            else:
                captions = text_prompt

            if tokens_positive == -1:
                return None, captions, None, captions
            else:
                assert isinstance(
                    tokens_positive, list
                ), "Positive tokens should be a list of list[int] if not -1."
                tokenized = self.language_model.tokenizer(
                    [captions], padding="longest", return_tensors="pt"
                )
                positive_map_label_to_token, positive_map = (
                    self.get_positive_map(tokenized, tokens_positive)
                )

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(captions[t[0] : t[1]])
                    entities.append(" / ".join(instance_entities))

                return (
                    positive_map_label_to_token,
                    captions,
                    positive_map,
                    entities,
                )

        if custom_entities:
            if isinstance(text_prompt, str):
                text_prompt = text_prompt.strip(self._special_tokens)
                text_prompt = text_prompt.split(self._special_tokens)
                text_prompt = list(filter(lambda x: len(x) > 0, text_prompt))
            text_prompt = [clean_label_name(i) for i in text_prompt]

        if self.chunked_size > 0:
            assert not self.training, "Chunked size is only for testing."
            (
                positive_map_label_to_token,
                captions,
                positive_map,
                entities,
            ) = self.get_tokens_positive_and_prompts_chunked(
                text_prompt, text_prompt_mapping
            )
        else:
            tokenized, captions, tokens_positive, entities = (
                self.get_tokens_and_prompts(text_prompt, text_prompt_mapping)
            )
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive
            )

        return positive_map_label_to_token, captions, positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
        self,
        text_prompt: list[str],
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
    ):
        """Get the tokens positive and prompts for the caption."""
        text_prompt_chunked = chunks(text_prompt, self.chunked_size)
        ids_chunked = chunks(
            list(range(1, len(text_prompt) + 1)), self.chunked_size
        )

        positive_map_label_to_token_chunked = []
        captions_chunked = []
        positive_map_chunked = []
        entities_chunked = []
        for i in range(len(ids_chunked)):
            captions, tokens_positive = self.get_captions_and_tokens_positive(
                text_prompt_chunked[i], text_prompt_mapping
            )

            tokenized = self.language_model.tokenizer(
                [captions], padding="longest", return_tensors="pt"
            )
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                rank_zero_warn(
                    "Caption is too long will result in poor performance. "
                    "Please reduce the chunked size."
                )

            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive
            )

            captions_chunked.append(captions)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token
            )
            positive_map_chunked.append(positive_map)
            entities_chunked.append(text_prompt_chunked[i])

        return (
            positive_map_label_to_token_chunked,
            captions_chunked,
            positive_map_chunked,
            entities_chunked,
        )

    # TODO: Move this to deformable DETR
    def pre_transformer(
        self,
        feats: list[Tensor],
        input_hw: list[tuple[int, int]],
        batch_input_shape: tuple[int, int],
        padding: list[list[int]] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor, Tensor]:
        """Process image features before transformer."""
        batch_size = feats[0].size(0)

        # construct binary masks for the transformer.
        batch_input_img_h, batch_input_img_w = batch_input_shape
        same_shape_flag = all(
            [
                s[0] == batch_input_img_h and s[1] == batch_input_img_w
                for s in input_hw
            ]
        )

        if same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(
                    self.positional_encoding(None, inputs=feat)
                )
        else:
            check_center = not (padding is None)
            masks = feats[0].new_ones(
                (batch_size, batch_input_img_h, batch_input_img_w)
            )
            for img_id in range(batch_size):
                img_h, img_w = input_hw[img_id]

                if padding is None:
                    masks[img_id, :img_h, :img_w] = 0
                else:
                    pad_left, pad_right, pad_top, pad_bottom = padding[img_id]
                    masks[
                        img_id,
                        pad_top : batch_input_img_h - pad_bottom,
                        pad_left : batch_input_img_w - pad_right,
                    ] = 0

            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:])
                    .to(torch.bool)
                    .squeeze(0)
                )
                mlvl_pos_embeds.append(
                    self.positional_encoding(mlvl_masks[-1])
                )

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
            zip(feats, mlvl_masks, mlvl_pos_embeds)
        ):
            batch_size, c, _, _ = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat(
            (
                spatial_shapes.new_zeros((1,)),  # (num_level)
                spatial_shapes.prod(1).cumsum(0)[:-1],
            )
        )
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [get_valid_ratio(m, check_center) for m in mlvl_masks],
                1,
            )
        else:
            valid_ratios = feats[0].new_ones(batch_size, len(feats), 2)

        return (
            feat_flatten,
            lvl_pos_embed_flatten,
            mask_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        )

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
                dn_meta,
            )

        return memory_text, text_token_mask, hidden_states, list(references)

    # TODO: Move this to deformable DETR
    def gen_encoder_output_proposals(
        self,
        memory: Tensor,
        memory_mask: Tensor | None,
        spatial_shapes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Generate proposals from encoded memory. The function will only be
        used when `as_two_stage` is `True`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).

        Returns:
            tuple: A tuple of transformed memory and proposals.

            - output_memory (Tensor): The transformed memory for obtaining
              top-k proposals, has shape (bs, num_feat_points, dim).
            - output_proposals (Tensor): The inverse-normalized proposal, has
              shape (batch_size, num_keys, 4) with the last dimension arranged
              as (cx, cy, w, h).
        """

        bs = memory.size(0)
        proposals = []
        _cur = 0  # start index in the sequence of the current level
        for lvl, HW in enumerate(spatial_shapes):
            H, W = HW

            if memory_mask is not None:
                mask_flatten_ = memory_mask[:, _cur : (_cur + H * W)].view(
                    bs, H, W, 1
                )
                valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1).unsqueeze(
                    -1
                )
                valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1).unsqueeze(
                    -1
                )
                scale = torch.cat([valid_W, valid_H], 1).view(bs, 1, 1, 2)
            else:
                if not isinstance(HW, Tensor):
                    HW = memory.new_tensor(HW)
                scale = HW.unsqueeze(0).flip(dims=[0, 1]).view(1, 1, 1, 2)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device
                ),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device
                ),
                indexing="ij",
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            grid = (grid.unsqueeze(0).expand(bs, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            proposal = torch.cat((grid, wh), -1).view(bs, -1, 4)
            proposals.append(proposal)
            _cur += H * W

        output_proposals = torch.cat(proposals, 1)

        # do not use `all` to make it exportable to onnx
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).sum(-1, keepdim=True) == output_proposals.shape[-1]

        # inverse_sigmoid
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        if memory_mask is not None:
            output_proposals = output_proposals.masked_fill(
                memory_mask.unsqueeze(-1), float("inf")
            )

        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        if memory_mask is not None:
            output_memory = memory.masked_fill(
                memory_mask.unsqueeze(-1), float(0)
            )
        else:
            output_memory = memory

        # [bs, sum(hw), 2]
        output_memory = output_memory.masked_fill(
            ~output_proposals_valid, float(0)
        )
        output_memory = self.memory_trans_fc(output_memory)
        output_memory = self.memory_trans_norm(output_memory)

        return output_memory, output_proposals

    def _forward_train(
        self,
        images: Tensor,
        input_texts: list[list[str]] | None,
        boxes2d: Tensor,
        boxes2d_classes: Tensor,
        input_hw: list[tuple[int, int]],
        input_tokens_positive: list[list[int, int]] | None = None,
    ) -> GroundingDINOOut:
        """Forward train."""
        batch_size = images.shape[0]

        # if "tokens_positive" in batch_data_samples[0]:
        if input_tokens_positive is not None:
            positive_maps = []
            for tokens_positive_dict, text_prompt, gt_label in zip(
                input_tokens_positive, input_texts, boxes2d_classes
            ):
                tokenized = self.language_model.tokenizer(
                    [text_prompt], padding="longest", return_tensors="pt"
                )
                new_tokens_positive = [
                    tokens_positive_dict[label.item()] for label in gt_label
                ]
                _, positive_map = self.get_positive_map(
                    tokenized, new_tokens_positive
                )
                positive_maps.append(positive_map)
            new_text_prompts = input_texts
        else:
            new_text_prompts = []
            positive_maps = []

            # All the text prompts are the same, so there is no need to
            # calculate them multiple times.
            if (
                input_texts is None
                or len(set(["".join(t) for t in input_texts])) == 1
            ):
                if input_texts is None:
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

        visual_feats = self.backbone(images)[2:]
        visual_feats = self.neck(visual_feats)

        batch_input_img_h, batch_input_img_w = images.shape[-2:]
        batch_input_shape = (batch_input_img_h, batch_input_img_w)

        (
            feat_flatten,
            lvl_pos_embed_flatten,
            memory_mask,
            spatial_shapes,
            level_start_index,
            valid_ratios,
        ) = self.pre_transformer(visual_feats, input_hw, batch_input_shape)

        (
            memory_text,
            text_token_mask,
            hidden_states,
            references,
            enc_outputs_class,
            enc_outputs_coord,
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
        )

        all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
            hidden_states, references, memory_text, text_token_mask
        )

        return GroundingDINOOut(
            all_layers_cls_scores,
            all_layers_bbox_preds,
            enc_outputs_class,
            enc_outputs_coord,
            text_token_mask,
            dn_meta,
            positive_maps,
        )

    def _forward_test(
        self,
        images: Tensor,
        input_texts: list[str] | None,
        text_prompt_mapping: list[dict[str, dict[str, str]]] | None,
        input_hw: list[tuple[int, int]],
        original_hw: list[tuple[int, int]],
    ) -> DetOut:
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

        visual_feats = self.backbone(images)[2:]
        visual_feats = self.neck(visual_feats)

        if isinstance(text_prompts[0], list):
            # TODO: Support chunked text prompts in the future.
            pass
            # assert batch_size == 1, "Batch size should be 1 for chunked text."
            # count = 0
            # results_list = []

            # entities = [[item for lst in entities[0] for item in lst]]

            # for i, captions in enumerate(text_prompts[0]):
            #     token_positive_map = token_positive_maps[0][i]

            #     text_dict = self.language_model(captions)

            #     # text feature map layer
            #     if self.text_feat_map is not None:
            #         text_dict["embedded"] = self.text_feat_map(
            #             text_dict["embedded"]
            #         )

            #     head_inputs_dict = self.forward_transformer(
            #         copy.deepcopy(visual_feats), text_dict, input_hw
            #     )
            #     pred_instances = self.bbox_head.predict(
            #         **head_inputs_dict,
            #         batch_token_positive_maps=token_positive_map,
            #     )[0]

            #     if len(pred_instances) > 0:
            #         pred_instances.labels += count
            #     count += len(token_positive_maps_once)
            #     results_list.append(pred_instances)
            # results_list = [results_list[0].cat(results_list)]
        else:
            # extract text feats
            text_dict = self.language_model(list(text_prompts))

            # text feature map layer
            if self.text_feat_map is not None:
                text_dict["embedded"] = self.text_feat_map(
                    text_dict["embedded"]
                )

            batch_input_shape = (batch_input_img_h, batch_input_img_w)

            (
                feat_flatten,
                lvl_pos_embed_flatten,
                memory_mask,
                spatial_shapes,
                level_start_index,
                valid_ratios,
            ) = self.pre_transformer(visual_feats, input_hw, batch_input_shape)

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
            )

            all_layers_cls_scores, all_layers_bbox_preds = self.bbox_head(
                hidden_states, references, memory_text, text_token_mask
            )

            cls_scores = all_layers_cls_scores[-1]
            bbox_preds = all_layers_bbox_preds[-1]

            boxes = []
            scores = []
            class_ids = []
            categories = []
            for i, bbox_pred in enumerate(bbox_preds):
                cls_score = cls_scores[i]
                det_bboxes, det_scores, det_labels = self.roi2det(
                    cls_score,
                    bbox_pred,
                    token_positive_maps[i],
                    input_hw[i],
                    original_hw[i],
                )
                boxes.append(det_bboxes)
                scores.append(det_scores)
                class_ids.append(det_labels)

                # Get the categories text
                cur_categories = []
                for label in det_labels:
                    cur_categories.append(entities[i][label])

                categories.append(cur_categories)

        return DetOut(boxes, scores, class_ids, categories=categories)

    def forward(
        self,
        images: Tensor,
        input_hw: list[tuple[int, int]],
        boxes2d: Tensor | None = None,
        boxes2d_classes: Tensor | None = None,
        original_hw: list[tuple[int, int]] | None = None,
        input_texts: Sequence[str] | str | None = None,
        input_tokens_positive: list[dict[int, list[int, int]]] | None = None,
        text_prompt_mapping: dict[str, dict[str, str]] | None = None,
    ) -> GroundingDINOOut | DetOut:
        """Forward function."""
        if self.training:
            assert boxes2d is not None and boxes2d_classes is not None
            return self._forward_train(
                images,
                input_texts,
                boxes2d,
                boxes2d_classes,
                input_hw,
                input_tokens_positive=input_tokens_positive,
            )

        assert original_hw is not None
        return self._forward_test(
            images,
            input_texts,
            text_prompt_mapping,
            input_hw,
            original_hw,
        )
