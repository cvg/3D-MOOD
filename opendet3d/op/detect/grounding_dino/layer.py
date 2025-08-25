"""Groudning DINO layers."""

from __future__ import annotations

import math

import torch
from fairscale.nn.checkpoint import checkpoint_wrapper
from torch import Tensor, nn
from torch.amp import autocast
from vis4d.op.layer.attention import MultiheadAttention
from vis4d.op.layer.ms_deform_attn import MultiScaleDeformableAttention
from vis4d.op.layer.transformer import FFN, get_clones, inverse_sigmoid

from opendet3d.op.detect.deformable_detr import (
    DeformableDetrTransformerDecoderLayer,
    DeformableDetrTransformerEncoder,
    DeformableDetrTransformerEncoderLayer,
)
from opendet3d.op.detect.detr import DetrTransformerEncoderLayer
from opendet3d.op.detect.dino import DinoTransformerDecoder
from opendet3d.op.layer.positional_encoding import coordinate_to_encoding

from .fuse import BiAttentionBlock


class GroundingDinoTransformerEncoder(DeformableDetrTransformerEncoder):
    """Grounding DINO Transformer Encoder."""

    def __init__(
        self,
        num_layers: int = 6,
        num_levels: int = 4,
        layer: DeformableDetrTransformerEncoderLayer | None = None,
        text_layer: DetrTransformerEncoderLayer | None = None,
        fusion_layer: BiAttentionBlock | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        """Create the instance of the class."""
        layer = layer or DeformableDetrTransformerEncoderLayer(
            self_attn=MultiScaleDeformableAttention(
                embed_dims=256, num_levels=num_levels
            ),
            ffn=FFN(embed_dims=256, feedforward_channels=2048),
        )

        super().__init__(num_layers, layer)

        text_layer = text_layer or DetrTransformerEncoderLayer(
            self_attn=MultiheadAttention(
                embed_dims=256, num_heads=4, batch_first=True
            ),
            ffn=FFN(embed_dims=256, feedforward_channels=1024),
        )

        fusion_layer = fusion_layer or BiAttentionBlock(
            v_dim=256,
            l_dim=256,
            embed_dim=1024,
            num_heads=4,
            init_values=1e-4,
        )

        self.text_layers = get_clones(text_layer, num_layers)
        self.fusion_layers = get_clones(fusion_layer, num_layers)

        if use_checkpoint:
            for i in range(self.num_layers):
                self.layers[i] = checkpoint_wrapper(self.layers[i])
                self.fusion_layers[i] = checkpoint_wrapper(
                    self.fusion_layers[i]
                )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor | None = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward function of Transformer encoder.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            query_pos (Tensor): The positional encoding for query, has shape
                (bs, num_queries, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (bs, num_queries).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            memory_text (Tensor, optional): Memory text. It has shape (bs,
                len_text, text_embed_dims).
            text_attention_mask (Tensor, optional): Text token mask. It has
                shape (bs,len_text).
            pos_text (Tensor, optional): The positional encoding for text.
                Defaults to None.
            text_self_attention_masks (Tensor, optional): Text self attention
                mask. Defaults to None.
            position_ids (Tensor, optional): Text position ids.
                Defaults to None.
        """
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device
        )

        # generate pos_text
        bs, n_text, _ = memory_text.shape
        if pos_text is None and position_ids is None:
            pos_text = (
                torch.arange(n_text, device=memory_text.device)
                .float()
                .unsqueeze(0)
                .unsqueeze(-1)
                .repeat(bs, 1, 1)
            )
            pos_text = get_text_sine_pos_embed(
                pos_text, num_pos_feats=256, exchange_xy=False
            )
        if position_ids is not None:
            pos_text = get_text_sine_pos_embed(
                position_ids[..., None],
                num_pos_feats=256,
                exchange_xy=False,
            )

        # main process
        for i, _ in enumerate(self.layers):
            # fusion layer
            fusion_layer: BiAttentionBlock = self.fusion_layers[i]

            query, memory_text = fusion_layer(
                query,
                memory_text,
                attention_mask_v=key_padding_mask,
                attention_mask_l=text_attention_mask,
            )

            # text layer
            text_layer: DetrTransformerEncoderLayer = self.text_layers[i]
            text_num_heads = text_layer.self_attn.num_heads

            memory_text = text_layer(
                query=memory_text,
                query_pos=(pos_text if pos_text is not None else None),
                attn_mask=~text_self_attention_masks.repeat(
                    text_num_heads, 1, 1
                ),
            )

            # image layer
            layer: DeformableDetrTransformerEncoderLayer = self.layers[i]

            query = layer(
                query=query,
                reference_points=reference_points,
                query_pos=query_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                padding_mask=key_padding_mask,
            )

        return query, memory_text

    def __call__(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        memory_text: Tensor = None,
        text_attention_mask: Tensor = None,
        pos_text: Tensor | None = None,
        text_self_attention_masks: Tensor = None,
        position_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Typing."""
        return self._call_impl(
            query,
            query_pos,
            key_padding_mask,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            memory_text,
            text_attention_mask,
            pos_text,
            text_self_attention_masks,
            position_ids,
        )


class GroundingDinoTransformerDecoder(DinoTransformerDecoder):
    """Transformer decoder for Grounding DINO."""

    def __init__(
        self,
        num_layers: int = 6,
        num_levels: int = 4,
        layer: GroundingDinoTransformerDecoderLayer | None = None,
        with_post_norm: bool = False,
        return_intermediate: bool = True,
    ) -> None:
        """Create the instance of the class."""
        layer = layer or GroundingDinoTransformerDecoderLayer(
            num_levels=num_levels
        )

        super().__init__(
            num_layers=num_layers,
            layer=layer,
            with_post_norm=with_post_norm,
            return_intermediate=return_intermediate,
        )

    def forward(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
        memory_text: Tensor,
        text_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input query, has shape (num_queries, bs, dim).
            value (Tensor): The input values, has shape (num_value, bs, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `self_attn`
                input. ByteTensor, has shape (num_queries, bs).
            self_attn_mask (Tensor): The attention mask to prevent information
                leakage from different denoising groups and matching parts, has
                shape (num_queries_total, num_queries_total). It is `None` when
                `self.training` is `False`.
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`): Used for refining the
                regression results.

        Returns:
            tuple[Tensor]: Output queries and references of Transformer
                decoder

            - query (Tensor): Output embeddings of the last decoder, has
              shape (num_queries, bs, embed_dims) when `return_intermediate`
              is `False`. Otherwise, Intermediate output embeddings of all
              decoder layers, has shape (num_decoder_layers, num_queries, bs,
              embed_dims).
            - reference_points (Tensor): The reference of the last decoder
              layer, has shape (bs, num_queries, 4)  when `return_intermediate`
              is `False`. Otherwise, Intermediate references of all decoder
              layers, has shape (num_decoder_layers, bs, num_queries, 4). The
              coordinates are arranged as (cx, cy, w, h)
        """
        intermediate = []
        intermediate_reference_points = [reference_points]
        for lid, _ in enumerate(self.layers):
            layer: GroundingDinoTransformerDecoderLayer = self.layers[lid]

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[:, None]
                )

            query_sine_embed = coordinate_to_encoding(
                reference_points_input[:, :, 0, :]
            )
            query_pos = self.ref_point_head(query_sine_embed)

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                self_attn_mask=self_attn_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points_input,
                memory_text=memory_text,
                text_attention_mask=text_attention_mask,
            )

            if reg_branches is not None:
                tmp: Tensor = reg_branches[lid](query)
                assert reference_points.shape[-1] == 4
                new_reference_points = tmp + inverse_sigmoid(
                    reference_points, eps=1e-3
                )
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(query))
                # NOTE this is for the "Look Forward Twice" module,
                # in the DeformDETR, reference_points was appended.
                intermediate_reference_points.append(new_reference_points)

        if self.return_intermediate:
            return (
                torch.stack(intermediate),
                torch.stack(intermediate_reference_points),
            )

        return query, reference_points

    def __call__(
        self,
        query: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        self_attn_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.ModuleList,
        memory_text: Tensor,
        text_attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Typing."""
        return self._call_impl(
            query,
            value,
            key_padding_mask,
            self_attn_mask,
            reference_points,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            reg_branches,
            memory_text,
            text_attention_mask,
        )


class GroundingDinoTransformerDecoderLayer(
    DeformableDetrTransformerDecoderLayer
):
    """Decoder layer of Grounding DINO."""

    def __init__(
        self,
        num_levels: int = 4,
        self_attn: MultiheadAttention | None = None,
        cross_attn: MultiScaleDeformableAttention | None = None,
        ffn: FFN | None = None,
        cross_attn_text: MultiheadAttention | None = None,
    ) -> None:
        """Create the instance of the class."""
        self_attn = self_attn or MultiheadAttention(
            embed_dims=256, num_heads=8, batch_first=True
        )
        cross_attn = cross_attn or MultiScaleDeformableAttention(
            embed_dims=256, num_levels=num_levels
        )
        ffn = ffn or FFN(embed_dims=256, feedforward_channels=2048)

        super().__init__(
            self_attn=self_attn,
            cross_attn=cross_attn,
            ffn=ffn,
        )

        # Text cross attention and extra norm
        self.cross_attn_text = cross_attn_text or MultiheadAttention(
            embed_dims=256, num_heads=8, batch_first=True
        )
        self.norms.append(nn.LayerNorm(self.embed_dims))

    @autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        query: Tensor,
        value: Tensor | None = None,
        query_pos: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        memory_text: Tensor | None = None,
        text_attention_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        reference_points: Tensor | None = None,
    ) -> Tensor:
        """Implements decoder layer in Grounding DINO transformer.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.
            memory_text (Tensor): Memory text. It has shape (bs, len_text,
                text_embed_dims).
            text_attention_mask (Tensor): Text token mask. It has shape (bs,
                len_text).

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        # self attention
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
        )
        query = self.norms[0](query)

        # cross attention between query and text
        query = self.cross_attn_text(
            query=query,
            query_pos=query_pos,
            key=memory_text,
            value=memory_text,
            key_padding_mask=text_attention_mask,
        )
        query = self.norms[1](query)

        # cross attention between query and image
        query = self.cross_attn(
            query=query,
            query_pos=query_pos,
            reference_points=reference_points,
            input_flatten=value,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=key_padding_mask,
        )
        query = self.norms[2](query)

        query = self.ffn(query)
        query = self.norms[3](query)

        return query

    def __call__(
        self,
        query: Tensor,
        value: Tensor | None = None,
        query_pos: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        memory_text: Tensor | None = None,
        text_attention_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        reference_points: Tensor | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(
            query,
            value,
            query_pos,
            self_attn_mask,
            key_padding_mask,
            memory_text,
            text_attention_mask,
            spatial_shapes,
            level_start_index,
            reference_points,
        )


def get_text_sine_pos_embed(
    pos_tensor: Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. For example,
            input tensor is [x,y], the results will be [pos(y), pos(x)].
            Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        num_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack(
            (sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        return sin_x

    pos_res = [
        sine_func(x)
        for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    ]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res
