"""Transformer encoder and decoder for Deformable DETR."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.amp import autocast
from vis4d.op.layer.attention import MultiheadAttention
from vis4d.op.layer.ms_deform_attn import MultiScaleDeformableAttention
from vis4d.op.layer.transformer import FFN, inverse_sigmoid

from .detr import (
    DetrTransformerDecoder,
    DetrTransformerDecoderLayer,
    DetrTransformerEncoder,
    DetrTransformerEncoderLayer,
)


class DeformableDetrTransformerEncoder(DetrTransformerEncoder):
    """Transformer Encoder of Deformable DETR."""

    def __init__(
        self,
        num_layers: int,
        layer: DeformableDetrTransformerEncoderLayer | None = None,
    ) -> None:
        """Create the instance of the class.

        Args:
            num_layers (int): Number of encoder layers.
            layer (DeformableDetrTransformerEncoderLayer, optional): The
                encoder layer.
        """
        layer = layer or DeformableDetrTransformerEncoderLayer()

        super().__init__(num_layers=num_layers, layer=layer)

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        key_padding_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
    ) -> Tensor:
        """Forward function of Transformer encoder."""
        reference_points = self.get_encoder_reference_points(
            spatial_shapes, valid_ratios, device=query.device
        )
        for i, _ in enumerate(self.layers):
            layer: DeformableDetrTransformerEncoderLayer = self.layers[i]

            query = layer(
                query=query,
                reference_points=reference_points,
                query_pos=query_pos,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                padding_mask=key_padding_mask,
            )
        return query

    @staticmethod
    def get_encoder_reference_points(
        spatial_shapes: Tensor,
        valid_ratios: Tensor,
        device: torch.device,
    ) -> Tensor:
        """Get the reference points used in encoder.

        Args:
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            device (obj:`device` or str): The device acquired by the
                `reference_points`.

        Returns:
            Tensor: Reference points used in decoder, has shape (bs, length,
            num_levels, 2).
        """

        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device
                ),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device
                ),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * H
            )
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * W
            )
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        # [bs, sum(hw), num_level, 2]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def __call__(
        self,
        query: Tensor,
        query_pos: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(query, query_pos, attn_mask, key_padding_mask)


class DeformableDetrTransformerEncoderLayer(DetrTransformerEncoderLayer):
    """Encoder layer of Deformable DETR."""

    def __init__(
        self,
        self_attn: MultiScaleDeformableAttention | None = None,
        ffn: FFN | None = None,
    ) -> None:
        """Create the instance of the class."""
        super().__init__(self_attn=self_attn, ffn=ffn)

    def _init_attn_layer(
        self, self_attn: MultiScaleDeformableAttention | None
    ) -> None:
        """Initialize attention layers."""
        self.self_attn = self_attn or MultiScaleDeformableAttention()

    @autocast(device_type="cuda", enabled=False)
    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        query_pos: Tensor | None,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward function of an encoder layer."""
        value = query

        query = self.self_attn(
            query=query,
            query_pos=query_pos,
            reference_points=reference_points,
            input_flatten=value,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=padding_mask,
        )

        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query

    def __call__(
        self,
        query: Tensor,
        reference_points: Tensor,
        query_pos: Tensor | None,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(
            query,
            reference_points,
            query_pos,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )


class DeformableDetrTransformerDecoder(DetrTransformerDecoder):
    """Transformer Decoder of Deformable DETR."""

    def __init__(
        self,
        num_layers: int,
        layer: DeformableDetrTransformerDecoderLayer | None = None,
        with_post_norm: bool = False,
        return_intermediate: bool = True,
    ) -> None:
        """Create an instance of the class."""
        layer = layer or DeformableDetrTransformerDecoderLayer()

        super().__init__(
            num_layers=num_layers,
            layer=layer,
            with_post_norm=with_post_norm,
            return_intermediate=return_intermediate,
        )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.Module | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward function of Transformer decoder.

        Args:
            query (Tensor): The input queries, has shape (bs, num_queries,
                dim).
            query_pos (Tensor): The input positional query, has shape
                (bs, num_queries, dim). It will be added to `query` before
                forward function.
            value (Tensor): The input values, has shape (bs, num_value, dim).
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h) when `as_two_stage` is `True`, otherwise has
                shape (bs, num_queries, 2) with the last dimension arranged
                as (cx, cy).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            reg_branches: (obj:`nn.ModuleList`, optional): Used for refining
                the regression results. Only would be passed when
                `with_box_refine` is `True`, otherwise would be `None`.

        Returns:
            tuple[Tensor]: Outputs of Deformable Transformer Decoder.

            - output (Tensor): Output embeddings of the last decoder, has
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
        intermediate_reference_points = []
        for layer_id, _ in enumerate(self.layers):
            layer: DeformableDetrTransformerDecoderLayer = self.layers[
                layer_id
            ]

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

            query = layer(
                query,
                query_pos=query_pos,
                value=value,
                key_padding_mask=key_padding_mask,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                reference_points=reference_points_input,
            )

            if reg_branches is not None:
                tmp_reg_preds = reg_branches[layer_id](query)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp_reg_preds + inverse_sigmoid(
                        reference_points
                    )
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp_reg_preds
                    new_reference_points[..., :2] = tmp_reg_preds[
                        ..., :2
                    ] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(query)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points
            )

        return query, reference_points

    def __call__(
        self,
        query: Tensor,
        query_pos: Tensor,
        value: Tensor,
        key_padding_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: nn.Module | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(
            query,
            query_pos,
            value,
            key_padding_mask,
            reference_points,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            reg_branches,
        )


class DeformableDetrTransformerDecoderLayer(DetrTransformerDecoderLayer):
    """Decoder layer of Deformable DETR."""

    def _init_attn_layer(
        self,
        self_attn: MultiheadAttention | None,
        cross_attn: MultiheadAttention | None,
    ) -> None:
        """Initialize attention layers."""
        self.self_attn = self_attn or MultiheadAttention(
            embed_dims=256,
            num_heads=8,
            attn_drop=0.1,
            batch_first=True,
            dropout_layer=nn.Dropout(0.1),
        )
        self.cross_attn = cross_attn or MultiScaleDeformableAttention(
            embed_dims=256
        )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        reference_points: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward function of a decoder layer."""
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
        )
        query = self.norms[0](query)
        query = self.cross_attn(
            query=query,
            reference_points=reference_points,
            input_flatten=value,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=key_padding_mask,
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        return query

    def __call__(
        self,
        query: Tensor,
        query_pos: Tensor | None = None,
        value: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
        spatial_shapes: Tensor | None = None,
        level_start_index: Tensor | None = None,
        reference_points: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(
            query,
            query_pos,
            value,
            key_padding_mask,
            spatial_shapes,
            level_start_index,
            reference_points,
            self_attn_mask,
        )


def get_valid_ratio(mask: Tensor, check_center: bool = False) -> Tensor:
    """Get the valid radios of feature map in a level.

    .. code:: text
                |---> valid_W <---|
             ---+-----------------+-----+---
              A |                 |     | A
              | |                 |     | |
              | |                 |     | |
        valid_H |                 |     | |
              | |                 |     | H
              | |                 |     | |
              V |                 |     | |
             ---+-----------------+     | |
                |                       | V
                +-----------------------+---
                |---------> W <---------|

        The valid_ratios are defined as:
            r_h = valid_H / H,  r_w = valid_W / W
        They are the factors to re-normalize the relative coordinates of the
        image to the relative coordinates of the current level feature map.

    Args:
        mask (Tensor): Binary mask of a feature map, has shape (bs, H, W).

    Returns:
        Tensor: valid ratios [r_w, r_h] of a feature map, has shape (1, 2).
    """
    _, H, W = mask.shape

    if check_center:
        h_idx = W // 2
        w_idx = H // 2
    else:
        h_idx = w_idx = 0

    valid_H = torch.sum(~mask[:, :, h_idx], 1)
    valid_W = torch.sum(~mask[:, w_idx, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio
