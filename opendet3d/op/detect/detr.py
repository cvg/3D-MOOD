"""DETR op."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from vis4d.op.layer.attention import MultiheadAttention
from vis4d.op.layer.transformer import FFN, get_clones


class DetrTransformerEncoder(nn.Module):
    """Transformer Encoder of DETR.

    It consists of multiple DetrTransformerEncoderLayer layers.
    """

    def __init__(
        self,
        num_layers: int,
        layer: DetrTransformerEncoderLayer | None = None,
    ) -> None:
        """Create the instance of the class.

        Args:
            num_layers (int): Number of encoder layers.
            layer (DetrTransformerEncoderLayer, optional): The encoder layer.
        """
        super().__init__()
        self.num_layers = num_layers

        layer = layer or DetrTransformerEncoderLayer()

        self.layers = get_clones(layer, num_layers)
        self.embed_dims = self.layers[0].embed_dims

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward function of encoder."""
        for i, _ in enumerate(self.layers):
            layer: DetrTransformerEncoderLayer = self.layers[i]
            query = layer(query, query_pos, attn_mask, key_padding_mask)
        return query

    def __call__(
        self,
        query: Tensor,
        query_pos: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(query, query_pos, attn_mask, key_padding_mask)


class DetrTransformerEncoderLayer(nn.Module):
    """Encoder layer in DETR transformer.

    It consists of a multi-head self-attention layer and a feed forward
    network layer. The normalization layers are LayerNorm.
    """

    def __init__(
        self,
        self_attn: MultiheadAttention | None = None,
        ffn: FFN | None = None,
    ) -> None:
        """Create the instance of the class.

        Args:
            self_attn (MultiheadAttention, optional): The self attention layer.
            ffn (FFN, optional): The feed forward network layer.
        """
        super().__init__()
        self._init_attn_layer(self_attn)

        self.ffn = ffn or FFN(embed_dims=256, feedforward_channels=1024)

        assert (
            self.self_attn.embed_dims == self.ffn.embed_dims
        ), "The embed_dims of self attention and FFN must be the same."
        self.embed_dims = self.self_attn.embed_dims

        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dims) for _ in range(2)]
        )

    def _init_attn_layer(self, self_attn: MultiheadAttention | None) -> None:
        """Initialize attention layers."""
        self.self_attn = self_attn or MultiheadAttention(
            embed_dims=256, num_heads=8, dropout=0.0, batch_first=True
        )

    def forward(
        self,
        query: Tensor,
        query_pos: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Forward function of an encoder layer."""
        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = self.norms[0](query)
        query = self.ffn(query)
        query = self.norms[1](query)

        return query

    def __call__(
        self,
        query: Tensor,
        query_pos: Tensor,
        attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(query, query_pos, attn_mask, key_padding_mask)


class DetrTransformerDecoder(nn.Module):
    """Transformer Decoder of DETR."""

    def __init__(
        self,
        num_layers: int,
        layer: DetrTransformerDecoderLayer | None = None,
        with_post_norm: bool = True,
        return_intermediate: bool = True,
    ) -> None:
        """Create the instance of the class."""
        super().__init__()
        self.num_layers = num_layers
        self.with_post_norm = with_post_norm
        self.return_intermediate = return_intermediate

        layer = layer or DetrTransformerDecoderLayer()

        self.layers = get_clones(layer, num_layers)
        self.embed_dims = self.layers[0].embed_dims

        if self.with_post_norm:
            self.post_norm = nn.LayerNorm(self.embed_dims)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        """Forward.

        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor): The input key, has shape (bs, num_keys, dim).
            value (Tensor): The input value with the same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`, with the
                same shape as `query`.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`.
            key_padding_mask (Tensor): The `key_padding_mask` of `cross_attn`
                input. ByteTensor, has shape (bs, num_value).

        Returns:
            Tensor: The forwarded results will have shape
            (num_decoder_layers, bs, num_queries, dim) if
            `return_intermediate` is `True` else (1, bs, num_queries, dim).
        """
        intermediate = []
        for i, _ in enumerate(self.layers):
            layer: DetrTransformerDecoderLayer = self.layers[i]

            query = layer(
                query,
                key=key,
                value=value,
                query_pos=query_pos,
                key_pos=key_pos,
                key_padding_mask=key_padding_mask,
            )

            if self.return_intermediate:
                if self.with_post_norm:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)

        if self.return_intermediate:
            return torch.stack(intermediate)

        if self.with_post_norm:
            query = self.post_norm(query)

        return query.unsqueeze(0)

    def __call__(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_pos: Tensor,
        key_pos: Tensor,
        key_padding_mask: Tensor,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(
            query, key, value, query_pos, key_pos, key_padding_mask
        )


class DetrTransformerDecoderLayer(nn.Module):
    """Implements decoder layer in DETR transformer."""

    def __init__(
        self,
        self_attn: MultiheadAttention | None = None,
        cross_attn: MultiheadAttention | None = None,
        ffn: FFN | None = None,
    ) -> None:
        """Create the instance of the class."""
        super().__init__()
        self._init_attn_layer(self_attn, cross_attn)

        self.embed_dims = self.self_attn.embed_dims
        self.ffn = ffn or FFN(embed_dims=256, feedforward_channels=1024)

        assert (
            self.self_attn.embed_dims == self.cross_attn.embed_dims
            and self.self_attn.embed_dims == self.ffn.embed_dims
        ), (
            "The embed_dims of self attention, cross attention and FFN "
            + "must be the same."
        )
        self.embed_dims = self.self_attn.embed_dims

        norms_list = [nn.LayerNorm(self.embed_dims) for _ in range(3)]
        self.norms = nn.ModuleList(norms_list)

    def _init_attn_layer(
        self,
        self_attn: MultiheadAttention | None,
        cross_attn: MultiheadAttention | None,
    ) -> None:
        """Initialize attention layers."""
        self.self_attn = self_attn or MultiheadAttention(
            embed_dims=256,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            batch_first=True,
        )
        self.cross_attn = cross_attn or MultiheadAttention(
            embed_dims=256,
            num_heads=8,
            attn_drop=0.0,
            proj_drop=0.0,
            batch_first=True,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
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

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
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
            key=key,
            value=value,
            query_pos=query_pos,
            key_pos=key_pos,
            attn_mask=cross_attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)
        return query

    def __call__(
        self,
        query: Tensor,
        key: Tensor | None = None,
        value: Tensor | None = None,
        query_pos: Tensor | None = None,
        key_pos: Tensor | None = None,
        self_attn_mask: Tensor | None = None,
        cross_attn_mask: Tensor | None = None,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Typing."""
        return self._call_impl(
            query,
            key,
            value,
            query_pos,
            key_pos,
            self_attn_mask,
            cross_attn_mask,
            key_padding_mask,
        )
