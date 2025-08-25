"""Attention layer."""

from functools import partial
from math import log2, pi

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn

from .mlp import MLP
from .nystorm import NystromAttention


class LayerScale(nn.Module):
    """Layer scale."""

    def __init__(
        self,
        dim: int,
        init_values: float | Tensor = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        """Forward."""
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class AttentionBlock(nn.Module):
    """Attention block."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = dim
        self.context_dim = context_dim or dim

        self.norm_attnx = nn.LayerNorm(self.hidden_dim)
        self.norm_attnctx = nn.LayerNorm(self.context_dim)

        self.q = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.kv = nn.Linear(self.context_dim, self.hidden_dim * 2)

        self.cosine = cosine
        self.dropout = dropout
        self.out = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.ls1 = (
            LayerScale(dim, layer_scale)
            if layer_scale > 0.0
            else nn.Identity()
        )

        self.mlp = MLP(
            self.hidden_dim, expansion=expansion, dropout=dropout, gated=gated
        )

        self.ls2 = (
            LayerScale(dim, layer_scale)
            if layer_scale > 0.0
            else nn.Identity()
        )

    def attn(
        self,
        x: Tensor,
        attn_bias: Tensor | None = None,
        context: Tensor | None = None,
        pos_embed: Tensor | None = None,
        pos_embed_context: Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> Tensor:
        """Attention."""
        x = self.norm_attnx(x)

        context = self.norm_attnctx(context)

        q = rearrange(self.q(x), "b n (h d) -> b h n d", h=self.num_heads)

        k, v = rearrange(
            self.kv(context),
            "b n (kv h d) -> b h n d kv",
            h=self.num_heads,
            kv=2,
        ).unbind(dim=-1)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b h n d", h=self.num_heads
                )
                q = q + pos_embed

            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b h n d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim

        x = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout, attn_mask=attn_bias
        )
        x = rearrange(x, "b h n d -> b n (h d)")
        x = self.out(x)
        return x

    def forward(
        self,
        x: Tensor,
        attn_bias: Tensor | None = None,
        context: Tensor | None = None,
        pos_embed: Tensor | None = None,
        pos_embed_context: Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> Tensor:
        """Forward."""
        context = x if context is None else context

        x = (
            self.ls1(
                self.attn(
                    x,
                    rope=rope,
                    attn_bias=attn_bias,
                    context=context,
                    pos_embed=pos_embed,
                    pos_embed_context=pos_embed_context,
                )
            )
            + x
        )

        return self.ls2(self.mlp(x)) + x


class NystromBlock(AttentionBlock):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        expansion: int = 4,
        dropout: float = 0.0,
        cosine: bool = False,
        gated: bool = False,
        layer_scale: float = 1.0,
        context_dim: int | None = None,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            expansion=expansion,
            dropout=dropout,
            cosine=cosine,
            gated=gated,
            layer_scale=layer_scale,
            context_dim=context_dim,
        )
        self.attention_fn = NystromAttention(
            num_landmarks=128, num_heads=num_heads, dropout=dropout
        )

    def attn(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        pos_embed: torch.Tensor | None = None,
        pos_embed_context: torch.Tensor | None = None,
        rope: nn.Module | None = None,
    ) -> torch.Tensor:
        x = self.norm_attnx(x)
        context = self.norm_attnctx(context)
        k, v = rearrange(
            self.kv(context),
            "b n (kv h d) -> b n h d kv",
            h=self.num_heads,
            kv=2,
        ).unbind(dim=-1)
        q = rearrange(self.q(x), "b n (h d) -> b n h d", h=self.num_heads)

        if rope is not None:
            q = rope(q)
            k = rope(k)
        else:
            if pos_embed is not None:
                pos_embed = rearrange(
                    pos_embed, "b n (h d) -> b n h d", h=self.num_heads
                )
                q = q + pos_embed
            if pos_embed_context is not None:
                pos_embed_context = rearrange(
                    pos_embed_context, "b n (h d) -> b n h d", h=self.num_heads
                )
                k = k + pos_embed_context

        if self.cosine:
            q, k = map(partial(F.normalize, p=2, dim=-1), (q, k))  # cosine sim
        x = self.attention_fn(q, k, v, key_padding_mask=attn_bias)
        x = rearrange(x, "b n h d -> b n (h d)")
        x = self.out(x)
        return x


class PositionEmbeddingSine(nn.Module):
    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * pi
        self.scale = scale

    def forward(
        self, x: torch.Tensor, mask: Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)),
                device=x.device,
                dtype=torch.bool,
            )
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
