"""UniDepth Head."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from einops import rearrange
from timm.layers import trunc_normal_
from torch import Tensor, nn
from torch.nn import functional as F

from opendet3d.op.geometric.ray import generate_rays, rsh_cart_8
from opendet3d.op.layer.attention import (
    AttentionBlock,
    NystromBlock,
    PositionEmbeddingSine,
)
from opendet3d.op.layer.mlp import MLP
from opendet3d.op.layer.upsample import ConvUpsample
from opendet3d.op.util import flat_interpolate


class UniDepthHead(nn.Module):
    """G-DINO3D depth estimation head."""

    def __init__(
        self,
        embed_dims: int = 256,
        depth_scale: float = 2.0,
        input_dims: Sequence[int] = (256, 256, 256),
        output_scales: int = 1,
    ) -> None:
        """Initialize the depth head."""
        super().__init__()
        self.depth_scale = depth_scale
        assert (
            output_scales >= 1 and output_scales <= 3
        ), "Invalid output scales."
        self.output_scales = output_scales

        num_resolutions = len(input_dims)
        self.input_dims = input_dims
        self.num_resolutions = num_resolutions

        # Pool features as depth query
        self.features_channel_cat = nn.Linear(
            embed_dims * self.num_resolutions, embed_dims
        )
        self.to_latents = MLP(embed_dims, expansion=2)

        self.pos_embed = PositionEmbeddingSine(embed_dims // 2, normalize=True)

        self.level_embeds = nn.Parameter(
            torch.randn(self.num_resolutions, embed_dims),
            requires_grad=True,
        )
        self.level_embed_layer = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.GELU(),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
        )

        self.aggregate_16 = AttentionBlock(
            embed_dims,
            num_heads=1,
            expansion=4,
            context_dim=embed_dims,
        )

        self.prompt_camera = AttentionBlock(
            embed_dims, num_heads=1, expansion=4, context_dim=embed_dims
        )

        # 1/16 resolution
        self.project_rays_16 = MLP(81, expansion=4, output_dim=embed_dims)

        self.layers_16 = nn.ModuleList(
            [
                AttentionBlock(embed_dims, num_heads=8, expansion=4),
                NystromBlock(embed_dims, num_heads=8, expansion=4),
            ]
        )

        self.up_8 = ConvUpsample(embed_dims, expansion=4)

        if self.output_scales == 1:
            self.out_8 = nn.Conv2d(embed_dims // 2, 1, 3, padding=1)

        if self.output_scales >= 2:
            # 1/8 resolution
            embed_dims_8 = embed_dims // 2
            self.project_rays_8 = MLP(81, expansion=4, output_dim=embed_dims_8)

            self.layers_8 = nn.ModuleList(
                [
                    AttentionBlock(embed_dims_8, num_heads=4, expansion=4),
                    NystromBlock(embed_dims_8, num_heads=4, expansion=4),
                ]
            )

            self.up_4 = ConvUpsample(embed_dims_8, expansion=4)

            if self.output_scales == 2:
                self.out_4 = nn.Conv2d(embed_dims_8 // 2, 1, 3, padding=1)

        if self.output_scales == 3:
            # 1/4 resolution
            embed_dims_4 = embed_dims // 4
            self.project_rays_4 = MLP(81, expansion=4, output_dim=embed_dims_4)

            self.layers_4 = nn.ModuleList(
                [
                    AttentionBlock(embed_dims_4, num_heads=2, expansion=4),
                    NystromBlock(embed_dims_4, num_heads=2, expansion=4),
                ]
            )

            self.up_2 = ConvUpsample(embed_dims_4, expansion=4)

            self.out_2 = nn.Conv2d(embed_dims_4 // 2, 1, 3, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_rsh_cart(self, rays_embedding: Tensor) -> Tensor:
        """Get real spherical harmonic."""
        return rsh_cart_8(rays_embedding)

    def forward(
        self, feats: Tensor, intrinsics: Tensor, image_hw: tuple[int, int]
    ) -> Tensor:
        """Forward."""
        # Camera Embedding
        rays_hr, _ = generate_rays(intrinsics, image_hw)

        # 1/16 shape
        shape = image_hw[0] // 16, image_hw[1] // 16

        latents = []
        for _, feat in enumerate(feats):
            latent = (
                F.interpolate(
                    feat,
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                .flatten(2)
                .permute(0, 2, 1)
            )

            latents.append(latent)

        # positional embeddings, spatial and level
        level_embed = torch.cat(
            [
                self.level_embed_layer(self.level_embeds)[i : i + 1]
                .unsqueeze(0)
                .repeat(feats[0].shape[0], shape[0] * shape[1], 1)
                for i in range(self.num_resolutions)
            ],
            dim=1,
        )
        pos_embed = self.pos_embed(
            torch.zeros(
                feats[0].shape[0],
                1,
                shape[0],
                shape[1],
                device=feats[0].device,
                requires_grad=False,
            )
        )
        pos_embed = rearrange(pos_embed, "b c h w -> b (h w) c").repeat(
            1, self.num_resolutions, 1
        )

        features_tokens = torch.cat(latents, dim=1)
        features_tokens_pos = pos_embed + level_embed

        features_channels = torch.cat(latents, dim=-1)
        features_16 = self.features_channel_cat(features_channels)
        latents_16 = self.to_latents(features_16)

        # Aggregate features: F -> D
        latents_16 = self.aggregate_16(
            latents_16,
            context=features_tokens,
            pos_embed_context=features_tokens_pos,
        )

        # 1/16 shape
        rays_embedding_16 = F.normalize(
            flat_interpolate(rays_hr, old=image_hw, new=shape), dim=-1
        )

        rays_embedding_16 = self.project_rays_16(
            self.get_rsh_cart(rays_embedding_16)
        )

        # Aggregate camera: D -> D|E
        latents_16 = self.prompt_camera(latents_16, context=rays_embedding_16)

        outs = []
        depth_latents = []

        # Block 16 - Out 8
        for layer in self.layers_16:
            latents_16 = layer(latents_16, pos_embed=rays_embedding_16)

        latents_8 = self.up_8(
            rearrange(
                latents_16,
                "b (h w) c -> b c h w",
                h=shape[0],
                w=shape[1],
            ).contiguous()
        )

        if self.output_scales == 1:
            out_8 = self.out_8(
                rearrange(
                    latents_8,
                    "b (h w) c -> b c h w",
                    h=shape[0] * 2,
                    w=shape[1] * 2,
                )
            )
            outs.append(out_8)
        depth_latents.append(latents_8.detach())

        if self.output_scales >= 2:
            # 1/8 shape
            rays_embedding_8 = F.normalize(
                flat_interpolate(
                    rays_hr, old=image_hw, new=(shape[0] * 2, shape[1] * 2)
                ),
                dim=-1,
            )

            rays_embedding_8 = self.project_rays_8(
                self.get_rsh_cart(rays_embedding_8)
            )

            # Block 8 - Out 4
            for layer in self.layers_8:
                latents_8 = layer(latents_8, pos_embed=rays_embedding_8)

            latents_4 = self.up_4(
                rearrange(
                    latents_8,
                    "b (h w) c -> b c h w",
                    h=shape[0] * 2,
                    w=shape[1] * 2,
                ).contiguous()
            )

            if self.output_scales == 2:
                out_4 = self.out_4(
                    rearrange(
                        latents_4,
                        "b (h w) c -> b c h w",
                        h=shape[0] * 4,
                        w=shape[1] * 4,
                    )
                )
                outs.append(out_4)
            depth_latents.append(latents_4.detach())

        if self.output_scales == 3:
            # 1/4 shape
            rays_embedding_4 = F.normalize(
                flat_interpolate(
                    rays_hr, old=image_hw, new=(shape[0] * 4, shape[1] * 4)
                ),
                dim=-1,
            )

            rays_embedding_4 = self.project_rays_4(
                self.get_rsh_cart(rays_embedding_4)
            )

            # Block 4 - Out 2
            for layer in self.layers_4:
                latents_4 = layer(latents_4, pos_embed=rays_embedding_4)

            latents_2 = self.up_2(
                rearrange(
                    latents_4,
                    "b (h w) c -> b c h w",
                    h=shape[0] * 4,
                    w=shape[1] * 4,
                ).contiguous()
            )
            out_2 = self.out_2(
                rearrange(
                    latents_2,
                    "b (h w) c -> b c h w",
                    h=shape[0] * 8,
                    w=shape[1] * 8,
                )
            )
            outs.append(out_2)
            depth_latents.append(latents_2.detach())

        # MS Outputs
        depth_preds = (
            sum(
                [
                    F.interpolate(
                        torch.exp((out / self.depth_scale).clamp(-10.0, 10.0)),
                        size=image_hw,
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                    for out in outs
                ]
            )
            / len(outs)
        ).squeeze(1)

        depth_latent = depth_latents[-1]

        return depth_preds, depth_latent
