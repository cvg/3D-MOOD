"""Upsampling layers."""

import torch
from einops import rearrange
from torch import Tensor, nn


class CvnxtBlock(nn.Module):
    def __init__(
        self,
        dim,
        kernel_size=7,
        layer_scale=1.0,
        expansion=4,
        dilation=1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=dilation * (kernel_size - 1) // 2,
            groups=dim,
            dilation=dilation,
            padding_mode=padding_mode,
        )  # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, expansion * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expansion * dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones((dim)))
            if layer_scale > 0.0
            else 1.0
        )

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        x = self.gamma * x
        x = input + x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return x


class ConvUpsample(nn.Module):
    """Convolutional upsampling layer."""

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int | None = None,
        num_layers: int = 2,
        expansion: int = 4,
        layer_scale: float = 1.0,
        kernel_size: int = 7,
    ) -> None:
        """Init."""
        super().__init__()

        if output_dim is None:
            output_dim = hidden_dim // 2

        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                CvnxtBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale=layer_scale,
                )
            )

        self.up = nn.Sequential(
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1, padding=0),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x


class ConvUpsampleShuffle(nn.Module):
    def __init__(
        self,
        hidden_dim,
        num_layers: int = 2,
        expansion: int = 4,
        layer_scale: float = 1.0,
        kernel_size: int = 7,
    ):
        super().__init__()
        self.convs = nn.ModuleList([])
        for _ in range(num_layers):
            self.convs.append(
                CvnxtBlock(
                    hidden_dim,
                    kernel_size=kernel_size,
                    expansion=expansion,
                    layer_scale=layer_scale,
                )
            )
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(
                hidden_dim // 4, hidden_dim // 2, kernel_size=3, padding=1
            ),
        )

    def forward(self, x: Tensor):
        for conv in self.convs:
            x = conv(x)
        x = self.up(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return x
