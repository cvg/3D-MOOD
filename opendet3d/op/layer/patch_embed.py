"""Image to Patch Embedding using Conv2d.

Modified from vision_transformer (https://github.com/google-research/vision_transformer), # pylint: disable=line-too-long
and mmdetection (https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/layers/transformer/utils.py). # pylint: disable=line-too-long
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch.nn.functional as F
from timm.layers import to_2tuple
from torch import Tensor, nn
from vis4d.op.layer.conv2d import Conv2d
from vis4d.op.layer.util import build_norm_layer


class AdaptivePadding(nn.Module):
    """Applies padding to input so that input can get fully covered by filter.

    It support two modes "same" and "corner". The "same" mode is same with
    "SAME" padding mode in TensorFlow, pad zero around input. The "corner" mode
    would pad zero to bottom right.

    Example:
        >>> kernel_size = 16
        >>> stride = 16
        >>> dilation = 1
        >>> input = torch.rand(1, 1, 15, 17)
        >>> adap_pad = AdaptivePadding(
        >>>     kernel_size=kernel_size,
        >>>     stride=stride,
        >>>     dilation=dilation,
        >>>     padding="corner")
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
        >>> input = torch.rand(1, 1, 16, 17)
        >>> out = adap_pad(input)
        >>> assert (out.shape[2], out.shape[3]) == (16, 32)
    """

    def __init__(
        self,
        kernel_size: int | tuple[int, int] = 1,
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding: str = "corner",
    ):
        """Create the AdaptivePadding module.

        Args:
            kernel_size (int | tuple[int, int]): Size of the kernel. Default to
                1.
            stride (int | tuple[int, int]): Stride of the filter. Default to 1.
            dilation (int | tuple[int, int]): Spacing between kernel elements.
                Default to 1.
            padding (str): Support "same" and "corner". The "corner" mode would
                pad zero to bottom right, and "same" mode would pad zero around
                input. Default to "corner".
        """
        super().__init__()

        assert padding in ("same", "corner")

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape: tuple[int, int]) -> tuple[int, int]:
        """Get the padding shape."""
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max(
            (output_h - 1) * stride_h
            + (kernel_h - 1) * self.dilation[0]
            + 1
            - input_h,
            0,
        )
        pad_w = max(
            (output_w - 1) * stride_w
            + (kernel_w - 1) * self.dilation[1]
            + 1
            - input_w,
            0,
        )
        return pad_h, pad_w

    def __call__(self, x: Tensor) -> Tensor:
        """Typing."""
        return self._call_impl(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == "same":
                x = F.pad(
                    x,
                    [
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    ],
                )
        return x


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        in_channels: int = 3,
        embed_dims: int = 768,
        patch_size: int = 16,
        padding: int | tuple[int, int] | str = 0,
        dilation: int = 1,
        norm: str | None = None,
        bias: bool = True,
        strict_img_size: bool = True,
    ):
        """Init PatchEmbed.

        Args:
            img_size (int, optional): Input image's size. Defaults to 224.
            in_channels (int, optional): Number of input image's channels.
                Defaults to 3.
            embed_dims (int, optional): Patch embedding's dim. Defaults to 768.
            patch_size (int, optional): Patch size. Defaults to 16.
            padding (int | tuple[int, int] | str, optional): Padding size.
            norm (str, optional): Normalization layer. Defaults to None, which
                means no normalization layer.
            flatten (bool, optional): If to flatten the output tensor.
                Defaults to True.
            bias (bool, optional): If to add bias to the convolution layer.
                Defaults to True.
            strict_img_size (bool, optional): If to strictly check the input
                image's size. Defaults to True.
            adaptive_padding (bool, optional): If to adaptively pad the input
                image. Defaults to False.

        Raises:
            ValueError: If the input image's size is not divisible by the patch
                size.
        """
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.embed_dims = embed_dims
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=patch_size, stride=patch_size, padding=padding
            )
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.strict_img_size = strict_img_size

        self.projection = Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            dilation=dilation,
            bias=bias,
            padding=padding,
        )
        self.norm = (
            build_norm_layer(norm, embed_dims) if norm else nn.Identity()
        )

    def __call__(self, data: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Applies the layer.

        Args:
            data (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, N, C), where N is the number of
                patches (N = H * W).
            tuple[int, int]: Spatial resolution of the output tensor.
        """
        return self._call_impl(data)

    def forward(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Forward function."""
        if self.strict_img_size:
            _, _, height, width = x.shape
            assert height == self.img_size[0], (
                f"Input image height ({height}) doesn't match model"
                f"({self.img_size})."
            )
            assert width == self.img_size[1], (
                f"Input image width ({width}) doesn't match model"
                f"({self.img_size})."
            )

        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)

        hw_shape = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
        x = self.norm(x)

        return x, hw_shape


class PatchMerging(nn.Module):
    """Merge patch feature map.

    This layer groups feature map by kernel_size, and applies norm and linear
    layers to the grouped feature map. Our implementation uses `nn.Unfold` to
    merge patch, which is about 25% faster than original implementation.
    Instead, we need to modify pretrained models for compatibility.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | str = "corner",
        dilation: int | tuple[int, int] = 1,
        bias: bool = False,
        norm: str | None = None,
    ) -> None:
        """Init.

        Args:
            in_channels (int): The num of input channels.
            out_channels (int): The num of output channels.
            kernel_size (int | tuple[int, int], optional): The kernel size in
                the unfold layer. Default to 2.
            stride (int | tuple[int, int], optional): The stride of the sliding
                blocks in the unfold layer. Default: None. If None, it will be
                set to kernel_size.
            padding (int | tuple[int, int] | string ): The padding length of
                embedding conv. When it is a string, it means the mode of
                adaptive padding, support "same" and "corner". Default to
                "corner".
            dilation (int | tuple[int, int], optional): Dilation parameter in
                the unfold layer. Default to 1.
            bias (bool, optional): Whether to add bias in linear layer or not.
                Default to False.
            norm (str | None, optional): Config str for normalization layer.
                Default to None.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if stride:
            stride = stride
        else:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adap_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            # disable the padding of unfold
            padding = 0
        else:
            self.adap_padding = None

        padding = to_2tuple(padding)
        self.sampler = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

        sample_dim = kernel_size[0] * kernel_size[1] * in_channels

        if norm is not None:
            self.norm = build_norm_layer(norm, sample_dim)
        else:
            self.norm = None

        self.reduction = nn.Linear(sample_dim, out_channels, bias=bias)

    def forward(self, x: Tensor, input_size: tuple[int, int]) -> Tensor:
        """Forward function.

        Args:
            x(Tensor): Input feature, tensor size (B, H*W, C).
            input_size (tuple[int, int]): The spatial shape of x, arrange as
                (H, W).
        """
        B, L, C = x.shape
        assert isinstance(input_size, Sequence), (
            f"Expect " f"input_size is " f"`Sequence` " f"but get {input_size}"
        )

        H, W = input_size
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C).permute([0, 3, 1, 2])  # B, C, H, W
        # Use nn.Unfold to merge patch. About 25% faster than original method,
        # but need to modify pretrained model for compatibility

        if self.adap_padding:
            x = self.adap_padding(x)
            H, W = x.shape[-2:]

        x = self.sampler(x)
        # if kernel_size=2 and stride=2, x should has shape (B, 4*C, H/2*W/2)

        out_h = (
            H
            + 2 * self.sampler.padding[0]
            - self.sampler.dilation[0] * (self.sampler.kernel_size[0] - 1)
            - 1
        ) // self.sampler.stride[0] + 1
        out_w = (
            W
            + 2 * self.sampler.padding[1]
            - self.sampler.dilation[1] * (self.sampler.kernel_size[1] - 1)
            - 1
        ) // self.sampler.stride[1] + 1

        output_size = (out_h, out_w)
        x = x.transpose(1, 2)  # B, H/2*W/2, 4*C
        x = self.norm(x) if self.norm else x
        x = self.reduction(x)
        return x, output_size
