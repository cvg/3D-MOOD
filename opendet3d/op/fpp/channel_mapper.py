"""Channel Mapper."""

from collections.abc import Sequence

from torch import Tensor, nn
from vis4d.op.layer.conv2d import Conv2d
from vis4d.op.layer.util import build_activation_layer, build_norm_layer
from vis4d.op.layer.weight_init import xavier_init


class ChannelMapper(nn.Module):
    """Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int,
        num_outs: int = 4,
        kernel_size: int = 3,
        norm: str | None = None,
        num_groups: int | None = None,
        activation: str | None = "ReLU",
        bias: bool = False,
    ) -> None:
        """Create an instance of ChannelMapper.

        Args:
            in_channels (Sequence[int]): Number of input channels per scale.
            out_channels (int): Number of output channels (used at each scale).
            num_outs (int, optional): Number of output feature maps. There
                would be extra_convs when num_outs larger than the length of
                in_channels.
            kernel_size (int, optional): kernel_size for reducing channels used
                at each scale. Default: 3.
            norm (str | None, optional): Type of normalization layer. Default:
                None.
            num_groups (int | None, optional): Number of groups for GroupNorm.
                Default: None.
            activation (str | None, optional): Type of activation layer.
                Default: "ReLU".
            bias (bool): Bias for Conv2d. Default: False.
        """
        super().__init__()
        self.in_channels = in_channels

        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            norm_layer = (
                build_norm_layer(norm, out_channels, num_groups=num_groups)
                if norm
                else None
            )
            activation_layer = (
                build_activation_layer(activation) if activation else None
            )

            self.convs.append(
                Conv2d(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    norm=norm_layer,
                    activation=activation_layer,
                    bias=bias,
                )
            )

        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels

                norm_layer = (
                    build_norm_layer(norm, out_channels, num_groups=num_groups)
                    if norm
                    else None
                )
                activation_layer = (
                    build_activation_layer(activation) if activation else None
                )

                self.extra_convs.append(
                    Conv2d(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        norm=norm_layer,
                        activation=activation_layer,
                        bias=bias,
                    )
                )
        else:
            self.extra_convs = None

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights."""
        for m in self.convs:
            if isinstance(m, Conv2d):
                xavier_init(m, distribution="uniform")

        if self.extra_convs is not None:
            for m in self.extra_convs:
                if isinstance(m, Conv2d):
                    xavier_init(m, distribution="uniform")

    def forward(self, inputs: list[Tensor]) -> list[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs is not None:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return outs

    def __call__(self, x: list[Tensor]) -> list[Tensor]:
        """Type definition for call implementation."""
        return self._call_impl(x)
