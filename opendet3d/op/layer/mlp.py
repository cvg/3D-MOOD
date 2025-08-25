"""Multi-layer perceptron (MLP)."""

import torch.nn.functional as F
from torch import Tensor, nn


class SwiGLU(nn.Module):
    """SwiGLU activation function."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x, gates = x.chunk(2, dim=-1)
        return x * F.silu(gates)


class MLP(nn.Module):
    """Multi-layer perceptron (MLP) module."""

    def __init__(
        self,
        input_dim: int,
        expansion: int = 4,
        dropout: float = 0.0,
        gated: bool = False,
        output_dim: int | None = None,
    ) -> None:
        """Creates an instance of the class."""
        super().__init__()
        if gated:
            expansion = int(expansion * 2 / 3)
        hidden_dim = int(input_dim * expansion)
        output_dim = output_dim if output_dim is not None else input_dim
        self.norm = nn.LayerNorm(input_dim)
        self.proj1 = nn.Linear(input_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.GELU() if not gated else SwiGLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.norm(x)
        x = self.proj1(x)
        x = self.act(x)
        x = self.proj2(x)
        x = self.dropout(x)
        return x

    def __call__(self, x: Tensor) -> Tensor:
        """Type definition for call implementation."""
        return self._call_impl(x)


class SimpleMLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
