"""Compact neural backbones for tabular data."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def embedding_dimension(cardinality: int) -> int:
    """Choose a conservative per-feature embedding width."""

    if cardinality < 3:
        raise ValueError("cardinality must include missing, unknown, and rare IDs.")
    return min(16, max(2, math.ceil(2.0 * cardinality**0.25)))


def _activation(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("activation must be 'silu' or 'gelu'.")


class ResidualBlock(nn.Module):
    """A small pre-normalized feed-forward residual block."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        activation: str,
        normalization: str,
    ) -> None:
        super().__init__()
        self.normalization = (
            nn.LayerNorm(hidden_dim) if normalization == "layer_norm" else nn.Identity()
        )
        self.linear_in = nn.Linear(hidden_dim, hidden_dim * 2)
        self.activation = _activation(activation)
        self.dropout_in = nn.Dropout(dropout)
        self.linear_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout_out = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the feed-forward branch and residual connection."""

        hidden = self.normalization(inputs)
        hidden = self.linear_in(hidden)
        hidden = self.activation(hidden)
        hidden = self.dropout_in(hidden)
        hidden = self.linear_out(hidden)
        return inputs + self.dropout_out(hidden)


class TabularNetwork(nn.Module):
    """Combine numerical values and categorical embeddings into one logit."""

    def __init__(
        self,
        n_numeric_features: int,
        categorical_cardinalities: Sequence[int],
        hidden_dim: int,
        n_blocks: int,
        dropout: float,
        *,
        architecture: str = "residual",
        activation: str = "silu",
        normalization: str = "layer_norm",
    ) -> None:
        super().__init__()
        if architecture not in {"plain", "residual"}:
            raise ValueError("architecture must be 'plain' or 'residual'.")
        if normalization not in {"none", "layer_norm"}:
            raise ValueError("normalization must be 'none' or 'layer_norm'.")
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dimension(cardinality))
                for cardinality in categorical_cardinalities
            ]
        )
        embedding_width = sum(
            embedding_dimension(cardinality)
            for cardinality in categorical_cardinalities
        )
        input_width = n_numeric_features + embedding_width
        if input_width == 0:
            raise ValueError("The network requires at least one input feature.")
        self.architecture = architecture
        if architecture == "residual":
            self.input_projection = nn.Sequential(
                nn.Linear(input_width, hidden_dim),
                _activation(activation),
            )
            self.backbone = nn.Sequential(
                *[
                    ResidualBlock(
                        hidden_dim,
                        dropout,
                        activation,
                        normalization,
                    )
                    for _ in range(n_blocks)
                ]
            )
        else:
            layers: list[nn.Module] = []
            current_width = input_width
            for _ in range(n_blocks):
                layers.extend(
                    [
                        nn.Linear(current_width, hidden_dim),
                        (
                            nn.LayerNorm(hidden_dim)
                            if normalization == "layer_norm"
                            else nn.Identity()
                        ),
                        _activation(activation),
                        nn.Dropout(dropout),
                    ]
                )
                current_width = hidden_dim
            self.input_projection = nn.Identity()
            self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0.0, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def set_output_bias(self, value: float) -> None:
        """Initialize the binary head from the weighted class prior."""

        with torch.no_grad():
            self.output.bias.fill_(value)

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable parameters."""

        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self, numerical: torch.Tensor, categorical: torch.Tensor
    ) -> torch.Tensor:
        """Return one binary-classification logit per row."""

        parts = [numerical] if numerical.shape[1] else []
        parts.extend(
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        )
        features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        hidden = self.input_projection(features)
        hidden = self.backbone(hidden)
        return self.output(hidden).squeeze(1)
