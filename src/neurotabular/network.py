"""Compact neural backbones for tabular data."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def embedding_dimension(
    cardinality: int,
    *,
    n_samples: int | None = None,
    n_features: int = 1,
    memory_budget: int = 128,
) -> int:
    """Choose a bounded width from cardinality, data size, and feature budget."""

    if cardinality < 3:
        raise ValueError("cardinality must include missing, unknown, and rare IDs.")
    sample_factor = 1.0
    if n_samples is not None:
        sample_factor = 0.9 + 0.1 * min(1.0, math.log10(max(10, n_samples)) / 4.0)
    unconstrained = min(
        16,
        max(2, math.ceil(2.0 * cardinality**0.25 * sample_factor)),
    )
    per_feature_budget = max(2, memory_budget // max(1, n_features))
    return min(unconstrained, per_feature_budget)


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


class GatedInputProjection(nn.Module):
    """Project features with one inexpensive learned feature gate."""

    def __init__(self, input_width: int, hidden_dim: int, activation: str) -> None:
        super().__init__()
        self.value = nn.Linear(input_width, hidden_dim)
        self.gate = nn.Linear(input_width, hidden_dim)
        self.activation = _activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a gated hidden representation."""

        return self.activation(self.value(inputs)) * torch.sigmoid(self.gate(inputs))


class NumericalEmbedding(nn.Module):
    """Embed standardized scalar features without mixing feature identities."""

    def __init__(
        self,
        n_continuous_features: int,
        n_side_features: int,
        mode: str,
        knots: torch.Tensor | None,
        embedding_dim: int,
    ) -> None:
        super().__init__()
        if mode not in {"scalar", "affine", "periodic", "piecewise"}:
            raise ValueError(
                "numerical_embedding must be 'scalar', 'affine', 'periodic', "
                "or 'piecewise'."
            )
        self.n_continuous_features = n_continuous_features
        self.n_side_features = n_side_features
        self.mode = mode
        self.embedding_dim = embedding_dim
        if mode == "piecewise" and n_continuous_features:
            if knots is None or knots.ndim != 2:
                raise ValueError("piecewise numerical embedding requires 2D knots.")
            if knots.shape[0] != n_continuous_features:
                raise ValueError("numerical knot count does not match input features.")
            self.register_buffer("knots", knots.to(dtype=torch.float32).clone())
            width = knots.shape[1] - 1
            self.scale = nn.Parameter(torch.ones(n_continuous_features, width))
            self.bias = nn.Parameter(torch.zeros(n_continuous_features, width))
            self.missing_embedding = nn.Parameter(
                torch.zeros(n_continuous_features, width)
            )
            self.continuous_output_width = n_continuous_features * width
        elif mode == "affine" and n_continuous_features:
            self.weight = nn.Parameter(
                torch.empty(n_continuous_features, embedding_dim)
            )
            self.bias = nn.Parameter(torch.zeros(n_continuous_features, embedding_dim))
            self.missing_embedding = nn.Parameter(
                torch.zeros(n_continuous_features, embedding_dim)
            )
            self.continuous_output_width = n_continuous_features * embedding_dim
        elif mode == "periodic" and n_continuous_features:
            frequency_count = max(1, embedding_dim // 2)
            frequencies = torch.logspace(-1.0, 0.7, frequency_count)
            self.frequencies = nn.Parameter(
                frequencies.repeat(n_continuous_features, 1)
            )
            self.periodic_projection = nn.Parameter(
                torch.empty(
                    n_continuous_features,
                    2 * frequency_count,
                    embedding_dim,
                )
            )
            self.bias = nn.Parameter(torch.zeros(n_continuous_features, embedding_dim))
            self.missing_embedding = nn.Parameter(
                torch.zeros(n_continuous_features, embedding_dim)
            )
            self.continuous_output_width = n_continuous_features * embedding_dim
        else:
            self.continuous_output_width = 2 * n_continuous_features
        self.output_width = self.continuous_output_width + n_side_features
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        if hasattr(self, "weight"):
            nn.init.normal_(self.weight, mean=0.0, std=0.2)
        if hasattr(self, "periodic_projection"):
            nn.init.normal_(self.periodic_projection, mean=0.0, std=0.2)
        if hasattr(self, "missing_embedding"):
            nn.init.normal_(self.missing_embedding, mean=0.0, std=0.02)

    def forward(self, numerical: torch.Tensor) -> torch.Tensor:
        """Transform values and append leakage-safe numerical side features."""

        feature_count = self.n_continuous_features
        side_start = 2 * feature_count
        side = numerical[:, side_start:]
        if not feature_count:
            return side
        if self.mode == "scalar":
            return numerical
        values = numerical[:, :feature_count]
        missing = numerical[:, feature_count:side_start]
        if self.mode == "piecewise":
            left = self.knots[:, :-1]
            width = (self.knots[:, 1:] - left).clamp_min(1e-12)
            encoded = ((values.unsqueeze(-1) - left) / width).clamp(0.0, 1.0)
            embedded = encoded * self.scale + self.bias
        elif self.mode == "affine":
            embedded = values.unsqueeze(-1) * self.weight + self.bias
        else:
            phase = 2.0 * math.pi * values.unsqueeze(-1) * self.frequencies
            periodic = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
            embedded = torch.einsum("bfi,fid->bfd", periodic, self.periodic_projection)
            embedded = embedded + self.bias
        embedded = embedded + missing.unsqueeze(-1) * self.missing_embedding
        flattened = embedded.flatten(1)
        return torch.cat((flattened, side), dim=1) if side.shape[1] else flattened


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
        n_continuous_features: int | None = None,
        numerical_knots: torch.Tensor | None = None,
        numerical_embedding: str = "scalar",
        numerical_embedding_dim: int = 8,
        dataset_size: int | None = None,
        embedding_memory_budget: int = 128,
        feature_gating: bool = False,
        categorical_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if architecture not in {"plain", "residual"}:
            raise ValueError("architecture must be 'plain' or 'residual'.")
        if normalization not in {"none", "layer_norm"}:
            raise ValueError("normalization must be 'none' or 'layer_norm'.")
        if n_continuous_features is None:
            n_continuous_features = n_numeric_features // 2
        n_side_features = n_numeric_features - 2 * n_continuous_features
        if n_side_features < 0:
            raise ValueError("n_numeric_features is inconsistent with feature counts.")
        if not 0.0 <= categorical_dropout < 1.0:
            raise ValueError("categorical_dropout must be in [0, 1).")
        feature_count = n_continuous_features + len(categorical_cardinalities)
        self.embedding_dimensions = [
            embedding_dimension(
                cardinality,
                n_samples=dataset_size,
                n_features=feature_count,
                memory_budget=embedding_memory_budget,
            )
            for cardinality in categorical_cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, width)
                for cardinality, width in zip(
                    categorical_cardinalities,
                    self.embedding_dimensions,
                    strict=True,
                )
            ]
        )
        self.numerical_embedding = NumericalEmbedding(
            n_continuous_features=n_continuous_features,
            n_side_features=n_side_features,
            mode=numerical_embedding,
            knots=numerical_knots,
            embedding_dim=numerical_embedding_dim,
        )
        embedding_width = sum(self.embedding_dimensions)
        input_width = self.numerical_embedding.output_width + embedding_width
        if input_width == 0:
            raise ValueError("The network requires at least one input feature.")
        self.input_width = input_width
        self.architecture = architecture
        self.categorical_dropout = categorical_dropout
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        if architecture == "residual":
            self.input_projection = (
                GatedInputProjection(input_width, hidden_dim, activation)
                if feature_gating
                else nn.Sequential(
                    nn.Linear(input_width, hidden_dim),
                    _activation(activation),
                )
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

        numerical_features = self.numerical_embedding(numerical)
        if self.training and self.categorical_dropout > 0.0:
            drop_mask = torch.rand_like(categorical, dtype=torch.float32)
            categorical = categorical.masked_fill(
                drop_mask < self.categorical_dropout, 1
            )
        parts = [numerical_features] if numerical_features.shape[1] else []
        categorical_parts = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        if categorical_parts:
            categorical_features = self.embedding_dropout(
                torch.cat(categorical_parts, 1)
            )
            parts.append(categorical_features)
        features = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        hidden = self.input_projection(features)
        hidden = self.backbone(hidden)
        return self.output(hidden).squeeze(1)
