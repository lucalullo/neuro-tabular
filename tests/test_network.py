import pytest
import torch

from neurotabular.network import TabularNetwork, embedding_dimension


@pytest.mark.parametrize(
    "mode",
    ["scalar", "affine", "periodic", "piecewise"],
)
def test_numerical_embedding_variants_forward_and_backpropagate(mode):
    knots = torch.tensor([[-2.0, -1.0, 0.0, 1.0, 2.0], [-3.0, -0.5, 0.5, 1.5, 3.0]])
    model = TabularNetwork(
        n_numeric_features=5,
        categorical_cardinalities=[8],
        hidden_dim=8,
        n_blocks=1,
        dropout=0.0,
        n_continuous_features=2,
        numerical_knots=knots,
        numerical_embedding=mode,
        dataset_size=100,
        feature_gating=True,
    )
    numerical = torch.tensor([[-0.5, 0.2, 0.0, 0.0, 0.8], [0.0, 1.0, 1.0, 0.0, 0.2]])
    logits = model(numerical, torch.tensor([[3], [4]]))
    logits.sum().backward()
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_adaptive_embedding_dimension_obeys_limits_and_memory_budget():
    assert embedding_dimension(3, n_samples=50, n_features=2) >= 2
    assert embedding_dimension(10_000, n_samples=1_000_000, n_features=1) <= 16
    assert (
        embedding_dimension(
            10_000,
            n_samples=1_000_000,
            n_features=32,
            memory_budget=64,
        )
        == 2
    )
