import torch

from ocean_emulators.aggregator.metrics import (
    area_weighted_mean,
    weighted_mean,
    weighted_mean_with_nan_as_zero,
)


def test_weighted_mean():
    # Test case 1: Basic weighted mean with no NaN values
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    result = weighted_mean(tensor, weights)
    expected = torch.sum(tensor * weights) / torch.sum(weights)
    assert torch.allclose(result, expected)

    # Test case 2: Weighted mean with NaN values
    tensor_with_nan = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    result_with_nan = weighted_mean(tensor_with_nan, weights)
    # Only non-NaN values should contribute to the mean
    valid_mask = ~torch.isnan(tensor_with_nan)
    expected_with_nan = torch.sum(
        tensor_with_nan[valid_mask] * weights[valid_mask]
    ) / torch.sum(weights[valid_mask])
    assert torch.allclose(result_with_nan, expected_with_nan)

    # Test case 3: Without weights
    result_no_weights = weighted_mean(tensor)
    expected_no_weights = tensor.nanmean(dim=(-2, -1))
    assert torch.allclose(result_no_weights, expected_no_weights)

    # Test case 4: With keepdim=True
    result_keepdim = weighted_mean(tensor, weights, keepdim=True)
    assert result_keepdim.shape == (1, 1)


def test_area_weighted_mean():
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    area_weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    result = area_weighted_mean(data, area_weights)
    expected = weighted_mean(data, area_weights)
    assert torch.allclose(result, expected)


def test_weighted_mean_with_nan_as_zero():
    tensor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    tensor_masked = torch.where(torch.isnan(tensor), 0.0, tensor)
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    result = weighted_mean_with_nan_as_zero(tensor_masked, weights)
    expected = weighted_mean(tensor, weights)
    assert expected > result
