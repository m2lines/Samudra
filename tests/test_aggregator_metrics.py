import torch

from ocean_emulators.aggregator.metrics import weighted_mean


def test_weighted_mean():
    # Test case 1: Basic weighted mean with no NaN values
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    result = weighted_mean(tensor, weights)
    assert torch.allclose(result, torch.tensor(3.0))

    # Test case 2: Weighted mean with NaN values
    tensor_with_nan = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    result_with_nan = weighted_mean(tensor_with_nan, weights)
    assert torch.allclose(result_with_nan, torch.tensor(3.25))

    # Test case 3: Without weights
    result_no_weights = weighted_mean(tensor)
    assert torch.allclose(result_no_weights, torch.tensor(2.5))

    # Test case 4: With keepdim=True
    result_keepdim = weighted_mean(tensor, weights, keepdim=True)
    assert result_keepdim.shape == (1, 1)


def test_weighted_mean_with_nan_as_zero():
    tensor = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])
    tensor_masked = torch.where(torch.isnan(tensor), 0.0, tensor)
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    wrong_result = weighted_mean(tensor_masked, weights)
    correct_result = weighted_mean(tensor, weights)
    assert correct_result > wrong_result
