import pytest
import torch


@pytest.fixture(params=[0, 1, 2])
def gradient_detach_interval(request):
    """Parametrized fixture for gradient detach intervals. `0` means no detaching."""
    return (i := request.param), "no detaching!" if i == 0 else f"detaching every {i}"


def test_samudra_forward_pass(create_samudra_model, gradient_detach_interval):
    """Test Samudra forward pass with various gradient detaching intervals."""
    interval, interval_desc = gradient_detach_interval
    model, train_data = create_samudra_model(interval)
    loss_fn = torch.nn.MSELoss()

    loss = model(train_data, loss_fn=loss_fn)
    assert not torch.isnan(loss), (
        f"Loss is NaN for interval={interval} ({interval_desc})"
    )
    assert loss.requires_grad, (
        f"Loss should require grad for interval={interval} ({interval_desc})"
    )


def test_samudra_backward_pass(create_samudra_model, gradient_detach_interval):
    """Test Samudra backward pass with various gradient detaching intervals."""
    interval, interval_desc = gradient_detach_interval
    model, train_data = create_samudra_model(interval)
    loss_fn = torch.nn.MSELoss()

    # Forward pass
    loss = model(train_data, loss_fn=loss_fn)

    # Backward pass
    loss.backward()

    # Check that gradients exist for model parameters
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    assert grad_count > 0, (
        f"Model should have gradients after backward pass for interval={interval} ({interval_desc})"
    )
    assert grad_count == total_params, (
        f"Expected all {total_params} parameters to have gradients for interval={interval} ({interval_desc}), "
        f"got {grad_count}"
    )
