import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.config import SamudraConfig, UNetBackboneConfig
from ocean_emulators.constants import TensorMap
from ocean_emulators.datasets import TrainData
from ocean_emulators.utils.data import DataSource, Normalize
from ocean_emulators.utils.multiton import MultitonScope


@pytest.fixture(params=[0, 1, 2])
def gradient_detach_interval(request):
    """Parametrized fixture for gradient detach intervals. `0` means no detaching."""
    return (i := request.param), "no detaching!" if i == 0 else f"detaching every {i}"


@pytest.fixture
def create_samudra_model():
    """Factory fixture for creating Samudra models with different gradient detach intervals."""

    def _create_model_helper(gradient_detach_interval: int):
        with MultitonScope():
            # Set up minimal data structures needed by Samudra
            h, w = 8, 8
            coords = {
                "lev": [0],
                "lat": (["y"], np.linspace(-90, 90, h)),
                "lon": (["x"], np.linspace(-180, 180, w)),
            }
            data = xr.Dataset(
                {
                    "thetao": (["lev", "y", "x"], np.random.randn(1, h, w)),
                    "hfds": (["y", "x"], np.random.randn(h, w)),
                },
                coords=coords,
            )
            ones = xr.Dataset(
                {
                    "thetao": (["lev", "y", "x"], np.ones((1, h, w))),
                    "hfds": (["y", "x"], np.ones((h, w))),
                },
                coords=coords,
            )
            src = DataSource(name="dummy", data=data, means=data, stds=ones)

            # Initialize TensorMap and Normalize
            TensorMap.init_instance("thetao_1", "hfds")
            Normalize.init_instance(
                src,
                TensorMap.get_instance().prognostic_var_names,
                TensorMap.get_instance().boundary_var_names,
                torch.ones(h, w),
                torch.ones(h, w),
            )

            # Create Samudra model with the specified gradient_detach_interval
            model = SamudraConfig(
                unet=UNetBackboneConfig(
                    ch_width=[4, 8],
                    dilation=[1, 1],
                    n_layers=[1, 1],
                ),
                pos_channels=0,
                gradient_detach_interval=gradient_detach_interval,
            ).build(
                in_channels=2,
                out_channels=1,
                hist=1,
                wet=torch.ones(1, h, w, dtype=torch.bool),
                area_weights=torch.ones(h, w),
                static_data=None,
            )

            # Create TrainData compatible with model dimensions
            train_data = TrainData(num_prognostic_channels=1)
            for step in range(4):
                input_tensor = torch.randn(1, 2, h, w, requires_grad=True)
                label_tensor = torch.randn(1, 1, h, w)
                train_data.insert(input_tensor, label_tensor)

            return model, train_data

    return _create_model_helper


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
