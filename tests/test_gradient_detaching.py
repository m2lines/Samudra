import pytest
import torch
import xarray as xr

from ocean_emulators.config import SamudraConfig, UNetBackboneConfig
from ocean_emulators.constants import TensorMap
from ocean_emulators.datasets import TrainData
from ocean_emulators.utils.data import DataSource, Normalize
from ocean_emulators.utils.multiton import MultitonScope


@pytest.fixture
def samudra_setup():
    with MultitonScope():
        # Set up minimal data structures needed by Samudra
        h, w = 8, 8
        coords = {
            "lev": [0],
            "lat": (["y"], torch.linspace(-90, 90, h).numpy()),
            "lon": (["x"], torch.linspace(-180, 180, w).numpy()),
        }
        data = xr.Dataset(
            {
                "thetao": (["lev", "y", "x"], torch.randn(1, h, w).numpy()),
                "hfds": (["y", "x"], torch.randn(h, w).numpy()),
            },
            coords=coords,
        )
        ones = xr.Dataset(
            {
                "thetao": (["lev", "y", "x"], torch.ones(1, h, w).numpy()),
                "hfds": (["y", "x"], torch.ones(h, w).numpy()),
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

        # Create Samudra model configuration
        config = SamudraConfig(
            unet=UNetBackboneConfig(
                ch_width=[4, 8],
                dilation=[1, 1],
                n_layers=[1, 1],
            ),
            pos_channels=0,
        )

        def create_samudra_model(gradient_detach_interval=0):
            # Set the gradient_detach_interval on the config object
            config.gradient_detach_interval = gradient_detach_interval
            return config.build(
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

        return create_samudra_model, train_data


class TestSamudraGradientDetaching:
    def test_samudra_forward_pass_no_detaching(self, samudra_setup):
        """Test Samudra forward pass without gradient detaching."""
        create_model, train_data = samudra_setup
        model = create_model(gradient_detach_interval=0)
        loss_fn = torch.nn.MSELoss()

        loss = model(train_data, loss_fn=loss_fn)
        assert not torch.isnan(loss)
        assert loss.requires_grad

    def test_samudra_forward_pass_with_detaching(self, samudra_setup):
        """Test Samudra forward pass with gradient detaching."""
        create_model, train_data = samudra_setup
        model = create_model(gradient_detach_interval=1)
        loss_fn = torch.nn.MSELoss()

        loss = model(train_data, loss_fn=loss_fn)
        assert not torch.isnan(loss)
        assert loss.requires_grad

    def test_samudra_backward_pass(self, samudra_setup):
        """Test Samudra backward pass without gradient detaching."""
        create_model, train_data = samudra_setup
        model = create_model(gradient_detach_interval=0)
        loss_fn = torch.nn.MSELoss()

        # Forward pass
        loss = model(train_data, loss_fn=loss_fn)

        # Backward pass
        loss.backward()

        # Check that gradients exist for model parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "Model should have gradients after backward pass"

    def test_samudra_backward_pass_with_detaching(self, samudra_setup):
        """Test Samudra backward pass with gradient detaching."""
        create_model, train_data = samudra_setup
        model = create_model(gradient_detach_interval=1)
        loss_fn = torch.nn.MSELoss()

        # Forward pass
        loss = model(train_data, loss_fn=loss_fn)

        # Backward pass
        loss.backward()

        # Check that gradients exist for model parameters
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        assert has_grad, "Model should have gradients after backward pass"

    def test_samudra_gradient_detaching_with_higher_interval(self, samudra_setup):
        """Test Samudra with gradient detaching interval of 2."""
        create_model, train_data = samudra_setup
        model = create_model(gradient_detach_interval=2)
        loss_fn = torch.nn.MSELoss()

        # Test forward pass
        loss = model(train_data, loss_fn=loss_fn)
        assert not torch.isnan(loss)
        assert loss.requires_grad

        # Test backward pass
        loss.backward()

        # Check that all parameters have gradients
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        assert grad_count > 0, "Model should have gradients after backward pass"
        assert grad_count == total_params, (
            f"Expected all {total_params} parameters to have gradients, got {grad_count}"
        )
