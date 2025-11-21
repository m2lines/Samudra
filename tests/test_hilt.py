"""Unit tests for HilT model architecture."""

import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.config import HilTConfig
from ocean_emulators.constants import TensorMap
from ocean_emulators.datasets import TrainData
from ocean_emulators.utils.data import DataSource, Normalize
from ocean_emulators.utils.multiton import MultitonScope


@pytest.fixture
def create_hilt_model():
    """Factory fixture for creating HilT models with different configurations."""

    def _create_model_helper(
        h: int = 8,
        w: int = 16,  # Must be divisible by 2 for stem downsampling
        embed_dim: int = 32,
        depths: list[int] | None = None,
        num_heads: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        stem_downsample: int = 2,
    ):
        """Create a minimal HilT model for testing.

        Args:
            h: Height of input (must be divisible by 2**num_levels after stem)
            w: Width of input (must be divisible by 2**num_levels after stem)
            embed_dim: Base embedding dimension
            depths: Encoder stage depths (default: [1, 1])
            num_heads: Attention heads per stage (default: [2, 4])
            kernel_sizes: NAT kernel sizes (default: [7, 5])
            stem_downsample: Stem downsampling factor
        """
        # Use minimal defaults for fast testing
        depths = depths or [1, 1]
        num_heads = num_heads or [2, 4]
        kernel_sizes = kernel_sizes or [7, 5]

        with MultitonScope():
            # Set up minimal data structures needed by HilT
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

            # Create HilT model
            model = HilTConfig(
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                kernel_sizes=kernel_sizes,
                stem_downsample=stem_downsample,
            ).build(
                in_channels=2,
                out_channels=1,
                hist=1,
                wet=torch.ones(1, h, w, dtype=torch.bool),
                area_weights=torch.ones(h, w),
                static_data=None,
                lat=torch.from_numpy(data.lat.values),
                lon=torch.from_numpy(data.lon.values),
            )

            # Create TrainData compatible with model dimensions
            train_data = TrainData(num_prognostic_channels=1)
            for step in range(4):
                input_tensor = torch.randn(1, 2, h, w, requires_grad=True)
                label_tensor = torch.randn(1, 1, h, w)
                train_data.insert(input_tensor, label_tensor)

            return model, train_data

    return _create_model_helper


def test_hilt_forward_pass(create_hilt_model):
    """Test HilT forward pass completes without errors."""
    model, train_data = create_hilt_model()
    loss_fn = torch.nn.MSELoss()

    loss = model(train_data, loss_fn=loss_fn)

    assert not torch.isnan(loss), "Loss is NaN"
    assert not torch.isinf(loss), "Loss is infinite"
    assert loss.requires_grad, "Loss should require grad"


def test_hilt_backward_pass(create_hilt_model):
    """Test HilT backward pass completes without errors."""
    model, train_data = create_hilt_model()
    loss_fn = torch.nn.MSELoss()

    # Forward pass
    loss = model(train_data, loss_fn=loss_fn)

    # Backward pass
    loss.backward()

    # Check that gradients exist for model parameters
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    assert grad_count > 0, "Model should have gradients after backward pass"
    assert grad_count == total_params, (
        f"Expected all {total_params} parameters to have gradients, got {grad_count}"
    )


@pytest.mark.skip(reason="NATTEN requires CUDA or Flex Attention support")
def test_hilt_shape_preservation(create_hilt_model):
    """Test that HilT preserves input spatial dimensions in output."""
    h, w = 16, 32
    model, _ = create_hilt_model(h=h, w=w)

    # Create a single input sample
    batch_size = 2
    input_tensor = torch.randn(batch_size, 2, h, w)

    # Forward pass (bypassing TrainData for direct shape testing)
    with torch.no_grad():
        # Call forward_once directly to get the core model output
        output = model.forward_once(input_tensor)

    # Check output shape matches input spatial dimensions
    assert output.shape == (batch_size, 1, h, w), (
        f"Expected output shape {(batch_size, 1, h, w)}, got {output.shape}"
    )


@pytest.mark.parametrize("stem_downsample", [1, 2])
def test_hilt_stem_downsampling(create_hilt_model, stem_downsample):
    """Test HilT with different stem downsampling factors."""
    h, w = 16, 32
    model, train_data = create_hilt_model(h=h, w=w, stem_downsample=stem_downsample)
    loss_fn = torch.nn.MSELoss()

    loss = model(train_data, loss_fn=loss_fn)
    assert not torch.isnan(loss), f"Loss is NaN with stem_downsample={stem_downsample}"


@pytest.mark.skip(reason="NATTEN requires CUDA or Flex Attention support")
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_hilt_different_batch_sizes(create_hilt_model, batch_size):
    """Test HilT with different batch sizes."""
    h, w = 16, 32
    model, _ = create_hilt_model(h=h, w=w)

    input_tensor = torch.randn(batch_size, 2, h, w)

    with torch.no_grad():
        output = model.forward_once(input_tensor)

    assert output.shape[0] == batch_size, (
        f"Expected batch size {batch_size}, got {output.shape[0]}"
    )
    assert output.shape[1:] == (1, h, w), (
        f"Expected spatial shape (1, {h}, {w}), got {output.shape[1:]}"
    )


@pytest.mark.skip(reason="NATTEN requires CUDA or Flex Attention support")
def test_hilt_larger_architecture(create_hilt_model):
    """Test HilT with a larger architecture closer to production config."""
    h, w = 32, 64  # Larger grid
    model, train_data = create_hilt_model(
        h=h,
        w=w,
        embed_dim=48,
        depths=[1, 2, 2],  # 3 encoder stages
        num_heads=[2, 4, 8],
        kernel_sizes=[9, 7, 5],
        stem_downsample=2,
    )
    loss_fn = torch.nn.MSELoss()

    loss = model(train_data, loss_fn=loss_fn)
    assert not torch.isnan(loss), "Loss is NaN for larger architecture"

    # Test shape preservation with direct forward pass
    with torch.no_grad():
        output = model.forward_once(torch.randn(1, 2, h, w))
    assert output.shape == (1, 1, h, w), (
        f"Shape mismatch: expected (1, 1, {h}, {w}), got {output.shape}"
    )


@pytest.mark.cuda
def test_hilt_cuda(create_hilt_model):
    """Test HilT on CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    model, train_data = create_hilt_model()
    model = model.cuda()

    # Move train_data tensors to CUDA
    train_data.input = train_data.input.cuda()
    train_data.label = train_data.label.cuda()

    loss_fn = torch.nn.MSELoss()
    loss = model(train_data, loss_fn=loss_fn)

    assert loss.device.type == "cuda", "Loss should be on CUDA device"
    assert not torch.isnan(loss), "Loss is NaN on CUDA"
