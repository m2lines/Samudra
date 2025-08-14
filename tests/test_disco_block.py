import pytest
import torch

from ocean_emulators.models.modules.blocks import DiscoBlock
from ocean_emulators.models.modules.factory import create_block


class TestDiscoBlock:
    """Test suite for DiscoBlock implementation."""

    def test_disco_block_creation(self):
        """Test that DiscoBlock can be created with default parameters."""
        block = DiscoBlock(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            grid_shape=(18, 36),  # Use smaller test grid for faster testing
        )

        assert block.grid_shape == (18, 36)
        assert block.disco_filter_type == "piecewise linear"
        assert hasattr(block, "layers")
        assert hasattr(block, "skip_module")

    def test_disco_block_different_channels(self):
        """Test DiscoBlock with different input/output channels."""
        block = DiscoBlock(
            in_channels=8, out_channels=16, kernel_size=3, grid_shape=(18, 36)
        )

        # Should create a skip module for channel adjustment
        assert not callable(block.skip_module) or hasattr(block.skip_module, "forward")

    def test_disco_block_factory_creation(self):
        """Test that DiscoBlock can be created via factory."""
        block = create_block(
            "disco_block",
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            grid_shape=(18, 36),
        )

        assert isinstance(block, DiscoBlock)
        assert block.grid_shape == (18, 36)

    @pytest.mark.parametrize(
        "in_channels,out_channels",
        [
            (8, 16),
            (16, 24),
            (24, 32),
            (16, 16),  # Same channels
        ],
    )
    def test_disco_block_forward_pass(self, in_channels, out_channels):
        """Test forward pass with various channel configurations."""
        block = DiscoBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            grid_shape=(18, 36),
        )

        # Create test input
        batch_size = 2
        x = torch.randn(batch_size, in_channels, 18, 36)

        # Forward pass
        with torch.no_grad():
            output = block(x)

        # Check output shape
        expected_shape = (batch_size, out_channels, 18, 36)
        assert output.shape == expected_shape

    def test_disco_block_forward_preserves_spatial_dims(self):
        """Test that forward pass preserves spatial dimensions."""
        grid_shapes = [(9, 18), (18, 36), (36, 72)]

        for grid_shape in grid_shapes:
            block = DiscoBlock(
                in_channels=16, out_channels=16, kernel_size=3, grid_shape=grid_shape
            )

            x = torch.randn(1, 16, *grid_shape)

            with torch.no_grad():
                output = block(x)

            assert output.shape == x.shape

    @pytest.mark.parametrize("norm_type", ["batch", "instance", "layer", "nonorm"])
    def test_disco_block_normalization_types(self, norm_type):
        """Test different normalization types."""
        block = DiscoBlock(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            norm=norm_type,
            grid_shape=(18, 36),
        )

        x = torch.randn(2, 16, 18, 36)

        with torch.no_grad():
            output = block(x)

        assert output.shape == x.shape

    @pytest.mark.parametrize("filter_type", ["piecewise linear", "morlet", "zernike"])
    def test_disco_block_filter_types(self, filter_type):
        """Test different DISCO filter basis types."""
        block = DiscoBlock(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            disco_filter_type=filter_type,
            grid_shape=(18, 36),
        )

        x = torch.randn(1, 16, 18, 36)

        with torch.no_grad():
            output = block(x)

        assert output.shape == x.shape
        assert block.disco_filter_type == filter_type

    def test_disco_block_gradient_flow(self):
        """Test that gradients flow properly through DiscoBlock."""
        block = DiscoBlock(
            in_channels=16, out_channels=16, kernel_size=3, grid_shape=(18, 36)
        )

        x = torch.randn(1, 16, 18, 36, requires_grad=True)
        output = block(x)

        # Compute dummy loss and backward pass
        loss = output.mean()
        loss.backward()

        # Check that input has gradients
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_disco_block_checkpointing(self):
        """Test checkpointing functionality."""
        block = DiscoBlock(
            in_channels=16,
            out_channels=16,
            kernel_size=3,
            checkpoint_simple=True,
            grid_shape=(18, 36),
        )

        x = torch.randn(1, 16, 18, 36)

        with torch.no_grad():
            output = block(x)

        assert output.shape == x.shape

    def test_disco_block_invalid_norm_type(self):
        """Test that invalid normalization type raises error."""
        with pytest.raises(NotImplementedError):
            DiscoBlock(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                norm="invalid_norm",
                grid_shape=(18, 36),
            )

    def test_disco_block_multiple_layers_assertion(self):
        """Test that n_layers > 1 raises assertion error."""
        with pytest.raises(AssertionError):
            DiscoBlock(
                in_channels=16,
                out_channels=16,
                kernel_size=3,
                n_layers=2,  # Should trigger assertion
                grid_shape=(18, 36),
            )
