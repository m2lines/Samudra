"""Unit tests for ScheduledDepthDropout and related dropout functionality."""

import logging

import pytest
import torch

from ocean_emulators.config import StochasticDepthConfig
from ocean_emulators.models.modules.activations import CappedGELU
from ocean_emulators.models.modules.dropout import ScheduledDepthDropout
from ocean_emulators.models.modules.factory import create_block

logger = logging.getLogger(__name__)


class TestScheduledDepthDropout:
    """Comprehensive unit tests for ScheduledDepthDropout class."""

    @pytest.mark.parametrize("schedule", ["early_only", "late_only", "constant"])
    def test_valid_schedule_types(self, schedule):
        """Test all valid schedule types."""
        dropout = ScheduledDepthDropout(schedule=schedule)
        assert dropout.schedule == schedule

    def test_set_epoch_functionality(self):
        """Test epoch setting and retrieval."""
        dropout = ScheduledDepthDropout()

        # Test setting various epochs
        dropout.set_epoch(10)
        assert dropout.current_epoch == 10

        dropout.set_epoch(0)
        assert dropout.current_epoch == 0

        dropout.set_epoch(100)
        assert dropout.current_epoch == 100

    def test_initial_epoch_state(self):
        """Test that initial epoch is 0."""
        dropout = ScheduledDepthDropout()
        assert dropout.current_epoch == 0

    def test_epoch_bounds(self):
        """Test edge cases for epoch values."""
        dropout = ScheduledDepthDropout()

        # Test negative epoch (should be handled gracefully)
        dropout.set_epoch(-1)
        assert dropout.current_epoch == -1

        # Test large epoch
        dropout.set_epoch(10000)
        assert dropout.current_epoch == 10000


class TestEarlyOnlySchedule:
    """Tests for early_only dropout schedule."""

    def test_disabled_dropout_zero_rate(self):
        """Test that zero drop rate always returns 0.0."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.0, early_epochs=30, schedule="early_only"
        )

        for epoch in [0, 10, 20, 30, 50]:
            dropout.set_epoch(epoch)
            assert dropout.get_current_drop_prob() == 0.0

    def test_disabled_dropout_zero_epochs(self):
        """Test that zero epochs always returns 0.0."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.2, early_epochs=0, schedule="early_only"
        )

        for epoch in [0, 10, 20, 30]:
            dropout.set_epoch(epoch)
            assert dropout.get_current_drop_prob() == 0.0

    def test_linear_decay_calculation(self):
        """Test linear decay formula accuracy."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.2, early_epochs=20, schedule="early_only", linear_decay=True
        )

        # Test specific points
        dropout.set_epoch(0)
        assert dropout.get_current_drop_prob() == 0.2  # Full rate at start

        dropout.set_epoch(10)  # Halfway through
        assert dropout.get_current_drop_prob() == 0.1  # Half rate

        dropout.set_epoch(19)  # Almost at end
        expected = 0.2 * (1.0 - 19 / 20)
        assert abs(dropout.get_current_drop_prob() - expected) < 1e-6

    def test_after_early_period_returns_zero(self):
        """Test that dropout is disabled after early period."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.3, early_epochs=25, schedule="early_only"
        )

        # Test at and after boundary
        dropout.set_epoch(25)
        assert dropout.get_current_drop_prob() == 0.0

        dropout.set_epoch(26)
        assert dropout.get_current_drop_prob() == 0.0

        dropout.set_epoch(100)
        assert dropout.get_current_drop_prob() == 0.0

    def test_constant_rate_during_early_period(self):
        """Test constant dropout rate when linear_decay=False."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.25, early_epochs=30, schedule="early_only", linear_decay=False
        )

        # Test constant rate throughout early period
        for epoch in [0, 5, 15, 29]:
            dropout.set_epoch(epoch)
            assert dropout.get_current_drop_prob() == 0.25

    def test_edge_cases_boundary_epochs(self):
        """Test edge cases around epoch boundaries."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.1, early_epochs=10, schedule="early_only", linear_decay=True
        )

        # Test exactly at boundaries
        dropout.set_epoch(0)
        assert dropout.get_current_drop_prob() == 0.1

        dropout.set_epoch(9)  # Last epoch of early period
        expected = 0.1 * (1.0 - 9 / 10)
        assert abs(dropout.get_current_drop_prob() - expected) < 1e-6

        dropout.set_epoch(10)  # First epoch after early period
        assert dropout.get_current_drop_prob() == 0.0


class TestLateOnlySchedule:
    """Tests for late_only dropout schedule."""

    def test_before_early_period_returns_zero(self):
        """Test that no dropout occurs before early period."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.2, early_epochs=20, schedule="late_only"
        )

        for epoch in [0, 5, 10, 19]:
            dropout.set_epoch(epoch)
            assert dropout.get_current_drop_prob() == 0.0

    def test_after_early_period_returns_base_rate(self):
        """Test that base rate is used after early period."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.15, early_epochs=15, schedule="late_only"
        )

        for epoch in [15, 20, 50, 100]:
            dropout.set_epoch(epoch)
            assert dropout.get_current_drop_prob() == 0.15

    def test_linear_decay_ignored_in_late_schedule(self):
        """Test that linear_decay setting is ignored for late_only."""
        dropout_linear = ScheduledDepthDropout(
            drop_prob=0.3, early_epochs=10, schedule="late_only", linear_decay=True
        )

        dropout_constant = ScheduledDepthDropout(
            drop_prob=0.3, early_epochs=10, schedule="late_only", linear_decay=False
        )

        # Both should behave identically
        for epoch in [0, 5, 10, 15]:
            dropout_linear.set_epoch(epoch)
            dropout_constant.set_epoch(epoch)
            assert (
                dropout_linear.get_current_drop_prob()
                == dropout_constant.get_current_drop_prob()
            )

    def test_transition_boundary(self):
        """Test exact transition boundary behavior."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.2, early_epochs=25, schedule="late_only"
        )

        dropout.set_epoch(24)  # Last epoch before activation
        assert dropout.get_current_drop_prob() == 0.0

        dropout.set_epoch(25)  # First epoch of activation
        assert dropout.get_current_drop_prob() == 0.2


class TestConstantSchedule:
    """Tests for constant dropout schedule."""

    def test_constant_rate_all_epochs(self):
        """Test that drop rate is constant across all epochs."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.12, early_epochs=20, schedule="constant"
        )

        for epoch in [0, 10, 20, 30, 100]:
            dropout.set_epoch(epoch)
            assert dropout.get_current_drop_prob() == 0.12

    def test_ignores_linear_decay_setting(self):
        """Test that linear_decay is ignored for constant schedule."""
        dropout_linear = ScheduledDepthDropout(
            drop_prob=0.25, early_epochs=30, schedule="constant", linear_decay=True
        )

        dropout_constant = ScheduledDepthDropout(
            drop_prob=0.25, early_epochs=30, schedule="constant", linear_decay=False
        )

        for epoch in [0, 15, 30, 45]:
            dropout_linear.set_epoch(epoch)
            dropout_constant.set_epoch(epoch)
            assert (
                dropout_linear.get_current_drop_prob()
                == dropout_constant.get_current_drop_prob()
                == 0.25
            )

    def test_ignores_early_epochs_setting(self):
        """Test that early_epochs doesn't affect constant schedule."""
        dropout1 = ScheduledDepthDropout(
            drop_prob=0.1, early_epochs=10, schedule="constant"
        )

        dropout2 = ScheduledDepthDropout(
            drop_prob=0.1, early_epochs=50, schedule="constant"
        )

        for epoch in [0, 25, 60]:
            dropout1.set_epoch(epoch)
            dropout2.set_epoch(epoch)
            assert (
                dropout1.get_current_drop_prob()
                == dropout2.get_current_drop_prob()
                == 0.1
            )


class TestForwardPass:
    """Tests for forward pass behavior."""

    def test_eval_mode_passthrough(self):
        """Test that eval mode passes input unchanged."""
        dropout = ScheduledDepthDropout(drop_prob=0.5, early_epochs=10)
        dropout.eval()  # Set to evaluation mode
        dropout.set_epoch(5)  # Should have high drop probability

        x = torch.randn(4, 8, 16, 16)
        output = dropout(x)

        # In eval mode, output should equal input exactly
        assert torch.equal(x, output)

    def test_training_mode_shape_preservation(self):
        """Test that output shape matches input shape."""
        dropout = ScheduledDepthDropout(drop_prob=0.3, early_epochs=20)
        dropout.train()
        dropout.set_epoch(10)

        # Test various tensor shapes
        shapes = [(4, 8), (2, 16, 32), (1, 64, 28, 28), (8, 128, 14, 14, 7)]

        for shape in shapes:
            x = torch.randn(*shape)
            output = dropout(x)
            assert output.shape == x.shape

    def test_dtype_and_device_preservation(self):
        """Test that tensor dtype and device are preserved."""
        dropout = ScheduledDepthDropout(drop_prob=0.2, early_epochs=15)
        dropout.train()
        dropout.set_epoch(5)

        # Test different dtypes
        for dtype in [torch.float32, torch.float64]:
            x = torch.randn(4, 8, dtype=dtype)
            output = dropout(x)
            assert output.dtype == dtype

        # Test GPU if available
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            x = torch.randn(4, 8, device=device)
            output = dropout(x)
            assert output.device == device

    def test_zero_dropout_probability_passthrough(self):
        """Test that zero drop probability passes input unchanged."""
        dropout = ScheduledDepthDropout(drop_prob=0.1, early_epochs=10)
        dropout.train()
        dropout.set_epoch(10)  # After early period, should be 0 drop prob

        x = torch.randn(4, 16, 8, 8)
        output = dropout(x)

        # Should pass through unchanged when drop_prob = 0
        assert torch.equal(x, output)

    @pytest.mark.parametrize("drop_prob", [0.1, 0.3, 0.5, 0.7, 0.9])
    def test_stochastic_behavior_statistics(self, drop_prob):
        """Test that dropout probability is approximately respected over many samples."""
        dropout = ScheduledDepthDropout(
            drop_prob=drop_prob, early_epochs=100, schedule="constant"
        )
        dropout.train()
        dropout.set_epoch(50)

        # Run many forward passes and count zeros
        num_trials = 1000
        zero_count = 0

        torch.manual_seed(42)  # For reproducible results

        for _ in range(num_trials):
            x = torch.ones(1, 1, 1, 1)  # Single element tensor
            output = dropout(x)
            if torch.allclose(output, torch.zeros_like(output)):
                zero_count += 1

        observed_drop_rate = zero_count / num_trials

        # Allow for statistical variance (±10% tolerance)
        expected_drop_rate = drop_prob
        tolerance = 0.1

        assert abs(observed_drop_rate - expected_drop_rate) < tolerance

    def test_mathematical_correctness_expectation(self):
        """Test that expectation is preserved: E[output] ≈ input."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.3, early_epochs=100, schedule="constant"
        )
        dropout.train()
        dropout.set_epoch(50)

        # Fixed input
        x = torch.ones(1, 1, 1, 1) * 5.0

        # Run many trials
        num_trials = 10000
        total = 0.0

        torch.manual_seed(123)

        for _ in range(num_trials):
            output = dropout(x)
            total += output.item()

        mean_output = total / num_trials
        expected_output = x.item()  # Should preserve expectation

        # Allow for statistical variance
        tolerance = 0.3
        assert abs(mean_output - expected_output) < tolerance


class TestIntegrationAndEdgeCases:
    """Integration tests and edge cases."""

    def test_schedule_transitions_smooth(self):
        """Test smooth transitions between schedule phases."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.2, early_epochs=10, schedule="early_only", linear_decay=True
        )

        # Test that probabilities decrease smoothly
        previous_prob = float("inf")
        for epoch in range(15):
            dropout.set_epoch(epoch)
            current_prob = dropout.get_current_drop_prob()

            if epoch < 10:  # During early period
                assert current_prob <= previous_prob  # Should decrease or stay same
                assert current_prob >= 0.0  # Should be non-negative
            else:  # After early period
                assert current_prob == 0.0

            previous_prob = current_prob

    def test_multiple_epoch_updates_consistency(self):
        """Test that multiple epoch updates work consistently."""
        dropout = ScheduledDepthDropout(
            drop_prob=0.3, early_epochs=20, schedule="early_only", linear_decay=True
        )

        # Test setting and resetting epochs
        dropout.set_epoch(5)
        prob1 = dropout.get_current_drop_prob()

        dropout.set_epoch(10)
        dropout.set_epoch(5)  # Reset to previous epoch
        prob2 = dropout.get_current_drop_prob()

        assert prob1 == prob2  # Should be consistent

    @pytest.mark.parametrize("seed", [42, 123, 456])
    def test_reproducibility_with_seeds(self, seed):
        """Test reproducible behavior with fixed seeds."""
        dropout = ScheduledDepthDropout(drop_prob=0.5, early_epochs=20)
        dropout.train()
        dropout.set_epoch(10)

        x = torch.randn(4, 8)

        # First run
        torch.manual_seed(seed)
        output1 = dropout(x)

        # Second run with same seed
        torch.manual_seed(seed)
        output2 = dropout(x)

        assert torch.equal(output1, output2)

    def test_extreme_drop_probabilities(self):
        """Test behavior with extreme drop probabilities."""
        # Test near-zero probability
        dropout_low = ScheduledDepthDropout(
            drop_prob=1e-6, early_epochs=10, schedule="constant"
        )
        dropout_low.train()
        dropout_low.set_epoch(5)

        x = torch.ones(100, 1)
        output = dropout_low(x)
        # Should almost always pass through
        zeros_mask = torch.isclose(output, torch.zeros_like(output))
        zero_count = zeros_mask.sum().item()
        assert zero_count <= 5  # Very few zeros expected

        # Test near-maximum probability
        dropout_high = ScheduledDepthDropout(
            drop_prob=0.999, early_epochs=10, schedule="constant"
        )
        dropout_high.train()
        dropout_high.set_epoch(5)

        # Most outputs should be zero (or scaled up significantly)
        outputs_nonzero = 0
        for _ in range(100):
            out = dropout_high(torch.ones(1, 1))
            if not torch.allclose(out, torch.zeros_like(out)):
                outputs_nonzero += 1

        # Should have very few non-zero outputs
        assert outputs_nonzero <= 5


def test_dropout_integration():
    """Test that dropout system integrates properly with model blocks."""

    logger.info("Testing dropout integration with ConvNext blocks...")

    # Create a stochastic depth config
    dropout_config = StochasticDepthConfig(
        drop_path_rate=0.2,
        early_dropout_epochs=10,
        dropout_schedule="early_only",
        linear_decay_to_zero=True,
    )

    # Build the dropout manager
    dropout_manager = dropout_config.build()
    logger.info(
        f"Created dropout manager with drop rate: {dropout_config.drop_path_rate}"
    )

    # Create a ConvNext block using the factory
    block = create_block(
        block_type="conv_next_block",
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        dilation=1,
        activation=CappedGELU,
        upscale_factor=4,
        norm="batch",
        dropout_manager=dropout_manager,
        layer_index=0,
    )

    logger.info(f"Created block: {type(block).__name__}")

    # Test that dropout module is attached
    assert hasattr(block, "drop_path"), "Block should have drop_path attribute"
    if block.drop_path is not None:
        logger.info(f"Block has dropout module: {type(block.drop_path).__name__}")
    else:
        logger.info("Block has no dropout (drop_path is None)")

    # Test epoch management
    if block.drop_path is not None:
        # Test early epochs (should have dropout)
        dropout_manager.update_epoch(5)  # Middle of early period
        prob_early = block.drop_path.get_current_drop_prob()
        logger.info(f"Dropout probability at epoch 5: {prob_early:.3f}")
        assert prob_early > 0, "Expected non-zero dropout during early epochs"

        # Test late epochs (should have no dropout)
        dropout_manager.update_epoch(15)  # After early period
        prob_late = block.drop_path.get_current_drop_prob()
        logger.info(f"Dropout probability at epoch 15: {prob_late:.3f}")
        assert prob_late == 0.0, "Expected zero dropout after early period"

    # Test forward pass
    block.train()
    dummy_input = torch.randn(2, 64, 32, 32)  # batch, channels, height, width

    # Test training mode
    output = block(dummy_input)
    logger.info(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, f"Expected same shape, got {output.shape}"

    # Test eval mode
    block.eval()
    output_eval = block(dummy_input)
    assert output_eval.shape == output.shape, (
        "Output shapes should match between train and eval modes"
    )

    # Test that eval mode passes through unchanged when dropout is disabled
    if block.drop_path is not None:
        # Set to early epoch with dropout
        dropout_manager.update_epoch(5)
        block.eval()  # Eval mode should pass through regardless of dropout probability

        # Multiple passes should be identical in eval mode
        output1 = block(dummy_input)
        output2 = block(dummy_input)
        assert torch.equal(output1, output2), "Eval mode should be deterministic"
        logger.info("✓ Eval mode is deterministic")

    logger.info("✓ All dropout integration tests passed!")
