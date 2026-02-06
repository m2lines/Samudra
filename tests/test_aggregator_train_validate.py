"""Tests for train and validate aggregators.

These tests verify the core functionality of:
- TrainAggregator: loss accumulation across batches
- ValidateAggregator: composition of sub-aggregators and multi-scale support
- MeanAggregator (validate): area-weighted metric computation with multi-scale support
"""

import pytest
import torch

from ocean_emulators.aggregator.metrics import (
    area_weighted_mean_bias,
    area_weighted_rmse,
)
from ocean_emulators.aggregator.train import TrainAggregator
from ocean_emulators.aggregator.validate.reduced import (
    AreaWeightedReducedMetric,
    MeanAggregator,
)
from ocean_emulators.utils.output import TrainBatchOutput


class TestTrainAggregator:
    """Tests for TrainAggregator."""

    def test_single_batch_accumulation(self):
        """Verify that a single batch is recorded correctly."""
        agg = TrainAggregator()

        loss = torch.tensor(0.5)
        loss_per_channel = torch.tensor([0.1, 0.2, 0.3, 0.4])
        batch = TrainBatchOutput(loss=loss, loss_per_channel=loss_per_channel)

        agg.record_batch(batch)

        assert agg._n_batches == 1
        assert torch.allclose(agg._loss, loss)
        assert torch.allclose(agg._loss_per_channel, loss_per_channel)

    def test_multiple_batch_accumulation(self):
        """Verify that multiple batches are accumulated correctly."""
        agg = TrainAggregator()

        batch1 = TrainBatchOutput(
            loss=torch.tensor(0.5), loss_per_channel=torch.tensor([0.1, 0.2])
        )
        batch2 = TrainBatchOutput(
            loss=torch.tensor(0.3), loss_per_channel=torch.tensor([0.3, 0.4])
        )

        agg.record_batch(batch1)
        agg.record_batch(batch2)

        assert agg._n_batches == 2
        assert torch.allclose(agg._loss, torch.tensor(0.8))
        assert torch.allclose(agg._loss_per_channel, torch.tensor([0.4, 0.6]))

    def test_get_logs_returns_mean_loss(self, mock_tensor_map_full):
        """Verify get_logs returns the mean loss across batches."""
        agg = TrainAggregator()

        # TensorMap "thermo_dynamic_5" has 21 channels (4 vars x 5 depths + zos)
        n_channels = 21
        batch1 = TrainBatchOutput(
            loss=torch.tensor(1.0), loss_per_channel=torch.ones(n_channels) * 0.5
        )
        batch2 = TrainBatchOutput(
            loss=torch.tensor(3.0), loss_per_channel=torch.ones(n_channels) * 1.5
        )

        agg.record_batch(batch1)
        agg.record_batch(batch2)

        logs = agg.get_logs(label="train")

        # Mean loss should be (1.0 + 3.0) / 2 = 2.0
        assert logs["train/mean/loss"] == pytest.approx(2.0)

    def test_get_logs_structure(self, mock_tensor_map_full):
        """Verify get_logs returns expected keys."""
        agg = TrainAggregator()

        # TensorMap "thermo_dynamic_5" has 21 channels
        n_channels = 21
        batch = TrainBatchOutput(
            loss=torch.tensor(0.5), loss_per_channel=torch.ones(n_channels) * 0.1
        )
        agg.record_batch(batch)

        logs = agg.get_logs(label="test_label")

        # Should have mean loss key
        assert "test_label/mean/loss" in logs
        # Should have depth/variable/channel loss keys based on TensorMap
        assert any("loss/depth" in k for k in logs.keys())
        assert any("loss/variable" in k for k in logs.keys())
        assert any("loss/channel" in k for k in logs.keys())


class TestAreaWeightedReducedMetric:
    """Tests for AreaWeightedReducedMetric."""

    def test_record_accumulates_metric(self):
        """Verify that recording multiple batches accumulates the metric."""

        def mock_metric(target, gen):
            return (gen - target).abs().mean(dim=(-2, -1))

        metric = AreaWeightedReducedMetric(
            device=torch.device("cpu"), compute_metric=mock_metric
        )

        # Shape: [batch, time, height, width]
        target1 = torch.zeros(2, 1, 4, 4)
        gen1 = torch.ones(2, 1, 4, 4)

        target2 = torch.zeros(2, 1, 4, 4)
        gen2 = torch.ones(2, 1, 4, 4) * 2

        metric.record(target1, gen1)
        metric.record(target2, gen2)

        result = metric.get()
        # First batch: abs diff mean = 1.0
        # Second batch: abs diff mean = 2.0
        # Total = 1.0 + 2.0 = 3.0 (accumulated, not averaged)
        assert result.shape == (1,)  # time dimension
        assert torch.allclose(result, torch.tensor([3.0]))


class TestMeanAggregator:
    """Tests for MeanAggregator (validate/reduced.py)."""

    def test_record_batch_with_matching_target_time(self, mock_data_source):
        """Verify that batches at the target time are recorded."""
        srcs = [mock_data_source]
        target_time = 0
        agg = MeanAggregator(srcs, target_time=target_time)

        h, w = mock_data_source.grid_size
        gen_data = {"var1/8x8": torch.randn(2, 1, h, w)}
        target_data = {"var1/8x8": torch.randn(2, 1, h, w)}

        agg.record_batch(
            target_data=target_data,
            gen_data=gen_data,
            target_data_norm={},
            gen_data_norm={},
        )

        assert agg._n_batches == 1
        assert agg._variable_metrics is not None

    def test_record_batch_skips_non_matching_target_time(self, mock_data_source):
        """Verify that batches before the target time are skipped."""
        srcs = [mock_data_source]
        target_time = 5  # Data only has 1 time step, so this won't match
        agg = MeanAggregator(srcs, target_time=target_time)

        h, w = mock_data_source.grid_size
        gen_data = {"var1/8x8": torch.randn(2, 1, h, w)}
        target_data = {"var1/8x8": torch.randn(2, 1, h, w)}

        agg.record_batch(
            target_data=target_data,
            gen_data=gen_data,
            target_data_norm={},
            gen_data_norm={},
        )

        # Should not have recorded because target_time > time_len
        assert agg._n_batches == 0

    def test_multiscale_grid_selection(self, mock_data_source, mock_data_source_large):
        """Verify that the correct grid is selected for multi-scale training."""
        srcs = [mock_data_source, mock_data_source_large]
        target_time = 0
        agg = MeanAggregator(srcs, target_time=target_time)

        # Use the large grid
        h, w = mock_data_source_large.grid_size
        gen_data = {"var1/16x16": torch.randn(2, 1, h, w)}
        target_data = {"var1/16x16": torch.randn(2, 1, h, w)}

        agg.record_batch(
            target_data=target_data,
            gen_data=gen_data,
            target_data_norm={},
            gen_data_norm={},
        )

        assert agg._n_batches == 1
        # Verify metrics were created with correct grid key
        assert agg._variable_metrics is not None
        keys = list(agg._variable_metrics.keys())
        assert any("16x16" in k for k in keys)

    def test_get_logs_returns_metrics(self, mock_data_source):
        """Verify get_logs returns properly structured metrics."""
        srcs = [mock_data_source]
        target_time = 0
        agg = MeanAggregator(srcs, target_time=target_time)

        h, w = mock_data_source.grid_size
        gen_data = {"var1/8x8": torch.randn(2, 1, h, w)}
        target_data = {"var1/8x8": torch.randn(2, 1, h, w)}

        agg.record_batch(
            target_data=target_data,
            gen_data=gen_data,
            target_data_norm={},
            gen_data_norm={},
        )

        logs = agg.get_logs(label="reduced")

        # Should have RMSE, bias, and gradient metrics
        assert any("weighted_rmse" in k for k in logs.keys())
        assert any("weighted_bias" in k for k in logs.keys())
        assert any("weighted_grad_mag_percent_diff" in k for k in logs.keys())

    def test_raises_when_no_batches_recorded(self, mock_data_source):
        """Verify _get_data raises when no batches have been recorded."""
        srcs = [mock_data_source]
        agg = MeanAggregator(srcs, target_time=0)

        with pytest.raises(ValueError, match="No batches have been recorded"):
            agg._get_data()


class TestMetricFunctions:
    """Tests for area-weighted metric functions."""

    def test_area_weighted_rmse_perfect_prediction(self):
        """Verify RMSE is zero for perfect predictions."""
        target = torch.ones(2, 4, 4)
        gen = torch.ones(2, 4, 4)
        weights = torch.ones(4, 4) / 16

        rmse = area_weighted_rmse(target, gen, weights)

        assert torch.allclose(rmse, torch.zeros(2))

    def test_area_weighted_rmse_with_error(self):
        """Verify RMSE correctly measures error."""
        target = torch.zeros(1, 4, 4)
        gen = torch.ones(1, 4, 4)  # Constant error of 1
        weights = torch.ones(4, 4) / 16

        rmse = area_weighted_rmse(target, gen, weights)

        # RMSE should be 1.0 for constant error of 1
        assert torch.allclose(rmse, torch.tensor([1.0]))

    def test_area_weighted_mean_bias(self):
        """Verify bias is correctly computed as mean(gen - target)."""
        target = torch.zeros(1, 4, 4)
        gen = torch.ones(1, 4, 4) * 2  # gen - target = 2
        weights = torch.ones(4, 4) / 16

        bias = area_weighted_mean_bias(target, gen, weights)

        assert torch.allclose(bias, torch.tensor([2.0]))


# Fixtures
#
# Note: These fixtures create minimal DataSources specifically for unit testing
# aggregator logic. They use small grids (8x8, 16x16) for fast test execution
# and provide multi-scale pairs. For integration tests with realistic data,
# use the `data_source` fixture from conftest.py instead.


def _create_mock_data_source(name: str, grid_size: int):
    """Helper to create a minimal DataSource for testing.

    Args:
        name: Name for the data source
        grid_size: Size of the square grid (height and width)
    """
    import dataclasses

    import numpy as np
    import xarray as xr

    from ocean_emulators.utils.data import DataSource, Masks

    half = grid_size // 2
    lat = np.arange(-half, half, dtype=np.float64)
    lon = np.arange(0, grid_size, dtype=np.float64)
    coords = {"lat": lat, "lon": lon}

    data = xr.Dataset(
        {
            "var1": xr.DataArray(
                np.random.randn(1, grid_size, grid_size), dims=["time", "lat", "lon"]
            )
        },
        coords=coords,
    )
    means = xr.Dataset({"var1": xr.DataArray(0.0)})
    stds = xr.Dataset({"var1": xr.DataArray(1.0)})

    prognostic_mask = torch.ones(1, grid_size, grid_size)
    boundary_mask = torch.ones(grid_size, grid_size)
    masks = Masks(prognostic=prognostic_mask, boundary=boundary_mask)

    return dataclasses.replace(
        DataSource(name=name, data=data, means=means, stds=stds, masks=masks),
    )


@pytest.fixture
def mock_tensor_map_full():
    """Create a TensorMap with multiple variables for testing loss breakdown."""
    from ocean_emulators.constants import TensorMap
    from ocean_emulators.utils.multiton import MultitonScope

    with MultitonScope():
        TensorMap.init_instance("thermo_dynamic_5", "tau_hfds_hfds_anom")
        yield


@pytest.fixture
def mock_data_source():
    """Create a minimal 8x8 DataSource for testing."""
    return _create_mock_data_source("test_8x8", grid_size=8)


@pytest.fixture
def mock_data_source_large():
    """Create a larger 16x16 DataSource for multi-scale testing."""
    return _create_mock_data_source("test_16x16", grid_size=16)
