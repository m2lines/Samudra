"""Black-box integration tests for ValidateAggregator API.

These tests verify the external API behavior of the validation aggregator
without depending on internal implementation details.
"""

import pytest
import torch
import xarray as xr

from ocean_emulators.aggregator.main import Aggregator
from ocean_emulators.aggregator.validate.main import ValidateAggregator
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS, TensorMap
from ocean_emulators.utils.data import DataSource, Normalize
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.output import TrainBatchOutput


class TestValidationAggregatorAPI:
    """Test suite for the ValidateAggregator external API."""

    @pytest.fixture(autouse=True)
    def setup_multitons(self):
        """Set up required multiton instances for each test."""
        with MultitonScope() as scope:
            # Define test variable configurations
            # Use naming convention that TensorMap expects (ending with depth level count)
            PROGNOSTIC_VARS["test_1"] = ["thetao_0", "so_0", "zos"]
            BOUNDARY_VARS["test_boundary"] = ["hfds"]

            # Initialize TensorMap
            TensorMap.init_instance("test_1", "test_boundary")

            # Create minimal data source for Normalize
            lat, lon = 10, 12
            coords = {"lat": range(lat), "lon": range(lon)}

            # Create datasets with the expected variables
            data_vars = {}
            means_vars = {}
            stds_vars = {}

            # Add 3D variables
            for var in ["thetao", "so"]:
                data_vars[var] = xr.DataArray(
                    0.0, dims=["lev", "lat", "lon"], coords={"lev": [0], **coords}
                )
                means_vars[var] = xr.DataArray(
                    0.0, dims=["lev", "lat", "lon"], coords={"lev": [0], **coords}
                )
                stds_vars[var] = xr.DataArray(
                    1.0, dims=["lev", "lat", "lon"], coords={"lev": [0], **coords}
                )

            # Add 2D variables
            for var in ["zos", "hfds"]:
                data_vars[var] = xr.DataArray(0.0, dims=["lat", "lon"], coords=coords)
                means_vars[var] = xr.DataArray(0.0, dims=["lat", "lon"], coords=coords)
                stds_vars[var] = xr.DataArray(1.0, dims=["lat", "lon"], coords=coords)

            data_src = DataSource(
                name="test",
                data=xr.Dataset(data_vars),
                means=xr.Dataset(means_vars),
                stds=xr.Dataset(stds_vars),
            )

            # Initialize Normalize
            wet_mask = torch.ones((lat, lon), dtype=torch.float32)
            Normalize.init_instance(
                src=data_src,
                prognostic_var_names=tuple(PROGNOSTIC_VARS["test_1"]),
                boundary_var_names=tuple(BOUNDARY_VARS["test_boundary"]),
                wet_mask=wet_mask,
                wet_mask_surface=wet_mask,
            )

            # Store test parameters
            self.metadata = {
                "thetao_0": {"units": "K", "long_name": "Temperature"},
                "so_0": {"units": "PSU", "long_name": "Salinity"},
                "zos": {"units": "m", "long_name": "Sea Surface Height"},
            }
            self.area_weights = torch.ones((lat, lon))
            self.wet_mask = wet_mask
            self.num_prognostic_channels = len(PROGNOSTIC_VARS["test_1"])

            yield scope

    def test_initialization(self):
        """Test that ValidateAggregator initializes correctly."""
        aggregator = Aggregator.get_validation_aggregator(
            metadata=self.metadata,
            hist=0,
            area_weights=self.area_weights,
            wet=self.wet_mask,
            num_prognostic_channels=self.num_prognostic_channels,
        )

        assert isinstance(aggregator, ValidateAggregator)
        assert aggregator.hist == 0
        assert aggregator.num_prognostic_channels == self.num_prognostic_channels

    def test_loss_recording(self):
        """Test recording and retrieving loss metrics."""
        aggregator = Aggregator.get_validation_aggregator(
            metadata=self.metadata,
            hist=0,
            area_weights=self.area_weights,
            wet=self.wet_mask,
            num_prognostic_channels=self.num_prognostic_channels,
        )

        # Record multiple batches
        losses = [0.5, 0.3, 0.4]
        for loss_val in losses:
            batch = TrainBatchOutput(
                loss=torch.tensor(loss_val),
                loss_per_channel=torch.ones(self.num_prognostic_channels) * loss_val,
            )
            aggregator.record_batch(batch)

        # Get logs
        logs = aggregator.get_logs("val")

        # Verify loss metric exists and is averaged
        assert "val/mean/loss" in logs
        expected_avg = sum(losses) / len(losses)
        actual_avg = (
            logs["val/mean/loss"].item()
            if isinstance(logs["val/mean/loss"], torch.Tensor)
            else logs["val/mean/loss"]
        )
        assert abs(float(actual_avg) - expected_avg) < 0.01

    def test_get_logs_labels(self):
        """Test that get_logs respects the label parameter."""
        aggregator = Aggregator.get_validation_aggregator(
            metadata=self.metadata,
            hist=0,
            area_weights=self.area_weights,
            wet=self.wet_mask,
            num_prognostic_channels=self.num_prognostic_channels,
        )

        # Record a batch
        batch = TrainBatchOutput(
            loss=torch.tensor(0.5),
            loss_per_channel=torch.ones(self.num_prognostic_channels) * 0.5,
        )
        aggregator.record_batch(batch)

        # Get logs with different labels
        train_logs = aggregator.get_logs("train")
        val_logs = aggregator.get_logs("val")
        test_logs = aggregator.get_logs("test")

        # All should have the appropriate prefix
        assert all(k.startswith("train/") for k in train_logs.keys())
        assert all(k.startswith("val/") for k in val_logs.keys())
        assert all(k.startswith("test/") for k in test_logs.keys())

    def test_sub_aggregator_access(self):
        """Test the get_sub_aggregator method."""
        aggregator = Aggregator.get_validation_aggregator(
            metadata=self.metadata,
            hist=0,
            area_weights=self.area_weights,
            wet=self.wet_mask,
            num_prognostic_channels=self.num_prognostic_channels,
        )

        # Test accessing existing sub-aggregators
        assert aggregator.get_sub_aggregator("snapshot") is not None
        assert aggregator.get_sub_aggregator("mean_map") is not None
        assert aggregator.get_sub_aggregator("reduced") is not None

        # Test accessing non-existent sub-aggregator
        assert aggregator.get_sub_aggregator("nonexistent") is None

    def test_multiple_aggregator_instances(self):
        """Test that multiple aggregator instances can coexist."""
        # Create first aggregator
        agg1 = Aggregator.get_validation_aggregator(
            metadata=self.metadata,
            hist=0,
            area_weights=self.area_weights,
            wet=self.wet_mask,
            num_prognostic_channels=self.num_prognostic_channels,
        )

        # Create second aggregator with different parameters
        agg2 = Aggregator.get_validation_aggregator(
            metadata=self.metadata,
            hist=0,
            area_weights=self.area_weights * 2,  # Different weights
            wet=self.wet_mask,
            num_prognostic_channels=self.num_prognostic_channels,
        )

        # Record different losses in each
        agg1.record_batch(
            TrainBatchOutput(
                loss=torch.tensor(0.5),
                loss_per_channel=torch.ones(self.num_prognostic_channels) * 0.5,
            )
        )

        agg2.record_batch(
            TrainBatchOutput(
                loss=torch.tensor(0.8),
                loss_per_channel=torch.ones(self.num_prognostic_channels) * 0.8,
            )
        )

        # Verify they maintain separate state
        logs1 = agg1.get_logs("val")
        logs2 = agg2.get_logs("val")

        assert logs1["val/mean/loss"] != logs2["val/mean/loss"]
