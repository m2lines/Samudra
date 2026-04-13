import numpy as np
import pytest
import xarray as xr

from ocean_emulators.constants import LoaderVersion
from ocean_emulators.rust_loader import TideDatasetHandle
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope
from tests.conftest import DEFAULT_CONFIG


def _with_rust_loader(train_config):
    return train_config.model_copy(
        update={
            "data": train_config.data.model_copy(
                update={
                    "loader_version": str(LoaderVersion.OM4_RUST_V0.value),
                    "num_workers": 0,
                }
            )
        }
    )


def test_tide_time_index_maps_sliced_times_to_backing_store(tmp_path):
    times = np.arange("1975-01-01", "1975-01-08", dtype="datetime64[D]")
    path = tmp_path / "data.zarr"
    xr.Dataset(
        {"sample": ("time", np.arange(times.size, dtype=np.float32))},
        coords={"time": times},
    ).to_zarr(path)

    mapped = TideDatasetHandle._build_time_index(str(path), times[2:5])

    assert np.array_equal(mapped, np.asarray([2, 3, 4], dtype=np.int64))


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", DEFAULT_CONFIG)],
    indirect=True,
)
@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
def test_tide_torch_batch_api_is_disabled(train_config):
    pytest.importorskip("tide")
    rust_config = _with_rust_loader(train_config)

    with MultitonScope():
        trainer = Trainer(rust_config)
        trainer.init_data_loaders(cur_step=rust_config.steps[0])
        batch = trainer.train_loader[0]

        with pytest.raises(NotImplementedError, match="JAX frontend"):
            batch.get_input(0)


@pytest.mark.parametrize(
    "data_source,config_name",
    [("mock-om4", DEFAULT_CONFIG)],
    indirect=True,
)
@pytest.mark.parametrize("backend", ["cpu"], indirect=True)
def test_tide_raw_batch_returns_full_spatial_window(train_config):
    pytest.importorskip("tide")
    rust_config = _with_rust_loader(train_config)

    with MultitonScope():
        trainer = Trainer(rust_config)
        trainer.init_data_loaders(cur_step=rust_config.steps[0])
        batch = trainer.train_loader[0]
        prognostic, boundary, label = batch.get_raw_step0_parts()

    assert prognostic.shape[-2:] == batch.prognostic_mask.shape[-2:]
    assert boundary.shape[-2:] == batch.boundary_mask.shape[-2:]
    assert label.shape[-2:] == batch.prognostic_mask.shape[-2:]
