import math

import numpy as np
import pytest
import torch
import xarray as xr
from scipy.stats import pearsonr

from ocean_emulators.constants import (
    BOUNDARY_VARS,
    DEPTH_I_LEVELS,
    DEPTH_LEVELS,
    PROGNOSTIC_VARS,
    TensorMap,
)
from ocean_emulators.utils.data import (
    DataSource,
    Masks,
    Normalize,
    _slice_llc_region,
    compute_anomalies,
    flatten_masks,
    get_aggregator_dicts,
    unflatten_masks,
    with_level_index_vars,
)
from ocean_emulators.utils.multiton import MultitonScope


def test_slice_llc_region_supports_channel_xy_patch_layout():
    data = xr.Dataset(
        {
            "prognostic": (
                ("time", "channel", "y", "x"),
                np.zeros((1, 2, 720, 720)),
            )
        },
        coords={
            "time": [0],
            "channel": [0, 1],
            "x": np.arange(720),
            "y": np.arange(720),
        },
    )

    sliced = _slice_llc_region(
        data,
        llc_face=1,
        llc_i_start=0,
        llc_i_end=320,
        llc_j_start=0,
        llc_j_end=320,
    )

    assert sliced.sizes["x"] == 320
    assert sliced.sizes["y"] == 320


def test_mask_roundtrip(data_source):
    data = data_source.data

    unflattened = unflatten_masks(data.copy())
    flattened = flatten_masks(unflattened.copy())

    assert flattened == data, "Assume a safe roundtrip"


def test_rename_vars():
    """Test renaming variables from OM4 format to standard format."""
    # Create test dataset with OM4 format variables
    test_data = {
        "so_lev_1050_0": (["time", "lat", "lon"], [[[1.0]]]),
        "thetao_lev_2_5": (["time", "lat", "lon"], [[[2.0]]]),
        "vo_lev_10_0": (["time", "lat", "lon"], [[[3.0]]]),
        "zos": (["time", "lat", "lon"], [[[4.0]]]),  # Should remain unchanged
    }
    ds = xr.Dataset(
        test_data,
        coords={
            "time": [0],
            "lat": [0],
            "lon": [0],
        },
    )

    # Apply rename_vars
    renamed_ds = with_level_index_vars(ds)

    # Test that variables are renamed correctly
    assert "so_11" in renamed_ds.variables  # 1040.0 is at index 11 in DEPTH_LEVELS
    assert "thetao_0" in renamed_ds.variables  # 2.5 is at index 0 in DEPTH_LEVELS
    assert "vo_1" in renamed_ds.variables  # 10.0 is at index 1 in DEPTH_LEVELS
    assert "zos" in renamed_ds.variables  # Should remain unchanged

    # Test that data values are preserved
    assert renamed_ds["so_11"].values[0, 0, 0] == 1.0
    assert renamed_ds["thetao_0"].values[0, 0, 0] == 2.0
    assert renamed_ds["vo_1"].values[0, 0, 0] == 3.0
    assert renamed_ds["zos"].values[0, 0, 0] == 4.0

    # Test that original dataset is not modified
    assert "so_lev_1050_0" in ds.variables
    assert "thetao_lev_2_5" in ds.variables
    assert "vo_lev_10_0" in ds.variables


def test_rename_vars_invalid_depth():
    """Test that invalid depth levels raise an error."""
    # Create test dataset with invalid depth level
    test_data = {
        "so_lev_9999_0": (["time", "lat", "lon"], [[[1.0]]]),  # Invalid depth
    }
    ds = xr.Dataset(
        test_data,
        coords={
            "time": [0],
            "lat": [0],
            "lon": [0],
        },
    )

    # Should raise ValueError because 9999.0 is not in DEPTH_LEVELS
    with pytest.raises(ValueError):
        with_level_index_vars(ds)


def test_compute_anomalies():
    """Test the compute_anomalies function."""
    # Create test dataset with OM4 format variables
    daterange = xr.cftime_range(
        "2000-08-05", "2010-12-31", freq="5D", calendar="julian"
    )
    N = len(daterange)

    clim = np.sin(np.linspace(-20 * np.pi, 20 * np.pi, N))
    true_anomaly = np.random.normal(0, 1, N)
    test_data = {
        "thetao_0": (
            ["lat", "lon", "time"],
            [[[clim[t] + true_anomaly[t] + 10 for t in range(N)]]],
        ),
    }

    ds = xr.Dataset(
        test_data,
        coords={
            "time": daterange,
            "lat": [0],
            "lon": [0],
        },
    )
    ds_mean = ds.mean().compute()
    ds_std = ds.std().compute()

    # compute anomalies
    anomalies, _, _ = compute_anomalies(ds, ds_mean, ds_std, ("thetao_0_anomalies",))
    anomalies_np = anomalies["thetao_0_anomalies"].to_numpy()
    anomalies_np_flat = anomalies_np[0][0]

    # check that anomalies are more correlated with true anomaly than climatology
    assert (
        pearsonr(anomalies_np_flat, true_anomaly)[0]
        > pearsonr(anomalies_np_flat, clim)[0]
    )


@pytest.fixture
def normalize_input():
    # Create test data with mean and std
    data_mean = xr.Dataset(
        {
            "var_0": 1.0,
            "var_1": 2.0,
            "var_2": 3.0,
        },
        coords={"lat": [0], "lon": [0]},
    )
    data_std = xr.Dataset(
        {
            "var_0": 0.5,
            "var_1": 1.0,
            "var_2": 2.0,
        },
        coords={"lat": [0], "lon": [0]},
    )

    # Create test wet mask
    wet_mask = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    masks = Masks(prognostic=wet_mask, boundary=wet_mask)

    # Warning: the 'data' field is not used because this test tries to test
    # normalization which only needs mean and std. Thus, we set it to `data_mean`.
    test = DataSource("test", data_mean, data_mean, data_std, masks=masks)

    # Initialize Normalize instance
    with MultitonScope():
        normalize = Normalize.init_instance(
            test,
            prognostic_var_names=["var_0", "var_1"],
            boundary_var_names=["var_2"],
        )
        yield normalize, wet_mask


def test_normalize_unnormalize_tensor_prognostic(normalize_input):
    normalize, wet_mask = normalize_input
    data = torch.randn([1, normalize._prognostic_std_np.shape[0], *wet_mask.shape])
    input_data = data * wet_mask
    normalized = normalize.normalize_tensor_prognostic(input_data)
    unnormalized = normalize.unnormalize_tensor_prognostic(normalized, fill_value=0.0)
    assert torch.allclose(input_data, unnormalized)


def test_normalize_compact_llc_stats_do_not_broadcast_surface_vars():
    levels = len(DEPTH_I_LEVELS)
    time = np.arange(4)
    lat = [0.0]
    lon = [0.0]
    lev = np.asarray(DEPTH_LEVELS, dtype=np.float32)

    def make_3d(offset: float) -> np.ndarray:
        base = np.arange(time.size * levels, dtype=np.float32).reshape(
            time.size, levels, 1, 1
        )
        return base + offset

    compact_data = xr.Dataset(
        {
            "U": (["time", "lev", "lat", "lon"], make_3d(0.0)),
            "V": (["time", "lev", "lat", "lon"], make_3d(1000.0)),
            "Theta": (["time", "lev", "lat", "lon"], make_3d(2000.0)),
            "Salt": (["time", "lev", "lat", "lon"], make_3d(3000.0)),
            "Eta": (
                ["time", "lat", "lon"],
                np.arange(time.size, dtype=np.float32).reshape(time.size, 1, 1),
            ),
            "oceTAUX": (
                ["time", "lat", "lon"],
                (10.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "oceTAUY": (
                ["time", "lat", "lon"],
                (20.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "oceQnet": (
                ["time", "lat", "lon"],
                (30.0 + np.arange(time.size, dtype=np.float32)).reshape(time.size, 1, 1),
            ),
            "wetmask": (
                ["lev", "lat", "lon"],
                np.ones((levels, 1, 1), dtype=bool),
            ),
        },
        coords={"time": time, "lev": lev, "lat": lat, "lon": lon},
    )
    compact_means = compact_data.drop_vars("wetmask").mean("time", keep_attrs=True)
    compact_stds = xr.zeros_like(compact_means, dtype=np.float32) + 1.0

    prognostic = PROGNOSTIC_VARS["all"]
    boundary = BOUNDARY_VARS["all"]
    src = DataSource.from_datasets(
        compact_data,
        compact_means,
        compact_stds,
        name="compact_llc_like",
        prognostic_var_names=prognostic,
        boundary_var_names=boundary,
    )

    with MultitonScope():
        TensorMap.init_instance("all", "all")
        normalize = Normalize.init_instance(
            src,
            prognostic_var_names=prognostic,
            boundary_var_names=boundary,
        )
        assert normalize._prognostic_mean_np.shape[0] == len(prognostic)
        assert normalize._boundary_mean_np.shape[0] == len(boundary)

        sample = torch.randn(1, 2, len(prognostic), 1, 1)
        unnormalized = normalize.unnormalize_tensor_prognostic(
            sample, fill_value=0.0
        )
        assert unnormalized.shape == sample.shape

        flat_sample = torch.randn(1, 2 * len(prognostic), 1, 1)
        _, unnorm_dict = get_aggregator_dicts(
            flat_sample,
            src.masks.prognostic,
            long_rollout=False,
            input_type="prognostic",
            num_prognostic_channels=2 * len(prognostic),
            hist=1,
        )
        assert unnorm_dict["Eta"].shape == (1, 2, 1, 1)


@pytest.mark.parametrize("fill_value", [float("nan"), 0.0])
def test_unnormalize_prognostic_tensor(normalize_input, fill_value):
    normalize, wet_mask = normalize_input
    data = torch.randn([1, normalize._prognostic_std_np.shape[0], *wet_mask.shape])
    input_data = data * wet_mask
    normalized = normalize.normalize_tensor_prognostic(input_data)
    unnormalized = normalize.unnormalize_tensor_prognostic(normalized, fill_value)
    assert (torch.sum(torch.isnan(unnormalized)) > 0) == (math.isnan(fill_value))


@pytest.fixture
def data_init(hist: int):
    with MultitonScope():
        levels = 19
        lats = 3
        lons = 3
        total_time_steps = 100

        tensor_map = TensorMap.init_instance("thetao_1", "hfds")

        wet_mask_ = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        wet_full = np.tile(wet_mask_, (total_time_steps, levels, 1, 1))

        # Even thetao, odd hfds for every time step
        # Ex, timestep 0: thetao = 0, hfds = 1
        # Ex, timestep 1: thetao = 2, hfds = 3
        # Ex, timestep 2: thetao = 4, hfds = 5
        # ...
        data = xr.Dataset(
            {
                **{
                    f"thetao_{lev}": (
                        ["time", "lat", "lon"],
                        np.tile(
                            np.arange(total_time_steps)[:, None, None] * 2,
                            (1, lats, lons),
                        ),
                    )
                    for lev in range(levels)
                },
                "hfds": (
                    ["time", "lat", "lon"],
                    np.tile(
                        np.arange(total_time_steps)[:, None, None] * 2 + 1,
                        (1, lats, lons),
                    ),
                ),
                "wetmask": (
                    ["time", "lev", "lat", "lon"],
                    wet_full,
                ),
            },
            coords={
                "time": np.arange(total_time_steps),
                "lev": DEPTH_LEVELS,
                "lat": np.arange(lats),
                "lon": np.arange(lons),
            },
        )
        data_mean = data.mean() * 0.0
        data_std = data.std() * 0.0 + 1.0
        val = DataSource.from_datasets(
            data,
            data_mean,
            data_std,
            name="test",
            prognostic_var_names=tensor_map.prognostic_var_names,
            boundary_var_names=tensor_map.boundary_var_names,
        )

        normalize = Normalize.init_instance(
            val,
            prognostic_var_names=tensor_map.prognostic_var_names,
            boundary_var_names=tensor_map.boundary_var_names,
        )
        yield normalize, val.masks.prognostic


@pytest.mark.parametrize("input_type", ["input", "target"])
@pytest.mark.parametrize("long_rollout", [True, False])
@pytest.mark.parametrize("hist", [0, 1, 2])
def test_get_norm_unnorm_dicts(data_init, input_type, long_rollout, hist):
    normalize, wet = data_init
    tensor_map: TensorMap = TensorMap.get_instance()

    num_prognostic_channels = normalize._prognostic_std_np.shape[0]
    num_boundary_channels = normalize._boundary_std_np.shape[0]
    if input_type == "target":
        data = torch.randn([1, num_prognostic_channels * (hist + 1), *wet.shape[1:]])
    elif input_type == "input":
        data = torch.randn(
            [
                6,
                num_prognostic_channels * (hist + 1) + num_boundary_channels,
                *wet.shape[1:],
            ]
        )
    data_dict, data_unnorm_dict = get_aggregator_dicts(
        data,
        wet,
        long_rollout,
        input_type=input_type,
        num_prognostic_channels=num_prognostic_channels * (hist + 1),
        hist=hist,
    )

    var_name = tensor_map.prognostic_var_names[0]
    assert data_dict[var_name].shape == data_unnorm_dict[var_name].shape

    assert torch.isnan(data_dict[var_name][:, :, 0, 1]).all()
    assert torch.isnan(data_dict[var_name][:, :, 1, 0]).all()
    assert torch.isnan(data_dict[var_name][:, :, 1, 2]).all()
