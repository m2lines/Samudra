# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
import torch
import xarray as xr
from scipy.stats import pearsonr

from samudra.constants import build_llc_layout
from samudra.utils.data import (
    BatchPreprocessor,
    CanonicalSource,
    Masks,
    compute_anomalies,
    flatten_masks,
    get_aggregator_dicts,
    stack_levels,
    unflatten_masks,
    with_depth_value_vars,
    with_lat_lon_coords,
    with_level_index_vars,
)
from samudra.utils.llc import (
    _flatten_llc_level_vars,
    _rename_llc_level_index_vars,
    _var_without_level,
    canonicalize_llc_datasets,
)
from tests.conftest import (
    TEST_DATA_LAYOUT,
    TEST_FULL_DATA_LAYOUT,
    canonicalize_mock_om4,
)
from tests.llc_fixtures import raw_llc_datasets


def test_mask_roundtrip(data_source):
    data, _, _ = data_source._xarray_datasets_for_testing()

    num_levels = len(TEST_DATA_LAYOUT.depth_levels)
    unflattened = unflatten_masks(data.copy(), num_levels=num_levels)
    flattened = flatten_masks(unflattened.copy())
    mask_vars = [f"mask_{level}" for level in range(num_levels)]

    xr.testing.assert_equal(
        flattened[mask_vars],
        data[mask_vars],
    )


@pytest.mark.parametrize("data_source", ["mock-om4"], indirect=True)
def test_level_index_vars_roundtrip(data_source):
    """`with_level_index_vars` and `with_depth_value_vars` are mutual inverses.

    Exercised on the mock OM4 dataset (in ``<var>_<level_index>`` form) in both
    orders, so each function is run against the other's real output.
    """
    data_layout = TEST_FULL_DATA_LAYOUT
    ds_idx, _, _ = data_source._xarray_datasets_for_testing()

    ds_lev = with_depth_value_vars(ds_idx, data_layout)
    # The inverse actually renamed the 3D vars to the depth-value form.
    assert any("_lev_" in str(v) for v in ds_lev.variables)

    # inverse -> forward recovers the index form ...
    xr.testing.assert_identical(
        with_level_index_vars(ds_lev, data_layout.depth_levels), ds_idx
    )
    # ... and forward -> inverse recovers the depth-value form.
    xr.testing.assert_identical(
        with_depth_value_vars(
            with_level_index_vars(ds_lev, data_layout.depth_levels), data_layout
        ),
        ds_lev,
    )


@pytest.mark.parametrize("data_source", ["mock-om4"], indirect=True)
def test_stack_levels(data_source):
    """`stack_levels` reassembles flattened OM4 data into depth-stacked form."""
    data_layout = TEST_FULL_DATA_LAYOUT
    ds, _, _ = data_source._xarray_datasets_for_testing()
    n = len(data_layout.depth_levels)

    stacked = stack_levels(ds, data_layout)

    # 3D vars gain a `lev` dimension; per-level channels are gone.
    for base in ["thetao", "so", "uo", "vo"]:
        assert stacked[base].sizes["lev"] == n
        assert f"{base}_0" not in stacked.variables
    # Per-level masks collapse into a single stacked wetmask.
    assert stacked["wetmask"].sizes["lev"] == n
    assert "mask_0" not in stacked.variables
    # Level-free variables are untouched.
    assert "lev" not in stacked["zos"].dims


def test_with_lat_lon_coords_preserves_2d_geometry():
    """Real OM4 layout (y/x dims, 2D lat/lon) becomes 1D lat/lon dims + lat_2d/lon_2d.

    The mock fixtures use lat/lon dims directly, so this is the only coverage of the
    y/x -> lat/lon rename and the 2D-coordinate preservation that eval relies on to
    propagate true geometry (essential for curvilinear grids, where the 2D lat/lon
    cannot be rebuilt by broadcasting).
    """
    ny, nx = 3, 4
    y = np.linspace(-60, 60, ny)
    x = np.linspace(0, 270, nx)
    # A curvilinear twist so lat_2d/lon_2d are NOT the outer product of the 1D axes.
    lat2d = y[:, None] + 0.1 * x[None, :]
    lon2d = x[None, :] + 0.1 * y[:, None]
    ds = xr.Dataset(
        {"thetao_0": (["y", "x"], np.ones((ny, nx)))},
        coords={
            "x": ("x", x),
            "y": ("y", y),
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        },
    )

    out = with_lat_lon_coords(ds)

    # x/y dims are renamed to 1D lat/lon dims ...
    assert out["thetao_0"].dims == ("lat", "lon")
    np.testing.assert_array_equal(out["lat"].values, y)
    np.testing.assert_array_equal(out["lon"].values, x)
    # ... and the real 2D geometry is kept verbatim under non-colliding names.
    assert out["lat_2d"].dims == ("lat", "lon")
    np.testing.assert_array_equal(out["lat_2d"].values, lat2d)
    np.testing.assert_array_equal(out["lon_2d"].values, lon2d)


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
    renamed_ds = with_level_index_vars(ds, depth_levels=TEST_DATA_LAYOUT.depth_levels)

    # Test that variables are renamed correctly
    assert "so_11" in renamed_ds.variables  # 1040.0 is OM4 depth index 11
    assert "thetao_0" in renamed_ds.variables  # 2.5 is OM4 depth index 0
    assert "vo_1" in renamed_ds.variables  # 10.0 is OM4 depth index 1
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

    # Should raise ValueError because 9999.0 is not an OM4 depth level
    with pytest.raises(ValueError):
        with_level_index_vars(ds, depth_levels=TEST_DATA_LAYOUT.depth_levels)


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
def preprocessor_input():
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
    test = CanonicalSource.from_canonical_datasets(
        "test",
        data_mean,
        data_mean,
        data_std,
        masks=masks,
        data_layout=TEST_DATA_LAYOUT,
    )

    preprocessor = BatchPreprocessor(
        test,
        prognostic_var_names=["var_0", "var_1"],
        boundary_var_names=["var_2"],
    )
    yield preprocessor, wet_mask


def test_normalize_unnormalize_tensor_prognostic(preprocessor_input):
    preprocessor, wet_mask = preprocessor_input
    data = torch.randn([1, preprocessor._prognostic_std_np.shape[0], *wet_mask.shape])
    input_data = data * wet_mask
    normalized = preprocessor.normalize_tensor_prognostic(input_data)
    unnormalized = preprocessor.unnormalize_tensor_prognostic(
        normalized, fill_value=0.0
    )
    assert torch.allclose(input_data, unnormalized)


@pytest.mark.parametrize("fill_value", [float("nan"), 0.0])
def test_unnormalize_prognostic_tensor(preprocessor_input, fill_value):
    preprocessor, wet_mask = preprocessor_input
    data = torch.randn([1, preprocessor._prognostic_std_np.shape[0], *wet_mask.shape])
    input_data = data * wet_mask
    normalized = preprocessor.normalize_tensor_prognostic(input_data)
    unnormalized = preprocessor.unnormalize_tensor_prognostic(normalized, fill_value)
    assert (torch.sum(torch.isnan(unnormalized)) > 0) == (math.isnan(fill_value))


@pytest.mark.parametrize("data_source", ["compact"], indirect=True)
def test_normalize_compact_mixed_depth_and_surface_stats(data_source):
    source = CanonicalSource.from_datasets(
        *data_source._xarray_datasets_for_testing(),
        data_layout=TEST_FULL_DATA_LAYOUT,
        name="compact-full",
        prognostic_var_names=TEST_FULL_DATA_LAYOUT.prognostic_var_names,
        boundary_var_names=TEST_FULL_DATA_LAYOUT.boundary_var_names,
    )
    preprocessor = BatchPreprocessor(
        source,
        prognostic_var_names=TEST_FULL_DATA_LAYOUT.prognostic_var_names,
        boundary_var_names=TEST_FULL_DATA_LAYOUT.boundary_var_names,
    )

    num_depth = len(TEST_FULL_DATA_LAYOUT.depth_levels)
    expected_prognostic_channels = 4 * num_depth + 1
    assert expected_prognostic_channels == len(
        TEST_FULL_DATA_LAYOUT.prognostic_var_names
    )
    assert preprocessor._prognostic_mean_np.shape == (expected_prognostic_channels,)
    assert preprocessor._prognostic_std_np.shape == (expected_prognostic_channels,)

    lat, lon = source.grid_size
    prognostic = torch.zeros(1, expected_prognostic_channels, lat, lon)
    assert (
        preprocessor.normalize_tensor_prognostic(prognostic).shape == prognostic.shape
    )


def test_rename_llc_level_index_vars():
    original = xr.Dataset(
        {
            "Theta_lev_0": 1.0,
            "Salt_lev_50": 2.0,
            "oceQnet": 3.0,
        }
    )

    renamed = _rename_llc_level_index_vars(original)

    assert set(renamed.data_vars) == {"Theta_0", "Salt_50", "oceQnet"}
    assert renamed["Theta_0"].item() == 1.0
    assert renamed["Salt_50"].item() == 2.0
    assert "Theta_lev_0" in original.data_vars


def test_flatten_llc_level_vars():
    steps, _, _ = raw_llc_datasets()
    data = steps[["Theta"]].isel(face=0, drop=True).rename({"k": "lev"})
    data_layout = build_llc_layout()

    flattened = _flatten_llc_level_vars(data, num_levels=len(data_layout.depth_levels))

    assert "Theta" not in flattened.data_vars
    assert set(flattened.data_vars) == {
        f"Theta_{level}" for level in data_layout.depth_i_levels
    }
    xr.testing.assert_identical(
        flattened["Theta_0"], data["Theta"].isel(lev=0, drop=True).rename("Theta_0")
    )
    last_level = data_layout.depth_i_levels[-1]
    xr.testing.assert_identical(
        flattened[f"Theta_{last_level}"],
        data["Theta"].isel(lev=-1, drop=True).rename(f"Theta_{last_level}"),
    )


@pytest.mark.parametrize(
    ("var_name", "expected"),
    [
        ("Theta_0", "Theta"),
        ("Theta_50", "Theta"),
        ("oceQnet", "oceQnet"),
    ],
)
def test_var_without_level(var_name, expected):
    assert _var_without_level(var_name) == expected


def test_canonicalize_llc_datasets_standardizes_layout():
    data, means, stds = raw_llc_datasets()
    data_layout = build_llc_layout(prognostic_vars_key="all", boundary_vars_key="all")
    expected_theta_0 = data["Theta"].isel(time=0, face=1, k=0, j=1, i=1).item()

    llc_data, llc_means, llc_stds, returned_layout = canonicalize_llc_datasets(
        data,
        means,
        stds,
        face=1,
        i_start=1,
        i_end=4,
        j_start=1,
        j_end=3,
        prognostic_vars_key="all",
        boundary_vars_key="all",
    )

    assert returned_layout == data_layout

    assert "face" not in llc_data.dims
    assert "Theta" not in llc_data.variables
    assert "wetmask" not in llc_data.variables
    assert "Theta_0" in llc_data.variables
    assert "Theta_50" in llc_data.variables
    assert "U_0" in llc_data.variables
    assert "V_0" in llc_data.variables
    assert "mask_0" in llc_data.variables
    assert "mask_w_0" in llc_data.variables
    assert "mask_s_0" in llc_data.variables
    assert llc_data["Theta_0"].dims == ("time", "lat", "lon")
    assert llc_data["U_0"].dims == ("time", "lat", "lon")
    assert llc_data["V_0"].dims == ("time", "lat", "lon")
    assert llc_data["mask_0"].dims == ("lat", "lon")
    assert llc_data["mask_w_0"].dims == ("lat", "lon")
    assert llc_data["mask_s_0"].dims == ("lat", "lon")
    assert llc_data["Theta_0"].shape == (3, 2, 3)
    assert llc_data["Theta_0"].isel(time=0, lat=0, lon=0).item() == expected_theta_0
    assert np.issubdtype(llc_data.time.dtype, np.datetime64)
    assert "Theta_0" in llc_means.variables
    assert "Theta_0" in llc_stds.variables
    assert "Theta_lev_0" not in llc_means.variables
    assert "Theta_lev_0" not in llc_stds.variables


def test_canonicalize_llc_datasets_selects_requested_vars_from_full_root():
    data, means, stds = raw_llc_datasets()

    llc_data, _, _, llc_layout = canonicalize_llc_datasets(
        data,
        means,
        stds,
        face=1,
        i_start=1,
        i_end=4,
        j_start=1,
        j_end=3,
        prognostic_vars_key="single_1",
        boundary_vars_key="single_1",
    )

    expected_vars = {
        *(f"Theta_{i}" for i in llc_layout.depth_i_levels),
        "oceQnet",
        *(f"mask_{i}" for i in llc_layout.depth_i_levels),
    }
    assert expected_vars.issubset(llc_data.data_vars)
    assert "XG" not in llc_data.data_vars
    assert "hFacW" not in llc_data.data_vars
    assert "mask_w_0" not in llc_data.data_vars


def test_llc_all_variable_masks_use_staggered_masks():
    data, means, stds = raw_llc_datasets()
    data_layout = build_llc_layout(prognostic_vars_key="all", boundary_vars_key="all")
    llc_data, llc_means, llc_stds, returned_layout = canonicalize_llc_datasets(
        data,
        means,
        stds,
        face=1,
        i_start=1,
        i_end=4,
        j_start=1,
        j_end=3,
        prognostic_vars_key="all",
        boundary_vars_key="all",
    )
    assert returned_layout == data_layout

    source = CanonicalSource.from_datasets(
        llc_data,
        llc_means,
        llc_stds,
        data_layout=data_layout,
        prognostic_var_names=data_layout.prognostic_var_names,
        boundary_var_names=data_layout.boundary_var_names,
    )

    theta_index = data_layout.prognostic_var_names.index("Theta_0")
    u_index = data_layout.prognostic_var_names.index("U_0")
    v_index = data_layout.prognostic_var_names.index("V_0")
    assert bool(source.masks.prognostic[theta_index, 0, 0])
    assert not bool(source.masks.prognostic[u_index, 0, 0])
    assert not bool(source.masks.prognostic[v_index, 0, 1])

    tau_x_index = data_layout.boundary_var_names.index("oceTAUX")
    tau_y_index = data_layout.boundary_var_names.index("oceTAUY")
    qnet_index = data_layout.boundary_var_names.index("oceQnet")
    assert source.masks.boundary.shape == (len(data_layout.boundary_var_names), 2, 3)
    assert not bool(source.masks.boundary[tau_x_index, 0, 0])
    assert not bool(source.masks.boundary[tau_y_index, 0, 1])
    assert bool(source.masks.boundary[qnet_index, 0, 0])


@pytest.fixture
def data_init(hist: int):
    levels = 19
    lats = 3
    lons = 3
    total_time_steps = 100

    data_layout = TEST_DATA_LAYOUT

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
            "lev": list(TEST_DATA_LAYOUT.depth_levels),
            "lat": np.arange(lats),
            "lon": np.arange(lons),
        },
    )
    data_mean = data.mean() * 0.0
    data_std = data.std() * 0.0 + 1.0
    data, data_mean, data_std = canonicalize_mock_om4(data, data_mean, data_std)
    val = CanonicalSource.from_datasets(
        data,
        data_mean,
        data_std,
        data_layout=TEST_DATA_LAYOUT,
        name="test",
        prognostic_var_names=data_layout.prognostic_var_names,
        boundary_var_names=data_layout.boundary_var_names,
    )

    preprocessor = BatchPreprocessor(
        val,
        prognostic_var_names=data_layout.prognostic_var_names,
        boundary_var_names=data_layout.boundary_var_names,
    )
    yield preprocessor, val.masks.prognostic, data_layout


@pytest.mark.parametrize("input_type", ["input", "target"])
@pytest.mark.parametrize("long_rollout", [True, False])
@pytest.mark.parametrize("hist", [0, 1, 2])
def test_get_norm_unnorm_dicts(data_init, input_type, long_rollout, hist):
    preprocessor, wet, data_layout = data_init

    num_prognostic_channels = preprocessor._prognostic_std_np.shape[0]
    num_boundary_channels = preprocessor._boundary_std_np.shape[0]
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
        preprocessor=preprocessor,
        data_layout=data_layout,
        wet=wet,
        long_rollout=long_rollout,
        input_type=input_type,
        num_prognostic_channels=num_prognostic_channels * (hist + 1),
        hist=hist,
    )

    var_name = data_layout.prognostic_var_names[0]
    assert data_dict[var_name].shape == data_unnorm_dict[var_name].shape

    assert torch.isnan(data_dict[var_name][:, :, 0, 1]).all()
    assert torch.isnan(data_dict[var_name][:, :, 1, 0]).all()
    assert torch.isnan(data_dict[var_name][:, :, 1, 2]).all()
