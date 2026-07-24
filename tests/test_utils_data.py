# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
import torch
import xarray as xr
from scipy.stats import pearsonr

from samudra.constants import TensorMap, build_llc_layout
from samudra.utils.data import (
    CanonicalSource,
    Masks,
    Normalize,
    OceanData,
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
from tests.conftest import TEST_DATA_LAYOUT, TEST_FULL_DATA_LAYOUT
from tests.llc_fixtures import raw_llc_datasets


def test_mask_roundtrip(data_source):
    data = data_source.data

    unflattened = unflatten_masks(data.copy(), data_layout=TEST_DATA_LAYOUT)
    flattened = flatten_masks(unflattened.copy(), data_layout=TEST_DATA_LAYOUT)

    assert flattened == data, "Assume a safe roundtrip"


@pytest.mark.parametrize("data_source", ["mock-om4"], indirect=True)
def test_level_index_vars_roundtrip(data_source):
    """`with_level_index_vars` and `with_depth_value_vars` are mutual inverses.

    Exercised on the mock OM4 dataset (in ``<var>_<level_index>`` form) in both
    orders, so each function is run against the other's real output.
    """
    spec = TEST_FULL_DATA_LAYOUT
    ds_idx = data_source.data  # OM4 data named <var>_<level_index>

    ds_lev = with_depth_value_vars(ds_idx, spec)
    # The inverse actually renamed the 3D vars to the depth-value form.
    assert any("_lev_" in str(v) for v in ds_lev.variables)

    # inverse -> forward recovers the index form ...
    xr.testing.assert_identical(with_level_index_vars(ds_lev, spec), ds_idx)
    # ... and forward -> inverse recovers the depth-value form.
    xr.testing.assert_identical(
        with_depth_value_vars(with_level_index_vars(ds_lev, spec), spec), ds_lev
    )


@pytest.mark.parametrize("data_source", ["mock-om4"], indirect=True)
def test_stack_levels(data_source):
    """`stack_levels` reassembles flattened OM4 data into depth-stacked form."""
    spec = TEST_FULL_DATA_LAYOUT
    ds = data_source.data
    n = len(spec.depth_levels)

    stacked = stack_levels(ds, spec)

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
    renamed_ds = with_level_index_vars(ds, data_layout=TEST_DATA_LAYOUT)

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
        with_level_index_vars(ds, data_layout=TEST_DATA_LAYOUT)


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
    test = CanonicalSource(
        "test",
        data_mean,
        data_mean,
        data_std,
        masks=masks,
        data_layout=TEST_DATA_LAYOUT,
    )

    normalize = Normalize(
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


@pytest.mark.parametrize("fill_value", [float("nan"), 0.0])
def test_unnormalize_prognostic_tensor(normalize_input, fill_value):
    normalize, wet_mask = normalize_input
    data = torch.randn([1, normalize._prognostic_std_np.shape[0], *wet_mask.shape])
    input_data = data * wet_mask
    normalized = normalize.normalize_tensor_prognostic(input_data)
    unnormalized = normalize.unnormalize_tensor_prognostic(normalized, fill_value)
    assert (torch.sum(torch.isnan(unnormalized)) > 0) == (math.isnan(fill_value))


@pytest.mark.parametrize("data_source", ["compact"], indirect=True)
def test_normalize_compact_mixed_depth_and_surface_stats(data_source):
    src = CanonicalSource.from_datasets(
        data_source.data,
        data_source.means,
        data_source.stds,
        data_layout=TEST_FULL_DATA_LAYOUT,
        name="compact-full",
        prognostic_var_names=TEST_FULL_DATA_LAYOUT.prognostic_var_names,
        boundary_var_names=TEST_FULL_DATA_LAYOUT.boundary_var_names,
    )
    normalize = Normalize(
        src,
        prognostic_var_names=TEST_FULL_DATA_LAYOUT.prognostic_var_names,
        boundary_var_names=TEST_FULL_DATA_LAYOUT.boundary_var_names,
    )

    num_depth = len(TEST_FULL_DATA_LAYOUT.depth_levels)
    expected_prognostic_channels = 4 * num_depth + 1
    assert expected_prognostic_channels == len(
        TEST_FULL_DATA_LAYOUT.prognostic_var_names
    )
    assert normalize._prognostic_mean_np.shape == (expected_prognostic_channels,)
    assert normalize._prognostic_std_np.shape == (expected_prognostic_channels,)

    lat, lon = src.grid_size
    prognostic = torch.zeros(1, expected_prognostic_channels, lat, lon)
    assert normalize.normalize_tensor_prognostic(prognostic).shape == prognostic.shape


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
    raw_data, _, _ = raw_llc_datasets()
    data = raw_data[["Theta"]].isel(face=0, drop=True).rename({"k": "lev"})
    data_layout = build_llc_layout()

    flattened = _flatten_llc_level_vars(data, data_layout=data_layout)

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

    llc_data, llc_means, llc_stds = canonicalize_llc_datasets(
        data,
        means,
        stds,
        face=1,
        i_start=1,
        i_end=4,
        j_start=1,
        j_end=3,
        data_layout=data_layout,
    )

    assert "face" not in llc_data.dims
    assert "Theta" not in llc_data.variables
    assert "wetmask" not in llc_data.variables
    assert "Theta_0" in llc_data.variables
    assert "Theta_50" in llc_data.variables
    assert "U_0" in llc_data.variables
    assert "V_0" in llc_data.variables
    assert "wetmask_0" in llc_data.variables
    assert "mask_w_0" in llc_data.variables
    assert "mask_s_0" in llc_data.variables
    assert llc_data["Theta_0"].dims == ("time", "y", "x")
    assert llc_data["U_0"].dims == ("time", "y", "x")
    assert llc_data["V_0"].dims == ("time", "y", "x")
    assert llc_data["wetmask_0"].dims == ("y", "x")
    assert llc_data["mask_w_0"].dims == ("y", "x")
    assert llc_data["mask_s_0"].dims == ("y", "x")
    assert llc_data["Theta_0"].shape == (3, 2, 3)
    assert llc_data["Theta_0"].isel(time=0, y=0, x=0).item() == expected_theta_0
    assert np.issubdtype(llc_data.time.dtype, np.datetime64)
    assert "Theta_0" in llc_means.variables
    assert "Theta_0" in llc_stds.variables
    assert "Theta_lev_0" not in llc_means.variables
    assert "Theta_lev_0" not in llc_stds.variables


def test_canonicalize_llc_datasets_selects_requested_vars_from_full_root():
    data, means, stds = raw_llc_datasets()

    llc_data, _, _ = canonicalize_llc_datasets(
        data,
        means,
        stds,
        face=1,
        i_start=1,
        i_end=4,
        j_start=1,
        j_end=3,
        data_layout=build_llc_layout(),
    )

    llc_spec = build_llc_layout()
    expected_vars = {
        *(f"Theta_{i}" for i in llc_spec.depth_i_levels),
        "oceQnet",
        *llc_spec.mask_vars,
    }
    assert expected_vars.issubset(llc_data.data_vars)
    assert "XG" not in llc_data.data_vars
    assert "hFacW" not in llc_data.data_vars
    assert "mask_w_0" not in llc_data.data_vars


def test_llc_all_variable_masks_use_staggered_masks():
    data, means, stds = raw_llc_datasets()
    data_layout = build_llc_layout(prognostic_vars_key="all", boundary_vars_key="all")
    llc_data, llc_means, llc_stds = canonicalize_llc_datasets(
        data,
        means,
        stds,
        face=1,
        i_start=1,
        i_end=4,
        j_start=1,
        j_end=3,
        data_layout=data_layout,
    )

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

    tensor_map = TensorMap(data_layout=TEST_DATA_LAYOUT)

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
    val = CanonicalSource.from_datasets(
        data,
        data_mean,
        data_std,
        data_layout=TEST_DATA_LAYOUT,
        name="test",
        prognostic_var_names=tensor_map.prognostic_var_names,
        boundary_var_names=tensor_map.boundary_var_names,
    )

    normalize = Normalize(
        val,
        prognostic_var_names=tensor_map.prognostic_var_names,
        boundary_var_names=tensor_map.boundary_var_names,
    )
    yield normalize, val.masks.prognostic, tensor_map


@pytest.mark.parametrize("input_type", ["input", "target"])
@pytest.mark.parametrize("long_rollout", [True, False])
@pytest.mark.parametrize("hist", [0, 1, 2])
def test_get_norm_unnorm_dicts(data_init, input_type, long_rollout, hist):
    normalize, wet, tensor_map = data_init

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
        normalize=normalize,
        tensor_map=tensor_map,
        wet=wet,
        long_rollout=long_rollout,
        input_type=input_type,
        num_prognostic_channels=num_prognostic_channels * (hist + 1),
        hist=hist,
    )

    var_name = tensor_map.prognostic_var_names[0]
    assert data_dict[var_name].shape == data_unnorm_dict[var_name].shape

    assert torch.isnan(data_dict[var_name][:, :, 0, 1]).all()
    assert torch.isnan(data_dict[var_name][:, :, 1, 0]).all()
    assert torch.isnan(data_dict[var_name][:, :, 1, 2]).all()


def test_ocean_data_with_time():
    """Test slicing OceanData across the time dimension."""
    batch, time, var, lat, lon = 2, 5, 3, 4, 6
    data = torch.randn(batch, time, var, lat, lon)
    means = torch.tensor([1.0, 2.0, 3.0])
    stds = torch.tensor([0.5, 1.0, 2.0])
    mask = torch.ones(var, dtype=torch.bool)
    ocean_data = OceanData(data=data, means=means, stds=stds, mask=mask)

    sliced = ocean_data.with_time(slice(0, 3))

    assert sliced.data.shape[1] == 3
    assert torch.equal(sliced.data, ocean_data.data[:, 0:3, :, :, :])
    # Other fields should be unchanged
    assert torch.equal(sliced.means, ocean_data.means)
    assert torch.equal(sliced.stds, ocean_data.stds)


@pytest.mark.parametrize("masked_fill_value", [0.0, -1.0])
def test_ocean_data_normalize_and_mask(masked_fill_value):
    """Test that masked positions receive the fill value when normalizing first."""
    batch, time, var, lat, lon = 2, 5, 3, 4, 6
    data = torch.randn(batch, time, var, lat, lon)
    data[:, :, :, 0, 0] = float("nan")  # Simulate land
    means = torch.tensor([1.0, 2.0, 3.0])
    stds = torch.tensor([0.5, 1.0, 2.0])
    mask = torch.ones(var, lat, lon, dtype=torch.bool)
    mask[:, 0, 0] = False  # Mark as land

    ocean_data = OceanData(data=data, means=means, stds=stds, mask=mask)
    result = ocean_data.normalize_and_mask(
        normalize_before_mask=True, masked_fill_value=masked_fill_value
    )

    assert result.shape == ocean_data.data.shape
    # Masked positions should have the fill value
    mask_expanded = mask.unsqueeze(0).unsqueeze(0)
    masked_positions = ~mask_expanded.expand_as(result)
    assert torch.all(result[masked_positions] == masked_fill_value)
    # Valid positions should not be NaN
    valid_positions = mask_expanded.expand_as(result)
    assert not torch.any(torch.isnan(result[valid_positions]))


def test_ocean_data_normalize_and_mask_values():
    """Test that normalization produces expected values with known inputs."""
    batch, time, num_var, lat, lon = 1, 1, 2, 2, 2
    data = torch.tensor(
        [[[[[10.0, 10.0], [10.0, 10.0]], [[20.0, 20.0], [20.0, 20.0]]]]]
    )
    means = torch.tensor([5.0, 10.0])
    stds = torch.tensor([5.0, 5.0])
    mask = torch.ones(num_var, lat, lon, dtype=torch.bool)

    ocean_data = OceanData(data=data, means=means, stds=stds, mask=mask)
    result = ocean_data.normalize_and_mask(
        normalize_before_mask=True, masked_fill_value=0.0
    )

    # Expected: (10 - 5) / 5 = 1.0 for var 0, (20 - 10) / 5 = 2.0 for var 1
    expected_var0 = torch.ones(batch, time, lat, lon)
    expected_var1 = torch.ones(batch, time, lat, lon) * 2.0

    assert torch.allclose(result[:, :, 0, :, :], expected_var0)
    assert torch.allclose(result[:, :, 1, :, :], expected_var1)
