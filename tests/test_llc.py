# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from samudra.utils.llc import canonicalize_llc_datasets
from tests.llc_fixtures import coarse_llc_dataset, write_coarse_llc_dataset


def canonicalize(data, **kwargs):
    return canonicalize_llc_datasets(
        data,
        data,
        data,
        prognostic_vars_key=kwargs.get("prognostic", "all"),
        boundary_vars_key=kwargs.get("boundary", "all"),
    )


def test_compact_llc_channels_statistics_and_geography(tmp_path):
    original = write_coarse_llc_dataset(tmp_path)
    with xr.open_zarr(tmp_path / "data.zarr") as stored:
        data, means, stds, layout = canonicalize(stored)
        assert stored.U.dims == ("time", "k", "y", "x")
        assert "U_0" not in stored
        assert layout.grid_type == "gaussian"
        assert len(layout.prognostic_var_names) == 205
        assert "Eta" not in layout.boundary_var_names
        np.testing.assert_array_equal(data.lat, original.lat)
        np.testing.assert_array_equal(data.lon, original.lon)
        np.testing.assert_array_equal(data.areacello, original.rA)
        assert data.attrs == original.attrs
        for name in ("U", "V", "Theta", "Salt"):
            for level in (0, 10, 50):
                channel = f"{name}_{level}"
                assert data[channel].dims == ("time", "lat", "lon")
                np.testing.assert_array_equal(
                    data[channel], original[name].isel(k=level)
                )
                assert means[channel].ndim == stds[channel].ndim == 0
                assert means[channel].item() == original[f"{name}_mean"][level].item()
                assert stds[channel].item() == original[f"{name}_std"][level].item()
        for name in ["Eta", *layout.boundary_var_names]:
            np.testing.assert_array_equal(data[name], original[name])
            assert means[name].ndim == stds[name].ndim == 0
        assert "Depth" not in data.data_vars
        assert "repaired" not in data.data_vars


def test_llc_masks_use_wet_area_and_keep_coastal_cells():
    original = coarse_llc_dataset()
    data, _, _, _ = canonicalize(original)
    for level in range(51):
        np.testing.assert_array_equal(
            data[f"mask_{level}"], original.wet_area[level] > 0
        )
    assert not data.mask_0[0, 0].item()
    assert data.mask_0[1, 2].item()
    assert not data.mask_10[1, 2].item()
    assert 0 < original.land_frac[0, 0, 1].item() < 1
    assert data.mask_0[0, 1].item()


def test_llc_single_channel_selection_ignores_unused_fields():
    original = coarse_llc_dataset().drop_vars(["U", "U_mean", "U_std", "oceFWflx"])
    data, means, stds, _ = canonicalize(
        original, prognostic="single_1", boundary="single_1"
    )
    assert "Theta_0" in data and "oceQnet" in data
    assert "U_0" not in data and "Eta" not in data
    assert "U_0" not in means and "U_0" not in stds


@pytest.mark.parametrize("dimension", ["face", "i_g", "j_g"])
def test_llc_rejects_native_layout(dimension):
    with pytest.raises(ValueError, match="native face/tile"):
        canonicalize(coarse_llc_dataset().expand_dims({dimension: [0]}))


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_llc_rejects_invalid_statistics(value):
    original = coarse_llc_dataset()
    original["U_std"][2] = value
    with pytest.raises(ValueError, match="U_std.*finite.*positive"):
        canonicalize(original)


def test_llc_rejects_missing_forcing():
    with pytest.raises(KeyError, match="oceFWflx"):
        canonicalize(coarse_llc_dataset().drop_vars("oceFWflx"))


def test_llc_rejects_wrong_depth_count():
    with pytest.raises(ValueError, match="51 depth levels"):
        canonicalize(coarse_llc_dataset().isel(k=slice(0, 19)))


def test_llc_rejects_statistics_with_spatial_dimensions():
    original = coarse_llc_dataset().drop_vars("U_mean")
    original["U_mean"] = (("k", "one"), np.ones((51, 1)))
    with pytest.raises(ValueError, match="U_mean must have dimensions"):
        canonicalize(original)


def test_llc_rejects_nonfinite_wet_area():
    original = coarse_llc_dataset()
    original["wet_area"][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="wet_area must be finite"):
        canonicalize(original)
