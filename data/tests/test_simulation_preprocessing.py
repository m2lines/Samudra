# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr
from ocean_preprocessing.simulation_preprocessing.gfdl_om4 import (
    normalize_vertical_coords,
    select_om4_variables,
    transplant_wfo,
    vertical_cell_metadata,
)


def _ds_with_vertical(names):
    """Build a tiny dataset whose vertical dimension coordinate(s) are ``names``."""
    coords = {name: (name, np.arange(3)) for name in names}
    return xr.Dataset(coords=coords)


def test_normalize_renames_z_l_without_z_i():
    # Snapshot sources expose z_l (cell centers) but no z_i; z_l must still
    # become lev so downstream rechunking on "lev" succeeds.
    ds = _ds_with_vertical(["z_l"])
    out = normalize_vertical_coords(ds)
    assert "lev" in out.coords
    assert "z_l" not in out.coords


def test_normalize_renames_both_when_present():
    # Averaged sources carry both interfaces and centers.
    ds = _ds_with_vertical(["z_l", "z_i"])
    out = normalize_vertical_coords(ds)
    assert {"lev", "ilev"} <= set(out.coords)
    assert "z_l" not in out.coords and "z_i" not in out.coords


def test_normalize_is_noop_without_raw_vertical_coords():
    # Already-normalized data passes through untouched.
    ds = _ds_with_vertical(["lev"])
    out = normalize_vertical_coords(ds)
    assert "lev" in out.coords


def test_averaged_vertical_cell_metadata_matches_processed_schema_dtype():
    ds = xr.Dataset(
        coords={
            "lev": ("lev", [2.5, 10.0]),
            "ilev": ("ilev", [0, 5, 15]),
        }
    )

    dz, ilev = vertical_cell_metadata(ds)

    assert dz.dims == ("lev",)
    assert dz.dtype == np.dtype("float64")
    assert ilev.dtype == np.dtype("float64")
    np.testing.assert_array_equal(dz, [5.0, 10.0])


def _om4_source(*extra_vars: str) -> xr.Dataset:
    required = [
        "hfds",
        "so",
        "tauuo",
        "tauvo",
        "thetao",
        "uo",
        "vo",
        "wfo",
        "zos",
    ]
    return xr.Dataset(
        {name: ("time", np.ones(2)) for name in [*required, *extra_vars]},
        coords={"time": [0, 1]},
    )


def test_select_om4_variables_retains_declared_snapshot_forcing():
    out = select_om4_variables(_om4_source("hfgeou", "wet"))

    assert set(out.data_vars) == {
        "hfds",
        "so",
        "tauuo",
        "tauvo",
        "thetao",
        "uo",
        "vo",
        "wfo",
        "zos",
    }


def test_select_om4_variables_rejects_incomplete_source():
    source = _om4_source().drop_vars("thetao")

    with pytest.raises(ValueError, match="missing required variables.*thetao"):
        select_om4_variables(source)


@pytest.fixture
def wfo_surgery_sources():
    midpoint = np.array(["1958-01-03T12:00", "1958-01-08T12:00"], dtype="datetime64[m]")
    interval_end = np.array(
        ["1958-01-06T00:00", "1958-01-11T00:00"], dtype="datetime64[m]"
    )
    xh = [0.25, 0.75]
    yh = [-0.25, 0.25]
    recipient = xr.Dataset(
        {
            "hfds": (("time", "yh", "xh"), np.ones((2, 2, 2))),
            "time_bnds": (
                ("time", "nv"),
                np.column_stack(
                    [
                        np.array(["1958-01-01", "1958-01-06"], dtype="datetime64[m]"),
                        interval_end,
                    ]
                ),
            ),
        },
        coords={"time": midpoint, "xh": xh, "yh": yh},
    )
    donor = xr.Dataset(
        {
            "wfo": (
                ("time", "yh", "xh"),
                np.arange(8, dtype="float32").reshape(2, 2, 2),
                {
                    "cell_methods": "area:mean time: mean",
                    "standard_name": "water_flux_into_sea_water",
                    "units": "kg m-2 s-1",
                },
            )
        },
        coords={"time": interval_end, "xh": xh, "yh": yh},
    )
    return recipient, donor


def test_transplant_wfo_relabels_matching_intervals_and_records_provenance(
    wfo_surgery_sources,
):
    recipient, donor = wfo_surgery_sources

    out = transplant_wfo(recipient, donor, donor_path="s3://example/donor.zarr")

    xr.testing.assert_equal(out.wfo, donor.wfo.assign_coords(time=recipient.time))
    assert out.time.identical(recipient.time)
    assert out.attrs["m2lines/wfo_surgery_source"] == "s3://example/donor.zarr"
    assert "upper bound" in out.attrs["m2lines/wfo_surgery_alignment"]


def test_transplant_wfo_rejects_misaligned_intervals(wfo_surgery_sources):
    recipient, donor = wfo_surgery_sources
    donor = donor.assign_coords(time=donor.time + np.timedelta64(5, "D"))

    with pytest.raises(ValueError, match="timestamps must exactly match"):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")


def test_transplant_wfo_requires_centered_five_day_recipient_intervals(
    wfo_surgery_sources,
):
    recipient, donor = wfo_surgery_sources
    recipient["time_bnds"][0, 0] = np.datetime64("1958-01-02")

    with pytest.raises(ValueError, match="centered five-day intervals"):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")


def test_transplant_wfo_rejects_different_native_grid(wfo_surgery_sources):
    recipient, donor = wfo_surgery_sources
    donor = donor.assign_coords(xh=[0.5, 1.0])

    with pytest.raises(ValueError, match="'xh' grid does not match"):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")


def test_transplant_wfo_requires_time_mean_metadata(wfo_surgery_sources):
    recipient, donor = wfo_surgery_sources
    donor.wfo.attrs["cell_methods"] = "area:mean time: point"

    with pytest.raises(ValueError, match="not identified as a time mean"):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")


def test_transplant_wfo_requires_cf_identity_metadata(wfo_surgery_sources):
    recipient, donor = wfo_surgery_sources
    donor.wfo.attrs["units"] = "m s-1"

    with pytest.raises(ValueError, match="units must be 'kg m-2 s-1'"):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")


def test_transplant_wfo_refuses_to_overwrite_existing_data(wfo_surgery_sources):
    recipient, donor = wfo_surgery_sources
    recipient["wfo"] = recipient.hfds

    with pytest.raises(ValueError, match="refusing to overwrite"):
        transplant_wfo(recipient, donor, donor_path="donor.zarr")
