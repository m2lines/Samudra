# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging

import fsspec
import numpy as np
import xarray as xr
from xgcm import Grid

from ocean_preprocessing.dataset_validation import ds_processed_validate
from ocean_preprocessing.schema import (
    OM4_3D_VARS,
    OM4_OPTIONAL_2D_VARS,
    OM4_REQUIRED_2D_VARS,
)
from ocean_preprocessing.utils import apply_mask

from .interpolate import interpolate_to_cell_centers

logger = logging.getLogger(__name__)

OM4_REQUIRED_DATA_VARS = frozenset(OM4_3D_VARS + OM4_REQUIRED_2D_VARS)
OM4_OPTIONAL_DATA_VARS = frozenset(OM4_OPTIONAL_2D_VARS)
WFO_DIMS = ("time", "yh", "xh")
FIVE_DAYS_SECONDS = 5 * 24 * 60 * 60
WFO_SURGERY_ALIGNMENT = (
    "donor time == recipient time_bnds upper bound; "
    "donor wfo relabeled to recipient interval-midpoint time"
)


# load supergrid and extract the angles
# Some awesome material to understand the 'supergrid' (is that the same as the mosaic?) https://gist.github.com/adcroft/c1e207024fe1189b43dddc5f1fe7dd6c
def convert_super_grid(ds_super_grid: xr.Dataset):
    h_rename = {"nyp": "yh", "nxp": "xh"}
    b_rename = {"nyp": "yh_b", "nxp": "xh_b"}

    h_indicies = dict(nyp=slice(1, None, 2), nxp=slice(1, None, 2))
    b_indicies = dict(
        nyp=slice(0, None, 2), nxp=slice(0, None, 2)
    )  # locations of 'bound variables required by xesmf

    angle_h = ds_super_grid.angle_dx.isel(**h_indicies).rename(h_rename)
    lon_h = ds_super_grid.x.isel(**h_indicies).rename(h_rename)
    lat_h = ds_super_grid.y.isel(**h_indicies).rename(h_rename)

    lon_b = ds_super_grid.x.isel(**b_indicies).rename(b_rename)
    lat_b = ds_super_grid.y.isel(**b_indicies).rename(b_rename)
    return angle_h, lon_h, lat_h, lon_b, lat_b


def normalize_vertical_coords(ds: xr.Dataset) -> xr.Dataset:
    """Rename raw OM4 vertical coordinates to the pipeline's expected names.

    In OM4 output, ``z_l`` is the depth of the cell centers (equivalent to
    ``lev``) and ``z_i`` is the depth of the cell interfaces (equivalent to
    ``ilev``, which holds one extra entry). The 5-daily snapshot sources expose
    ``z_l`` without a matching ``z_i``, so each coordinate is renamed
    independently rather than gating the center rename (``z_l`` -> ``lev``) on
    the interface coordinate being present.
    """
    vertical_rename = {
        raw: new
        for raw, new in (("z_l", "lev"), ("z_i", "ilev"))
        if raw in ds.coords or raw in ds.dims
    }
    return ds.rename(vertical_rename) if vertical_rename else ds


def select_om4_variables(ds: xr.Dataset) -> xr.Dataset:
    """Keep the stable emulator inputs and reject incomplete OM4 sources.

    Snapshot archives also contain diagnostic and native-grid fields that the
    averaged archive does not. Passing every compatible source variable through
    would make the published dataset depend accidentally on the source archive's
    contents. ``wfo`` is retained when available because it is an intentional
    five-day-mean forcing in the snapshot product.
    """
    available = set(ds.data_vars)
    missing = OM4_REQUIRED_DATA_VARS - available
    if missing:
        raise ValueError(f"OM4 source is missing required variables: {sorted(missing)}")

    selected = OM4_REQUIRED_DATA_VARS | (OM4_OPTIONAL_DATA_VARS & available)
    ignored = available - selected
    if ignored:
        logger.info("ignoring undeclared OM4 source variables: %s", sorted(ignored))
    return ds[sorted(selected)]


def transplant_wfo(
    recipient: xr.Dataset, donor: xr.Dataset, *, donor_path: str
) -> xr.Dataset:
    """Add five-day-mean ``wfo`` to an averaged OM4 source.

    The averaged archive labels each five-day interval at its midpoint, while
    the combined snapshot archive labels the same interval at its upper bound.
    Validate that relationship and the native tracer grid before relabeling the
    donor data with the recipient's timestamps. The operation remains lazy;
    data chunks are read only when the preprocessing result is computed.
    """
    if "wfo" in recipient:
        raise ValueError(
            "OM4 recipient already contains 'wfo'; refusing to overwrite it"
        )
    if "wfo" not in donor:
        raise ValueError("OM4 freshwater-flux donor is missing required variable 'wfo'")
    if "time_bnds" not in recipient:
        raise ValueError(
            "OM4 recipient must provide time_bnds to validate freshwater-flux alignment"
        )

    wfo = donor["wfo"]
    if wfo.dims != WFO_DIMS:
        raise ValueError(f"Donor 'wfo' has dimensions {wfo.dims}; expected {WFO_DIMS}")
    if "time" not in donor.coords:
        raise ValueError("OM4 freshwater-flux donor has no time coordinate")

    bounds = recipient["time_bnds"]
    if bounds.dims != ("time", "nv") or bounds.sizes["nv"] != 2:
        raise ValueError(
            "OM4 recipient time_bnds must have dimensions ('time', 'nv') with nv=2"
        )

    def timedelta_seconds(value) -> float:
        if hasattr(value, "total_seconds"):
            return value.total_seconds()
        return float(value / np.timedelta64(1, "s"))

    starts = bounds.isel(nv=0).values
    ends = bounds.isel(nv=1).values
    midpoints = recipient["time"].values
    interval_seconds = np.array(
        [timedelta_seconds(end - start) for start, end in zip(starts, ends)]
    )
    left_seconds = np.array(
        [
            timedelta_seconds(midpoint - start)
            for midpoint, start in zip(midpoints, starts)
        ]
    )
    right_seconds = np.array(
        [timedelta_seconds(end - midpoint) for end, midpoint in zip(ends, midpoints)]
    )
    if not (
        np.all(interval_seconds == FIVE_DAYS_SECONDS)
        and np.array_equal(left_seconds, right_seconds)
    ):
        raise ValueError(
            "OM4 recipient time_bnds must describe centered five-day intervals"
        )

    for coord in ("xh", "yh"):
        if coord not in recipient.coords or coord not in donor.coords:
            raise ValueError(f"Both OM4 stores must provide the {coord!r} coordinate")
        try:
            xr.testing.assert_identical(recipient[coord], donor[coord])
        except AssertionError as error:
            raise ValueError(
                f"OM4 freshwater-flux donor {coord!r} grid does not match recipient"
            ) from error

    interval_ends = bounds.isel(nv=1, drop=True)
    if not np.array_equal(donor["time"].values, interval_ends.values):
        raise ValueError(
            "OM4 freshwater-flux donor timestamps must exactly match the upper "
            "bounds of the recipient's five-day intervals"
        )

    if "time: mean" not in wfo.attrs.get("cell_methods", ""):
        raise ValueError(
            "Donor 'wfo' is not identified as a time mean in its cell_methods attribute"
        )
    expected_attrs = {
        "standard_name": "water_flux_into_sea_water",
        "units": "kg m-2 s-1",
    }
    for attr, expected in expected_attrs.items():
        if wfo.attrs.get(attr) != expected:
            raise ValueError(
                f"Donor 'wfo' {attr} must be {expected!r}; got {wfo.attrs.get(attr)!r}"
            )

    transplanted = wfo.assign_coords(time=recipient["time"])
    result = recipient.assign(wfo=transplanted)
    result.attrs.update(
        {
            "m2lines/wfo_surgery_source": donor_path,
            "m2lines/wfo_surgery_alignment": WFO_SURGERY_ALIGNMENT,
        }
    )
    return result


def vertical_cell_metadata(ds: xr.Dataset) -> tuple[xr.DataArray, xr.DataArray]:
    """Return consistently typed cell thickness and interface coordinates."""
    if "ilev" in ds.coords:
        dz = xr.DataArray(ds.ilev.diff("ilev").values, dims=["lev"])
        ilev = ds["ilev"]
    else:
        # The snapshot archive omits interface depths, so add the canonical
        # OM4 vertical grid used by the existing processed datasets.
        dz = xr.DataArray(
            [
                5,
                10,
                15,
                20,
                30,
                50,
                70,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
                600,
                800,
                1000,
                1000,
                1000,
            ],
            dims=["lev"],
        )
        ilev = xr.DataArray(
            [
                0,
                5,
                15,
                30,
                50,
                80,
                130,
                200,
                300,
                450,
                650,
                900,
                1200,
                1600,
                2100,
                2700,
                3500,
                4500,
                5500,
                6500,
            ],
            dims=["ilev"],
        )

    # The processed coordinate contract is float64 for both averaged sources
    # (which provide ilev) and snapshot sources (which use the fallback above).
    return dz.astype("float64"), ilev.astype("float64")


def om4_preprocessing(
    zarr_data_path,
    nc_grid_path,
    nc_mosaic_path,
    fs=fsspec,
    backend_kwargs=None,
    wfo_source_path=None,
):
    """OM4 specific preprocessing."""
    ds = xr.open_dataset(
        zarr_data_path, engine="zarr", chunks={}, backend_kwargs=backend_kwargs
    )

    if wfo_source_path is not None:
        donor = xr.open_dataset(
            wfo_source_path,
            engine="zarr",
            chunks={},
            backend_kwargs=backend_kwargs,
        )
        ds = transplant_wfo(ds, donor, donor_path=wfo_source_path)

    ds = select_om4_variables(ds)
    ds = normalize_vertical_coords(ds)

    dz, ilev = vertical_cell_metadata(ds)

    ds = ds.assign_coords(dz=dz)

    # trim excess padding
    if ds["xq"].size == ds["xh"].size + 1:
        ds = ds.isel(xq=slice(1, None))
    if ds["yq"].size == ds["yh"].size + 1:
        ds = ds.isel(yq=slice(1, None))

    grid = Grid(
        ds,
        coords={
            "X": {"center": "xh", "right": "xq"},
            "Y": {"center": "yh", "right": "yq"},
        },
        boundary="extend",
        periodic=["xh", "xq"],
    )
    ds_interpolated = interpolate_to_cell_centers(ds, ds.thetao, grid)

    # remove the same areas as for the tracers again
    tracer_wetmask = ~np.isnan(ds_interpolated.thetao.isel(time=0)).drop_vars("time")
    ds = apply_mask(ds_interpolated, tracer_wetmask)
    ds = ds.assign_coords(ilev=ilev, wetmask=tracer_wetmask)

    if nc_grid_path.endswith(".zarr"):
        ds_grid = xr.open_zarr(nc_grid_path, chunks={})
    else:
        with fs.open(nc_grid_path) as f:
            ds_grid = xr.open_dataset(f).load()

    ds_grid = ds_grid.drop_vars("time", errors="ignore")
    ds_grid = ds_grid.set_coords([v for v in ds_grid.data_vars])

    ds = ds.assign_coords(
        lon=ds_grid.geolon, lat=ds_grid.geolat, areacello=ds_grid.areacello
    )

    # drop (for now) all the coords on non-tracer position
    required_coords = [
        "lon",
        "time",
        "xh",
        "lat",
        "ilev",
        "lev",
        "yh",
        "areacello",
        "wetmask",
        "dz",
    ]
    drop_coords = [co for co in ds.coords.keys() if co not in required_coords]
    ds = ds.drop(drop_coords)

    if nc_mosaic_path.endswith(".zarr"):
        ds_super_grid = xr.open_zarr(nc_mosaic_path, chunks={})
    else:
        with fs.open(nc_mosaic_path) as f:
            ds_super_grid = xr.open_dataset(f).load()

    a, lon, lat, lon_b, lat_b = convert_super_grid(ds_super_grid)
    lon_expected = ds_grid.load().geolon.reset_coords(drop=True).drop(["xh", "yh"])
    lat_expected = ds_grid.load().geolat.reset_coords(drop=True).drop(["xh", "yh"])

    # asser that the grid positions extracted are correct (this should maybe live in a test for an upstream function?)
    xr.testing.assert_allclose(lon, lon_expected)
    xr.testing.assert_allclose(lat, lat_expected)

    ds = ds.assign_coords(lon_b=lon_b, lat_b=lat_b, angle=a, lon=lon, lat=lat)
    ds = ds.rename({"xh": "x", "yh": "y", "xh_b": "x_b", "yh_b": "y_b"})
    if "time_bnds" in ds.data_vars:
        ds = ds.drop_vars(["time_bnds"])
    ds = ds.astype(np.float32)
    # higher precision for the area
    ds = ds.assign_coords(areacello=ds.areacello.astype("float64"))
    if wfo_source_path is not None:
        ds.attrs.update(
            {
                "m2lines/wfo_surgery_source": wfo_source_path,
                "m2lines/wfo_surgery_alignment": WFO_SURGERY_ALIGNMENT,
            }
        )
    ds_processed_validate(ds)
    return ds
