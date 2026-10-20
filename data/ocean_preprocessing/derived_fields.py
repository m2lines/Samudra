# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Precompute training anomalies and visualization profiles on the final grid.

Run ``python -m ocean_preprocessing.derived_fields SOURCE.zarr`` to backfill an
existing processed store, then regenerate its normalization stores.
"""

import argparse

import numpy as np
import xarray as xr

from ocean_preprocessing.make_norm_datasets import normalization_fields
from ocean_preprocessing.utils import zstd_encoding


def heat_flux_anomalies(ds: xr.Dataset) -> xr.DataArray:
    """Remove the full-record day-of-year climatology, as training used to do."""
    flux = ds.hfds
    climatology = flux.groupby("time.dayofyear").mean("time")
    anomaly = (flux - climatology.sel(dayofyear=flux.time.dt.dayofyear)).drop_vars(
        "dayofyear"
    )
    return anomaly.astype(flux.dtype).assign_attrs(
        **{
            **flux.attrs,
            "long_name": "Surface heat flux anomaly",
            "anomaly_method": "subtract full-record time.dayofyear mean at each cell",
            "climatology_start": str(flux.time.values[0]),
            "climatology_end": str(flux.time.values[-1]),
        }
    )


def visualization_fields(ds: xr.Dataset) -> xr.Dataset:
    """Area weights and wet-ocean profiles, retaining time and each depth level."""
    area = ds.areacello.transpose("y", "x").astype("float64")
    # Geometry is small and static; fail before scheduling the physical fields.
    values = area.values
    if (
        not np.isfinite(values).any()
        or np.nansum(values) <= 0
        or np.isinf(values).any()
        or np.any(values < 0)
    ):
        raise ValueError(
            "areacello must contain nonnegative finite areas with a positive sum"
        )
    weights = area.fillna(0) / area.sum(skipna=True)
    weights.attrs = {"units": "1"}
    if "wetmask" in ds or "mask_0" in ds:
        fields = normalization_fields(ds)
    else:
        # Prediction writers already put NaN on land and may omit the mask.
        fields = ds[
            [
                name
                for name, field in ds.data_vars.items()
                if {"time", "y", "x"}.issubset(field.dims)
            ]
        ]
    profiles = fields.weighted(weights).mean(("y", "x"), keep_attrs=True)
    # Do not carry spatial auxiliary coordinates into the reduced products.
    profiles = profiles.reset_coords(drop=True)
    profiles = profiles.rename({name: f"profile_{name}" for name in profiles.data_vars})
    return profiles.assign_coords(areacello_weights=weights)


def add_derived_fields(ds: xr.Dataset) -> xr.Dataset:
    """Add lazy derived arrays after filtering/regridding and before publication."""
    ds = ds.copy()
    if "hfds" in ds:
        ds["hfds_anomalies"] = heat_flux_anomalies(ds)
    derived = visualization_fields(ds)
    ds = ds.assign(derived.data_vars).assign_coords(
        areacello_weights=derived.areacello_weights
    )
    return ds.chunk({dim: 1 if dim == "time" else -1 for dim in ds.dims})


def main(src: str) -> None:
    """Backfill derived arrays without rewriting the source's physical fields."""
    with xr.open_zarr(src, chunks={}) as ds:
        enriched = add_derived_fields(ds)
        names = [
            name for name in enriched.data_vars if str(name).startswith("profile_")
        ]
        if "hfds_anomalies" in enriched:
            names.append("hfds_anomalies")
        derived = enriched[names].assign_coords(ds.coords)
        derived = derived.assign_coords(areacello_weights=enriched.areacello_weights)
        derived.attrs = ds.attrs.copy()
        # Match existing chunks when refreshing derived arrays, including
        # profiles written in multi-time prediction buffers. This keeps parallel
        # writes from overlapping a stored chunk.
        for name in names:
            if name in ds and (chunks := ds[name].encoding.get("chunks")):
                derived[name] = derived[name].chunk(
                    dict(zip(derived[name].dims, chunks))
                )
        encoding = zstd_encoding(derived)
        # xarray rejects explicit encoding for arrays already in a store.
        encoding = {name: value for name, value in encoding.items() if name not in ds}
        derived.to_zarr(
            src,
            mode="a",
            zarr_format=2,
            consolidated=True,
            encoding=encoding,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="Local or remote processed Zarr store to backfill")
    main(parser.parse_args().src)
