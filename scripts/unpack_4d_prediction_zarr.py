#!/usr/bin/env python3
"""Rebuild a flat-channel predictions.zarr (U_0 ... V_50, Eta) from a repacked 4D zarr."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import dask
import xarray as xr


def unpack(ds: xr.Dataset) -> xr.Dataset:
    """Inverse of repack_flat_prediction_zarr._repack."""
    flat = {}
    for name, da in ds.data_vars.items():
        if "k" in da.dims:
            for k in ds.k.values:
                flat[f"{name}_{int(k)}"] = da.sel(k=k, drop=True)
        else:
            flat[name] = da
    out = xr.Dataset(flat, coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon}).drop_encoding()
    out.attrs.update({k: v for k, v in ds.attrs.items() if k != "repacked_from_flat_channels"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zarr", required=True)
    parser.add_argument("--output-zarr", required=True)
    args = parser.parse_args()

    output = Path(args.output_zarr)
    temp = output.with_name(output.name + ".tmp")
    for path in (output, temp):
        if path.exists():
            raise FileExistsError(f"{path} already exists")

    ds = xr.open_zarr(args.input_zarr, chunks={})
    out = unpack(ds)
    encoding = {"time": {"dtype": "float64", "units": ds.time.encoding["units"],
                         "calendar": ds.time.encoding["calendar"]}}
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    print(f"Writing {len(out.data_vars)} variables {dict(out.sizes)} to {temp} on {workers} threads", flush=True)
    with dask.config.set(scheduler="threads", num_workers=workers):
        out.to_zarr(temp, mode="w", encoding=encoding, consolidated=True)
    temp.rename(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
