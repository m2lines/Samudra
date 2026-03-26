#!/usr/bin/env python3
"""
Repack a flat-channel prediction zarr into 4D fields with explicit depth.

Input format (from ocean_emulators.utils.writer.ZarrWriter):
  - Variables like U_0 ... U_50, V_0 ... V_50, Theta_0 ... Theta_50, Salt_0 ... Salt_50
  - Dims: time, lat, lon

Output format:
  - Variables: U, V, Theta, Salt
  - Dims: time, k, lat, lon
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import xarray as xr


DEFAULT_FIELDS = ("U", "V", "Theta", "Salt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-zarr",
        required=True,
        help="Path to input flat prediction zarr (e.g. .../predictions.zarr).",
    )
    parser.add_argument(
        "--output-zarr",
        required=True,
        help="Path to output repacked zarr.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=list(DEFAULT_FIELDS),
        help="Field names to repack (default: U V Theta Salt).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output zarr if it already exists.",
    )
    return parser.parse_args()


def _matching_level_vars(ds: xr.Dataset, field: str) -> list[tuple[int, str]]:
    pat = re.compile(rf"^{re.escape(field)}_(\d+)$")
    matched: list[tuple[int, str]] = []
    for var_name in ds.data_vars:
        m = pat.match(var_name)
        if m:
            matched.append((int(m.group(1)), str(var_name)))
    matched.sort(key=lambda x: x[0])
    return matched


def _repack(ds: xr.Dataset, fields: list[str]) -> xr.Dataset:
    repacked: dict[str, xr.DataArray] = {}
    shared_levels: list[int] | None = None

    for field in fields:
        level_vars = _matching_level_vars(ds, field)
        if not level_vars:
            raise ValueError(
                f"No variables found for field '{field}'. "
                f"Expected names like '{field}_0', '{field}_1', ..."
            )

        levels = [lev for lev, _ in level_vars]
        if shared_levels is None:
            shared_levels = levels
        elif levels != shared_levels:
            raise ValueError(
                f"Field '{field}' has depth levels {levels}, "
                f"but previous fields use {shared_levels}."
            )

        parts = [ds[var_name].expand_dims(k=[lev]) for lev, var_name in level_vars]
        da = xr.concat(parts, dim="k").transpose("time", "k", "lat", "lon")
        da.name = field
        da.attrs.update(ds[level_vars[0][1]].attrs)
        repacked[field] = da

    if shared_levels is None:
        raise ValueError("No fields were repacked.")

    coords: dict[str, xr.DataArray | tuple[str, list[int]]] = {"k": ("k", shared_levels)}
    for coord_name in ("time", "lat", "lon"):
        if coord_name in ds.coords:
            coords[coord_name] = ds[coord_name]

    out = xr.Dataset(repacked, coords=coords)
    out.attrs.update(ds.attrs)
    out.attrs["repacked_from_flat_channels"] = "true"
    return out


def main() -> None:
    args = parse_args()
    input_zarr = Path(args.input_zarr).expanduser().resolve()
    output_zarr = Path(args.output_zarr).expanduser().resolve()

    if not input_zarr.exists():
        raise FileNotFoundError(f"Input zarr not found: {input_zarr}")
    if output_zarr.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output zarr already exists: {output_zarr}. Use --overwrite to replace it."
        )

    ds = xr.open_zarr(input_zarr, chunks="auto")
    out = _repack(ds, args.fields)

    if output_zarr.exists():
        import shutil

        shutil.rmtree(output_zarr)

    encoding = {var: {"compressor": None} for var in out.data_vars}
    out.to_zarr(output_zarr, mode="w", encoding=encoding)

    print(f"Wrote repacked zarr: {output_zarr}")
    print(f"Variables: {list(out.data_vars)}")
    print(f"Dims: {dict(out.sizes)}")


if __name__ == "__main__":
    main()
