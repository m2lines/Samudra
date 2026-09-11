# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Stream local velocity sources into float32 Zarr and fit training-only statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]
from numcodecs import Blosc  # type: ignore[import-untyped]

from samudra.metrics.kernels import (
    coerce_datetime_values,
    geostrophic_velocity_from_zos,
)

TRAIN_END = "2018-09-30"
SPLITS = {
    "train": ("1958-01-01", TRAIN_END),
    "validation": ("2019-04-01", "2020-09-30"),
    "test": ("2021-04-01", "2022-12-31"),
}


def duacs_grid(ds: xr.Dataset) -> xr.Dataset:
    """Normalize the local prepared and raw archives' documented coordinate names."""
    for lat, lon in (("lat", "lon"), ("latitude", "longitude"), ("y", "x")):
        if lat in ds.dims and lon in ds.dims:
            if ds[lat].ndim != 1 or ds[lon].ndim != 1:
                raise ValueError(
                    "DUACS requires one-dimensional geographic coordinates"
                )
            return ds.rename({lat: "y", lon: "x"}) if lat != "y" else ds
    raise ValueError("No recognized DUACS latitude/longitude dimensions")


def velocity_view(ds: xr.Dataset, kind: str) -> xr.Dataset:
    """Use absolute geostrophic velocities, with no reconstruction of SSH."""
    if kind == "duacs":
        ds = duacs_grid(ds)
        ds = ds.assign_coords(x=ds.x % 360).sortby("x").sortby("y")
        weights = np.cos(np.deg2rad(ds.y))
        fields = {}
        for target, source in (("u", "ugos"), ("v", "vgos")):
            # Area-weighted 2x2 averaging; require all four native cells to exist.
            valid = ds[source].notnull()
            numerator = (ds[source].fillna(0) * weights).coarsen(y=2, x=2).sum()
            denominator = (xr.ones_like(ds[source]) * weights).coarsen(y=2, x=2).sum()
            fields[target] = (numerator / denominator).where(
                valid.coarsen(y=2, x=2).reduce(np.sum) == 4
            )
        result = xr.Dataset(fields)
    elif kind == "om4":
        u, v = geostrophic_velocity_from_zos(ds.zos.where(ds.mask_0))
        result = xr.Dataset({"u": u, "v": v})
    else:
        raise ValueError(f"Unknown source kind: {kind}")
    result = result.where(abs(result.y) >= 5)
    for name in ("u", "v"):
        result[name].attrs["units"] = "m s-1"
    return result.assign_coords(time=coerce_datetime_values(result.time.values))


def prepare(input_path: Path, output_path: Path, kind: str, batch_times: int = 4):
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace {output_path}")
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    source = xr.open_zarr(input_path, consolidated=True)
    if kind == "om4":
        # Model auxiliary tasks never see the observation validation/test years.
        source = source.sel(time=slice(None, TRAIN_END))
    if kind == "duacs":
        source = duacs_grid(source)
        if (source.sizes["y"], source.sizes["x"]) != (1440, 2880):
            raise ValueError("Expected the local native 0.125-degree DUACS archive")
    output_path.mkdir(parents=True)
    values_sum = values_sq = count = monthly_sum = monthly_count = None
    for begin in range(0, source.sizes["time"], batch_times):
        fields = velocity_view(
            source.isel(time=slice(begin, begin + batch_times)), kind
        )
        fields = fields.transpose("time", "y", "x").astype(np.float32).compute()
        fields.attrs = {
            "source": str(input_path),
            "kind": kind,
            "target": "absolute geostrophic velocity (m/s)",
        }
        for name in fields.variables:
            fields[name].encoding = {}
        if begin == 0:
            h, w = fields.sizes["y"], fields.sizes["x"]
            compressor = Blosc(cname="lz4", clevel=1, shuffle=Blosc.SHUFFLE)
            encoding = {
                v: {"chunks": (1, h, w), "compressor": compressor} for v in ("u", "v")
            }
            fields.to_zarr(
                output_path / "fields.zarr",
                mode="w",
                encoding=encoding,
                consolidated=False,
            )
            shape = (2, h, w)
            values_sum = np.zeros(shape, dtype=np.float64)
            values_sq = np.zeros(shape, dtype=np.float64)
            count = np.zeros(shape, dtype=np.int32)
            monthly_sum = np.zeros((12, *shape), dtype=np.float64)
            monthly_count = np.zeros((12, *shape), dtype=np.int32)
        else:
            fields.to_zarr(
                output_path / "fields.zarr", append_dim="time", consolidated=False
            )
        assert values_sum is not None and values_sq is not None and count is not None
        assert monthly_sum is not None and monthly_count is not None
        training = fields.sel(time=slice(None, TRAIN_END))
        data = (
            training[["u", "v"]]
            .to_array()
            .transpose("time", "variable", "y", "x")
            .values
        )
        valid = np.isfinite(data)
        clean = np.where(valid, data, 0).astype(np.float64)
        values_sum += clean.sum(axis=0)
        values_sq += (clean**2).sum(axis=0)
        count += valid.sum(axis=0).astype(np.int32)
        months = training.time.dt.month.values
        for month in np.unique(months):
            monthly_sum[month - 1] += clean[months == month].sum(axis=0)
            monthly_count[month - 1] += (
                valid[months == month].sum(axis=0).astype(np.int32)
            )
        print(
            json.dumps(
                {
                    "prepared": min(begin + batch_times, source.sizes["time"]),
                    "total": source.sizes["time"],
                    "source": kind,
                }
            ),
            flush=True,
        )
    assert count is not None and values_sum is not None and values_sq is not None
    assert monthly_sum is not None and monthly_count is not None
    zarr.consolidate_metadata(output_path / "fields.zarr")
    mean = values_sum / np.maximum(count, 1)
    mask = (count >= 24).all(axis=0)
    area = np.cos(np.deg2rad(fields.y.values))[:, None] * np.ones(
        (1, fields.sizes["x"])
    )
    weight = area[None] * mask[None] * count
    variance = np.maximum(values_sq / np.maximum(count, 1) - mean**2, 0)
    std = np.sqrt((variance * weight).sum(axis=(1, 2)) / weight.sum(axis=(1, 2)))
    if not np.isfinite(std).all() or (std < 1e-6).any():
        raise ValueError(f"Invalid training scales: {std}")
    monthly_mean = np.where(
        monthly_count > 0, monthly_sum / np.maximum(monthly_count, 1), mean[None]
    )
    np.savez(
        output_path / "stats.npz",
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        mask=mask,
        monthly_mean=monthly_mean.astype(np.float32),
    )
    manifest = {
        "source": str(input_path),
        "source_attrs": source.attrs,
        "kind": kind,
        "splits": SPLITS,
        "train_end": TRAIN_END,
        "shape": [source.sizes["time"], *mask.shape],
        "standard_deviation_m_per_s": std.tolist(),
        "equatorial_exclusion_deg": 5,
        "duacs_coarsening": "cosine-area-weighted 2x2 with complete native-cell coverage"
        if kind == "duacs"
        else None,
        "temporal_processing": "Preserve source timestamps and averages; retrospective prediction, not issuance-vintage forecasting",
        "complete": True,
    }
    (output_path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("duacs", "om4"), required=True)
    parser.add_argument("--batch-times", type=int, default=4)
    args = parser.parse_args()
    if args.batch_times < 1:
        parser.error("--batch-times must be positive")
    prepare(args.input, args.output, args.kind, args.batch_times)


if __name__ == "__main__":
    main()
