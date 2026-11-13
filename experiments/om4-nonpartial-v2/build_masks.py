# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Build conservative fully-wet masks from native OM4 bathymetry (CPU only)."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe  # type: ignore[import-untyped]

BASE = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/"
EDGES = np.array(
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
    ]
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def read(path):
        return xr.open_zarr(BASE + path, consolidated=True, decode_times=False)

    static = read("raw/ocean_static_no_mask_table.zarr")
    hgrid = read("raw/grids/ocean_hgrid.zarr")
    data = read("v2026-09/om4_onedeg/OM4.zarr")
    source = xr.Dataset()
    # Match preprocessing's float32 conversion before constructing ESMF coordinates.
    for name, field, stride, dims in [
        ("lon", "x", 1, ("y", "x")),
        ("lat", "y", 1, ("y", "x")),
        ("lon_b", "x", 0, ("y_b", "x_b")),
        ("lat_b", "y", 0, ("y_b", "x_b")),
    ]:
        source[name] = (
            dims,
            hgrid[field]
            .values[stride::2, stride::2]
            .astype("float32")
            .astype("float64"),
        )
    target = (
        xr.Dataset({name: data[name] for name in ["lon", "lat", "lon_b", "lat_b"]})
        .astype("float64")
        .load()
    )
    weights_path = args.output / "conservative_weights.nc"
    regrid = xe.Regridder(
        source,
        target,
        "conservative",
        periodic=True,
        ignore_degenerate=True,
        unmapped_to_nan=True,
        filename=str(weights_path),
        reuse_weights=weights_path.exists(),
    )
    depth = static.deptho.squeeze().values
    wet = static.wet.squeeze().values == 1
    assert depth.shape == source.lon.shape
    # Unnormalized destination-area fractions: land contributes zero, not NaN.
    full = np.stack([wet & np.isfinite(depth) & (depth >= b) for b in EDGES[1:]])
    frac = np.asarray(
        regrid(xr.DataArray(full.astype("float64"), dims=("lev", "y", "x")))
    ).reshape(19, 180, 360)
    coverage = np.asarray(regrid(xr.ones_like(source.lon))).reshape(180, 360)
    original = np.stack([data[f"mask_{k}"].values.astype(bool) for k in range(19)])
    horizontal = data.ocean_fraction.values
    tol = 1e-6
    # Coverage check also excludes unmapped cells instead of treating NaNs as wet.
    masks = (
        original & np.isfinite(frac) & (frac >= 1 - tol) & (np.abs(coverage - 1) <= tol)
    )
    assert np.all(masks <= original)
    assert np.all(masks[1:] <= masks[:-1])
    area = data.areacello.values
    rows = []
    for k in range(19):
        rows.append(
            dict(
                level=k,
                top_m=int(EDGES[k]),
                bottom_m=int(EDGES[k + 1]),
                original_cells=int(original[k].sum()),
                retained_cells=int(masks[k].sum()),
                retained_original_area_fraction=float(
                    area[masks[k]].sum() / area[original[k]].sum()
                ),
                horizontally_full_cells=int(
                    (original[k] & (horizontal[k] >= 1 - tol)).sum()
                ),
            )
        )
    output = xr.Dataset(
        {f"mask_{k}": (("y", "x"), masks[k]) for k in range(19)},
        coords={"y": data.y.values, "x": data.x.values},
    )
    output.to_zarr(args.output / "masks.zarr", mode="w", consolidated=True)
    np.savez_compressed(
        args.output / "geometry.npz",
        masks=masks,
        original=original,
        full_fraction=frac,
        horizontal_fraction=horizontal,
        coverage=coverage,
        area=area,
        edges=EDGES,
        latitude=data.y.values,
        longitude=data.x.values,
    )
    manifest = dict(
        source=BASE + "raw/ocean_static_no_mask_table.zarr",
        target=BASE + "v2026-09/om4_onedeg/OM4.zarr",
        tolerance=tol,
        definition="Every positive-area native contributor wet to nominal layer bottom; fraction >= 1-tolerance",
        xesmf_version=xe.__version__,
        mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
        weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        coverage_min=float(np.nanmin(coverage)),
        coverage_max=float(np.nanmax(coverage)),
        levels=rows,
    )
    (args.output / "mask_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
