# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

# Independent xESMF reproduction of the areacello Option-1 vs Option-2 comparison.
# Run: micromamba run -n esmf_check python areacello_xesmf_check.py
#
# Option 1 (Adam): assume each target cell is fully ocean -> xe.util.cell_area(target).
# Option 2 (Adam): land-aware area via *conservative* regridding of the native ocean mask.
#
# Method: build a proper conservative regridder from the native OM4 tracer grid
# (cell centers = geolon/geolat, cell corners from the supergrid) onto each regular
# gaussian target grid, then conservatively regrid the native binary wet mask. With
# method="conservative_normed", each target cell receives the area-weighted mean of the
# source wet mask = the fraction of the target cell that is ocean. Land-aware areacello
# is then frac * (full geometric cell area).
import numpy as np
import xarray as xr
import xesmf as xe

SO = {"anon": True, "client_kwargs": {"endpoint_url": "https://nyu1.osn.mghpcc.org/"}}
R = 6.371e6
PUB = "s3://m2lines-pubs/FOMO/raw"


def native_source():
    """Native OM4 tracer grid with centers (geolon/geolat) and corner bounds (from supergrid)."""
    stat = xr.open_zarr(
        f"{PUB}/ocean_static_no_mask_table.zarr", storage_options=SO, decode_times=False
    )
    sg = xr.open_zarr(
        f"{PUB}/grids/ocean_hgrid.zarr", storage_options=SO, decode_times=False
    )
    # supergrid -> tracer-cell corners are the even-indexed supergrid points (1081 x 1441)
    lon_b = sg.x.isel(nyp=slice(0, None, 2), nxp=slice(0, None, 2)).values
    lat_b = sg.y.isel(nyp=slice(0, None, 2), nxp=slice(0, None, 2)).values
    src = xr.Dataset(
        {
            "wet": (("y", "x"), stat["wet"].values.astype("float64")),
            "areacello": (("y", "x"), stat["areacello"].values.astype("float64")),
        },
        coords={
            "lon": (("y", "x"), stat["geolon"].values),
            "lat": (("y", "x"), stat["geolat"].values),
            "lon_b": (("y_b", "x_b"), lon_b),
            "lat_b": (("y_b", "x_b"), lat_b),
        },
    )
    return src


def target_grid(path):
    """Regular gaussian target grid with centers + corner bounds, in xESMF convention."""
    g = xr.open_zarr(path, storage_options=SO)
    return xr.Dataset(
        coords={
            "lon": (("y", "x"), g["grid_lont"].values),
            "lat": (("y", "x"), g["grid_latt"].values),
            "lon_b": (("y_b", "x_b"), g["grid_lon"].values),
            "lat_b": (("y_b", "x_b"), g["grid_lat"].values),
        }
    )


def run(name, path, src):
    tgt = target_grid(path)
    ny, nx = tgt.lat.shape

    # Conservative, area-normalized regrid of the binary wet mask -> ocean fraction per target cell.
    rg = xe.Regridder(src, tgt, method="conservative_normed", periodic=True)
    frac = rg(src["wet"]).values  # (ny, nx) in [0,1]

    # Option 1: full geometric cell area (assumes 100% ocean).
    opt1_area = xe.util.cell_area(tgt, earth_radius=R)  # (ny, nx), m^2
    opt1_area = np.asarray(opt1_area)

    tgt_wet = frac > 1e-6
    o1_ocean = np.nansum(np.where(tgt_wet, opt1_area, 0.0))  # full area on wet cells
    o2_ocean = np.nansum(frac * opt1_area)  # land-aware area
    native_ocean = float((src["areacello"] * src["wet"]).sum())

    print(f"\n=== {name}  ({ny} x {nx}) ===")
    print(f"native ocean area (truth)            : {native_ocean:.5e} m^2")
    print(f"Opt1 ocean area (full area, wet cells): {o1_ocean:.5e} m^2")
    print(f"Opt2 ocean area (conservative, land-aware): {o2_ocean:.5e} m^2")
    print(
        f"  -> Opt1 overestimates ocean area by {100 * (o1_ocean - o2_ocean) / o2_ocean:.2f} %"
    )
    print(
        f"  -> Opt2 vs native truth mismatch     : {100 * (o2_ocean - native_ocean) / native_ocean:.3f} %"
    )
    coastal = tgt_wet & (frac < 0.95)
    print(
        f"  wet cells={int(tgt_wet.sum())}  coastal(frac<0.95)={int(coastal.sum())} "
        f"({100 * coastal.sum() / tgt_wet.sum():.1f}% of wet)  median frac={np.median(frac[tgt_wet]):.4f}"
    )
    return frac


if __name__ == "__main__":
    src = native_source()
    print(
        "native total ocean area:",
        float((src["areacello"] * src["wet"]).sum()),
        "m^2 (Earth ocean ~3.61e14)",
    )
    for nm, p in [
        ("1deg", f"{PUB}/grids/gaussian_grid_180_by_360.zarr"),
        ("halfdeg", f"{PUB}/grids/gaussian_grid_360_by_720.zarr"),
    ]:
        run(nm, p, src)
