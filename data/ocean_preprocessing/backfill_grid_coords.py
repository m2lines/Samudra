# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Backfill grid-metadata coordinates into already-published OM4 Zarr stores.

Older input datasets were written before `flatten_by_depth_level` was fixed to
retain grid metadata, so they only carry `time/x/y` coordinates plus the per-level
`mask_i` variables. Downstream physical analysis (conservation diagnostics, OHC,
zonal means) needs `areacello`, `ocean_fraction`, the cell bounds, and the depth
axis. This module recomputes those coordinates from the source grids -- reusing
the pipeline's own `horizontal_regrid` so the result is identical to what newly
generated datasets now contain -- and writes them *additively* into an existing
store without rewriting any existing chunks.

Run on a local copy first. Writing to a shared corpus is irreversible; use
`--dry_run` to inspect, and the built-in read-back verification.
"""

import logging

import fire
import numpy as np
import xarray as xr
import zarr

from ocean_preprocessing.preprocessing import horizontal_regrid

logger = logging.getLogger("ocean_preprocessing.backfill")

# Canonical OM4 vertical grid (19 levels). Mirrors `build_om4_spec` in the
# training package and the values in `manual_v0_fixes`.
DZ = np.array(
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
    dtype="float64",
)
LEV = np.array(
    [
        2.5,
        10,
        22.5,
        40,
        65,
        105,
        165,
        250,
        375,
        550,
        775,
        1050,
        1400,
        1850,
        2400,
        3100,
        4000,
        5000,
        6000,
    ],
    dtype="float64",
)
# Depth of each level's top interface: [0, dz0, dz0+dz1, ...] (length 19).
ILEV_TOP = np.concatenate([[0.0], np.cumsum(DZ)[:-1]])

# The coordinates this module produces and writes.
BACKFILL_COORDS = (
    "lon",
    "lat",
    "lon_b",
    "lat_b",
    "areacello",
    "ocean_fraction",
    "dz",
    "lev",
)

# xESMF target-grid coordinate renaming (same mapping the CLI uses).
_TARGET_GRID_RENAME = {
    "grid_x": "x_b",
    "grid_y": "y_b",
    "grid_xt": "x",
    "grid_yt": "y",
    "grid_lon": "lon_b",
    "grid_lat": "lat_b",
    "grid_lont": "lon",
    "grid_latt": "lat",
}


def _native_source(static_ds: xr.Dataset, supergrid_ds: xr.Dataset) -> xr.Dataset:
    """Build the native-grid source dataset that `horizontal_regrid` consumes.

    Carries a per-level ocean mask (derived from `deptho` and the level interfaces)
    plus the tracer-cell centers and corner bounds. A `thetao` data variable
    (1 over ocean, NaN over land) is included because `horizontal_regrid` derives
    the target binary wetmask from the regridded `thetao` NaN pattern.
    """
    static = static_ds.squeeze(drop=True)
    deptho = static["deptho"].astype("float64")  # (yh, xh)

    # A tracer cell at level k is ocean where the floor is below the cell top.
    wetmask = xr.concat([deptho > top for top in ILEV_TOP], dim="lev").assign_coords(
        lev=("lev", LEV)
    )  # (lev, yh, xh) bool

    # Tracer-cell corners are the even-indexed supergrid points (n+1 per side).
    lon_b = supergrid_ds.x.isel(nyp=slice(0, None, 2), nxp=slice(0, None, 2))
    lat_b = supergrid_ds.y.isel(nyp=slice(0, None, 2), nxp=slice(0, None, 2))

    thetao = (
        wetmask.astype("float64").where(wetmask).expand_dims(time=[0])
    )  # (time, lev, yh, xh): 1.0 over ocean, NaN over land

    source = xr.Dataset(
        {"thetao": thetao.rename({"yh": "y", "xh": "x"})},
        coords={
            "wetmask": wetmask.rename({"yh": "y", "xh": "x"}),
            "lon": static["geolon"].astype("float64").rename({"yh": "y", "xh": "x"}),
            "lat": static["geolat"].astype("float64").rename({"yh": "y", "xh": "x"}),
            "lon_b": lon_b.rename({"nyp": "y_b", "nxp": "x_b"}),
            "lat_b": lat_b.rename({"nyp": "y_b", "nxp": "x_b"}),
        },
    )
    return source


def compute_target_grid_coords(
    static_ds: xr.Dataset,
    supergrid_ds: xr.Dataset,
    target_grid_ds: xr.Dataset,
) -> xr.Dataset:
    """Compute the grid-metadata coordinates for one target resolution.

    Reuses `horizontal_regrid` so `areacello` (full geometric cell area) and the
    per-level `ocean_fraction` (conservative regrid of the native wetmask) match
    the live preprocessing pipeline exactly. Returns a coordinates-only dataset on
    the target `x`/`y` grid.

    Note on area weighting: `areacello` here is the *geometric* cell area, i.e. it
    assumes every cell is fully ocean. For conservation diagnostics (OHC, area- or
    volume-weighted global means) the correct, land-aware weight is
    ``areacello * ocean_fraction`` -- NOT `areacello` alone. Using bare `areacello`
    overestimates coastal/global ocean area by ~2-4% at 1deg (less at finer grids).
    Both fields are stored so analysis can recover geometric area, land-aware area,
    or the wet fraction as needed.
    """
    source = _native_source(static_ds, supergrid_ds)
    target = target_grid_ds.rename(_TARGET_GRID_RENAME)

    regridded = horizontal_regrid(source, target)

    coords = xr.Dataset(
        coords={
            "x": regridded["x"],
            "y": regridded["y"],
            "lev": ("lev", LEV),
            "dz": ("lev", DZ),
            "lon": regridded["lon"],
            "lat": regridded["lat"],
            "lon_b": regridded["lon_b"],
            "lat_b": regridded["lat_b"],
            "areacello": regridded["areacello"],
            "ocean_fraction": regridded["ocean_fraction"],
        }
    )
    return coords


def _open(path: str, storage_options: dict | None = None, **kwargs) -> xr.Dataset:
    # storage_options only apply to remote stores; local paths ignore them.
    so = storage_options if "://" in str(path) else None
    return xr.open_zarr(path, storage_options=so, **kwargs)


def backfill_store(
    store_path: str,
    static_path: str,
    supergrid_path: str,
    target_grid_path: str,
    dry_run: bool = True,
    overwrite_existing: bool = False,
    rtol: float = 1e-6,
    storage_options: dict | None = None,
) -> None:
    """Additively write grid-metadata coordinates into an existing OM4 Zarr store.

    Existing arrays (the prognostic `var_i`/`mask_i` chunks) are never rewritten;
    only new coordinate variables are appended. The store's `x`/`y` are checked
    against the computed grid before writing so the additive write is positionally
    correct.

    Args:
        store_path: The OM4.zarr to backfill (use a local copy for testing).
        static_path: Native `ocean_static_no_mask_table.zarr` (deptho/geolon/geolat).
        supergrid_path: `ocean_hgrid.zarr` supergrid (for cell corner bounds).
        target_grid_path: `gaussian_grid_<N>_by_<M>.zarr` matching the store's grid.
        dry_run: If True (default), report what would be written and stop.
        overwrite_existing: Allow re-writing coords that already exist in the store.
        rtol: Relative tolerance for the store-vs-grid `x`/`y` alignment check.
        storage_options: fsspec options for remote stores (e.g. ``{"anon": True}``);
            ignored for local paths.
    """
    static = _open(static_path, storage_options, decode_times=False)
    supergrid = _open(supergrid_path, storage_options, decode_times=False)
    target_grid = _open(target_grid_path, storage_options, decode_times=False).load()

    logger.info("computing grid-metadata coordinates from source grids")
    coords = compute_target_grid_coords(static, supergrid, target_grid).load()

    apply_grid_coords(
        store_path,
        coords,
        dry_run=dry_run,
        overwrite_existing=overwrite_existing,
        rtol=rtol,
        storage_options=storage_options,
    )


def apply_grid_coords(
    store_path: str,
    coords: xr.Dataset,
    *,
    dry_run: bool = True,
    overwrite_existing: bool = False,
    rtol: float = 1e-6,
    storage_options: dict | None = None,
) -> None:
    """Additively write precomputed grid coords into an existing store.

    Split out from `backfill_store` so the (network-free) write/verify mechanics
    can be exercised independently of the source-grid regridding. Existing arrays
    are never rewritten; only the new coordinate variables are appended, then
    promoted to coordinates.
    """
    store = _open(store_path, storage_options, decode_times=False)

    # Positional-correctness guard: the computed grid must match the store's axes.
    for axis in ("x", "y"):
        if store.sizes[axis] != coords.sizes[axis]:
            raise ValueError(
                f"{axis} size mismatch: store has {store.sizes[axis]}, "
                f"computed grid has {coords.sizes[axis]}. Wrong target grid?"
            )
        if not np.allclose(store[axis].values, coords[axis].values, rtol=rtol):
            raise ValueError(
                f"{axis} coordinate values differ between store and computed grid; "
                "refusing to write to avoid misaligned coordinates."
            )

    present = [c for c in BACKFILL_COORDS if c in store.variables]
    if present and not overwrite_existing:
        raise ValueError(
            f"Store already has coords {present}; pass overwrite_existing=True to "
            "replace them (this rewrites those arrays, not the prognostic data)."
        )

    to_write = [
        c for c in BACKFILL_COORDS if c not in store.variables or overwrite_existing
    ]
    # Build a clean dataset of just the backfill arrays (dims + values), dropping
    # auxiliary coords like `wetmask` and the store-owned `x`/`y`/`time` so the
    # append never rewrites existing arrays. `lev` (name == dim) is written as the
    # new depth dimension coordinate; the rest are promoted to coords afterward.
    write_ds = xr.Dataset(
        {
            name: (coords[name].dims, np.asarray(coords[name].values))
            for name in to_write
        }
    )

    logger.info(
        "coords to add: %s",
        {v: tuple(write_ds[v].dims) for v in write_ds.data_vars},
    )

    if dry_run:
        logger.info("[dry_run] not writing. Re-run with dry_run=False to apply.")
        return

    logger.info("appending coordinates to %s", store_path)
    write_ds.to_zarr(store_path, mode="a", consolidated=True)
    _promote_to_coords(store_path, to_write)

    _verify(store_path, coords, to_write, rtol, storage_options)
    logger.info("backfill complete and verified.")


def _promote_to_coords(store_path: str, coord_names: list[str]) -> None:
    """Mark the newly-written arrays as coordinates (metadata-only, no chunk rewrite).

    Appending to an existing store writes the grid arrays as data variables, since
    the pre-existing prognostic arrays don't reference them. Adding a CF
    ``coordinates`` attribute to each data variable makes xarray promote them back
    to coordinates on read -- matching the layout newly generated datasets have.
    """
    group = zarr.open_group(store_path, mode="a")
    dim_coords = {"x", "y", "time", "lev", "x_b", "y_b"}
    skip = set(coord_names) | dim_coords
    ref = " ".join(coord_names)
    for name, arr in group.arrays():
        if name in skip:
            continue
        existing = arr.attrs.get("coordinates", "")
        merged = list(dict.fromkeys([*existing.split(), *coord_names]))
        arr.attrs["coordinates"] = " ".join(merged) if existing else ref
    zarr.consolidate_metadata(group.store)


def _verify(
    store_path: str,
    coords: xr.Dataset,
    written: list[str],
    rtol: float,
    storage_options: dict | None = None,
):
    """Re-open the store and confirm the new coords landed and prognostics survived."""
    reopened = _open(store_path, storage_options, decode_times=False)
    for name in written:
        if name not in reopened.variables:
            raise ValueError(f"verification failed: {name} missing after write")
        if not np.allclose(
            np.nan_to_num(reopened[name].values),
            np.nan_to_num(coords[name].values),
            rtol=rtol,
        ):
            raise ValueError(f"verification failed: {name} values differ after write")
    # A representative prognostic array must still be readable and unchanged in shape.
    sample = next((v for v in reopened.data_vars if v.startswith("thetao_")), None)
    if sample is not None:
        reopened[sample].isel(time=0).load()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    fire.Fire(backfill_store)
