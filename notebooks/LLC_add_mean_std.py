# Adds one or more extra variables' means and stds to the existing LLC4320
# mean/std zarr caches, without recomputing or touching what is already there.
#
# The statistics machinery is the same as LLC_mean_std.py, which is where the
# time sampling comes from; this script only differs in writing with mode="a"
# so that the existing variables in the store are left exactly as they are.
#
# Typical use, adding vertical velocity:
#   uv run notebooks/LLC_add_mean_std.py --vars W
import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from dask.distributed import Client, LocalCluster

from LLC_mean_std import calc_time_indices

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)
log = logging.getLogger(__name__)

OUTPUT_ROOT = Path("/orcd/data/abodner/002/cody/LLC_means_stds")
LLC_PATH = "/orcd/data/abodner/003/LLC4320/LLC4320"

# LLC4320 staggers the vertical: tracers and velocities sit on `k` (51 cell
# centres) while W sits on `k_p1` (52 cell interfaces).
VERTICAL_DIMS = ("k", "k_p1", "k_l", "k_u")


def vertical_dim(da: xr.DataArray) -> str | None:
    """The vertical dimension of `da`, or None if it is a surface field."""
    found = [d for d in da.dims if d in VERTICAL_DIMS]
    if len(found) > 1:
        raise ValueError(f"{da.name} has more than one vertical dim: {found}")
    return found[0] if found else None


def reduction_dims(da: xr.DataArray, k_dim: str | None) -> list[str]:
    """Dims to average over: everything horizontal, plus face.

    Read off the array rather than hard-coded per variable, so a staggered
    variable (U on i_g, V on j_g, W on j/i) is handled without a lookup table.
    """
    return [d for d in da.dims if d != "time_sampled" and d != k_dim]


def existing_level_count(mean_ds: xr.Dataset) -> int:
    """How many depth levels the cache already stores per 3D variable."""
    levels = {
        int(name.rsplit("_lev_", 1)[-1])
        for name in mean_ds.data_vars
        if "_lev_" in name
    }
    if not levels:
        raise ValueError("Found no `*_lev_N` variables in the existing mean cache.")
    if levels != set(range(max(levels) + 1)):
        raise ValueError(f"Existing cache has gaps in its level indices: {levels}")
    return max(levels) + 1


def compute_stats(
    da: xr.DataArray, num_levels: int
) -> tuple[dict[str, xr.DataArray], dict[str, xr.DataArray]]:
    """Per-level mean and std for one variable, as in LLC_mean_std.main()."""
    k_dim = vertical_dim(da)
    dims = reduction_dims(da, k_dim)
    log.info(f"  reducing {da.name} over {dims}, vertical dim {k_dim}")

    mean_da = da.mean(skipna=True, dim=dims).persist()
    var_da = ((da - mean_da) ** 2).mean(skipna=True, dim=dims)

    # Average the mean over time, and the variance over time before rooting it,
    # so std is spatiotemporal exactly as in LLC_mean_std.py.
    mean_da = mean_da.mean(dim="time_sampled")
    std_da = (var_da.mean(dim="time_sampled")) ** 0.5

    if k_dim is None:
        return {str(da.name): mean_da.reset_coords(drop=True)}, {
            str(da.name): std_da.reset_coords(drop=True)
        }

    means, stds = {}, {}
    for klev in range(num_levels):
        means[f"{da.name}_lev_{klev}"] = mean_da.isel({k_dim: klev}).reset_coords(
            drop=True
        )
        stds[f"{da.name}_lev_{klev}"] = std_da.isel({k_dim: klev}).reset_coords(
            drop=True
        )
    return means, stds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vars", nargs="+", default=["W"], help="Variables to add (default: W)."
    )
    parser.add_argument("--num-time-samples", type=int, default=96)
    parser.add_argument("--downsample", type=int, default=8)
    parser.add_argument("--mean-path", default=str(OUTPUT_ROOT / "var_96_LLC_means.zarr"))
    parser.add_argument("--std-path", default=str(OUTPUT_ROOT / "var_96_LLC_stds.zarr"))
    parser.add_argument(
        "--levels",
        type=int,
        default=None,
        help="Levels to write. Defaults to the level count already in the cache.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed the time sampling. The original cache was written unseeded, "
        "so this cannot reproduce its exact sample, only make this run repeatable.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing variables that are already in the cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report the statistics without writing anything.",
    )
    parser.add_argument("--n-workers", type=int, default=11)
    parser.add_argument("--memory-limit", default="60GB")
    args = parser.parse_args()

    mean_path, std_path = Path(args.mean_path), Path(args.std_path)

    # Read the existing stores before touching anything, both to work out how
    # many levels to match and to refuse a write that would clobber them.
    log.info(f"opening existing caches\n  {mean_path}\n  {std_path}")
    existing_mean = xr.open_zarr(mean_path)
    existing_std = xr.open_zarr(std_path)
    before_mean = set(existing_mean.data_vars)
    before_std = set(existing_std.data_vars)
    log.info(f"existing cache holds {len(before_mean)} mean / {len(before_std)} std vars")

    num_levels = args.levels or existing_level_count(existing_mean)
    log.info(f"writing {num_levels} levels per 3D variable")

    if args.seed is not None:
        np.random.seed(args.seed)

    # The cluster owns nothing past this block: the statistics are pulled into
    # memory inside it, so a failure tears the workers down instead of leaking
    # them for the rest of the SLURM allocation, and the write below is plain
    # in-memory I/O.
    with LocalCluster(
        n_workers=args.n_workers,
        threads_per_worker=1,
        memory_limit=args.memory_limit,
        dashboard_address=None,
    ) as cluster, Client(cluster) as client:
        log.info(client)

        log.info(f"opening dataset for {args.vars}")
        LLC = xr.open_zarr(LLC_PATH, consolidated=False)[args.vars]
        LLC = LLC.isel(
            i=slice(None, None, args.downsample), j=slice(None, None, args.downsample)
        )
        # Only chunk dims this subset actually has, and pin every vertical dim to 1.
        horizontal_chunk = max(1, 4320 // (args.downsample * 2))
        chunks = {"i": horizontal_chunk, "j": horizontal_chunk, "time": 1, "face": 13}
        chunks = {d: c for d, c in chunks.items() if d in LLC.dims}
        chunks.update({d: 1 for d in VERTICAL_DIMS if d in LLC.dims})
        LLC = LLC.chunk(chunks)

        time_samples = calc_time_indices(args.num_time_samples)
        log.info(f"sampling {args.num_time_samples} times: {time_samples[:5]}...")
        LLC_sampled = xr.concat(
            [LLC.isel(time=t) for t in time_samples], dim="time_sampled"
        )

        mean_dict, std_dict = {}, {}
        for var in args.vars:
            log.info(f"building graph for {var}")
            means, stds = compute_stats(LLC_sampled[var], num_levels)
            mean_dict.update(means)
            std_dict.update(stds)

        mean_ds = xr.Dataset(mean_dict)
        std_ds = xr.Dataset(std_dict)

        # Names are known before any real work, so bail out here rather than
        # after burning the compute.
        clashes = (set(mean_ds.data_vars) & before_mean) | (
            set(std_ds.data_vars) & before_std
        )
        if clashes and not args.overwrite:
            raise SystemExit(
                f"Refusing to write: {len(clashes)} variable(s) already in the cache, "
                f"e.g. {sorted(clashes)[:5]}. Pass --overwrite to replace them."
            )

        log.info(
            f"computing {len(mean_ds.data_vars)} mean / {len(std_ds.data_vars)} values"
        )
        mean_ds = mean_ds.compute()
        std_ds = std_ds.compute()

    names = list(mean_ds.data_vars)
    for name in dict.fromkeys(names[:3] + names[-1:]):
        log.info(
            f"  {name}: mean={float(mean_ds[name]):.6g} std={float(std_ds[name]):.6g}"
        )
    if not np.isfinite([float(std_ds[n]) for n in std_ds.data_vars]).all():
        log.warning("some stds are not finite -- check for all-NaN levels")

    if args.dry_run:
        log.info("dry run: nothing written")
        return

    # mode="a" adds these data_vars to the group and leaves the rest untouched.
    log.info("appending to the existing stores")
    mean_ds.to_zarr(mean_path, mode="a")
    std_ds.to_zarr(std_path, mode="a")

    # Confirm we added rather than replaced. Reopening the way the training code
    # does also catches a store whose consolidated metadata went stale, which
    # would leave the new variables written but invisible to readers.
    reopened_mean = xr.open_zarr(mean_path)
    reopened_std = xr.open_zarr(std_path)
    after_mean = set(reopened_mean.data_vars)
    after_std = set(reopened_std.data_vars)
    missing = (before_mean - after_mean) | (before_std - after_std)
    if missing:
        raise SystemExit(f"Existing variables disappeared from the cache: {missing}")
    unreadable = (set(mean_ds.data_vars) - after_mean) | (
        set(std_ds.data_vars) - after_std
    )
    if unreadable:
        raise SystemExit(
            f"Wrote {len(unreadable)} variable(s) that do not read back, e.g. "
            f"{sorted(unreadable)[:5]}. The store's consolidated metadata is stale."
        )
    for name in mean_ds.data_vars:
        if float(reopened_mean[name]) != float(mean_ds[name]):
            raise SystemExit(f"{name} did not round-trip through the store.")
    log.info(
        f"done: mean cache {len(before_mean)} -> {len(after_mean)} vars, "
        f"std cache {len(before_std)} -> {len(after_std)} vars"
    )


if __name__ == "__main__":
    main()
