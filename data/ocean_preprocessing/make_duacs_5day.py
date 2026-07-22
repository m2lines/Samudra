#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Coarsen the daily (P1D) DUACS L4 altimetry product to a 5-day-average (P5D) Zarr.

DUACS is a global, gridded (0.125 deg) observational sea-surface-height product
distributed at daily resolution. This utility reduces it to non-overlapping
consecutive 5-day means, mirroring the 5-day "simple average" convention used for
the OM4 model data, so the observations can sit alongside the emulator inputs.

The reduction is a streaming time-coarsen: the native on-disk time chunk is 50
steps -- an exact multiple of the 5-step window -- so each output step is produced
from a single input chunk with no cross-chunk shuffle. That keeps the whole job a
cheap, embarrassingly-parallel pass that runs on a laptop LocalCluster or a Coiled
cluster alike.

Usage (dry run first, then the real thing on Coiled):

    export FSSPEC_S3_ENDPOINT_URL=https://nyu1.osn.mghpcc.org/
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...

    # Validate the pipeline against a small slice without writing anything.
    OCEAN_DUACS_CLUSTER=local python -m ocean_preprocessing.make_duacs_5day --dry_run

    # Run the full reduction on a Coiled cluster (the default target).
    python -m ocean_preprocessing.make_duacs_5day

    # ... or locally, streaming the ~67 GB read on your own machine.
    OCEAN_DUACS_CLUSTER=local python -m ocean_preprocessing.make_duacs_5day
"""

import os

# Match the OM4 pipeline: pin blosc/numexpr to single-threaded so concurrent
# workers reading blosc-compressed Zarr chunks don't trip thread-safety bugs.
# Must be set before any blosc operation occurs.
os.environ.setdefault("BLOSC_NTHREADS", "1")
os.environ.setdefault("BLOSC_NOLOCK", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import datetime
import logging
import sys

import fire
import numpy as np
import xarray as xr

from ocean_preprocessing.__main__ import init_cluster
from ocean_preprocessing.utils import get_git_url_hash

logger = logging.getLogger("ocean_preprocessing")
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)-8s][%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# The daily DUACS L4 multi-satellite product on the OSN pod.
DEFAULT_SRC = (
    "s3://emulators/jr7309/data/duacs/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2022-01-01-2023-01-01.zarr"
)
# Sibling P5D store in the user's own bucket. Mirrors the source name but with the
# P1D -> P5D temporal-resolution token swapped.
DEFAULT_OUTPUT = (
    "s3://emulators/am16581/data/2026-06/"
    "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P5D_multi-vars_"
    "179.94W-179.94E_89.94S-89.94N_2022-01-01-2023-01-01.zarr"
)

# Sea-surface-height + geostrophic velocity fields used for emulation. The formal
# mapping-error fields (err_*) and tpa_correction are intentionally dropped.
CORE_VARS = ("adt", "sla", "ugos", "ugosa", "vgos", "vgosa")
# A 0/1 sea-ice status flag. Averaging it over a window yields the fraction of days
# flagged as ice, which is the form we keep.
ICE_VAR = "flag_ice"


def coarsen_5day(
    ds: xr.Dataset,
    window: int = 5,
    core_vars=CORE_VARS,
    ice_var: str = ICE_VAR,
) -> xr.Dataset:
    """Reduce a daily dataset to non-overlapping ``window``-day means.

    Selects the core ocean fields plus the ice flag, then applies
    ``coarsen(time=window, boundary="trim").mean()``: consecutive, non-overlapping
    blocks, with any trailing remainder of fewer than ``window`` days dropped. The
    mean skips NaNs, so land/missing cells stay NaN and the ice flag becomes the
    fractional ice presence over each window.

    The ``time`` coordinate is coarsened by the same mean, labelling each window by
    its midpoint.

    Args:
        ds: Daily dataset with a ``time`` dimension. Values are expected to already
            be decoded to floats (e.g. via ``xr.open_zarr`` with mask-and-scale on).
        window: Number of consecutive time steps per output step.
        core_vars: Continuous variables to average.
        ice_var: Name of the 0/1 ice flag to fold in as a fraction (skipped if absent).

    Returns:
        A dataset with the same spatial dims and a ``time`` dimension of length
        ``len(ds.time) // window``.
    """
    keep = [v for v in (*core_vars, ice_var) if v in ds.data_vars]
    missing = set(core_vars) - set(ds.data_vars)
    if missing:
        raise KeyError(
            f"source dataset is missing expected variables: {sorted(missing)}"
        )

    coarsened = ds[keep].coarsen(time=window, boundary="trim").mean()

    if ice_var in coarsened:
        # No longer a status flag once averaged -- relabel so the metadata is honest.
        ice = coarsened[ice_var]
        ice.attrs = {
            "long_name": f"fraction of days flagged as sea ice over each {window}-day window",
            "comment": (
                f"Mean of the original 0/1 '{ice_var}' status flag over each "
                f"{window}-day window; ranges from 0 (ice-free every day) to 1 "
                f"(ice-flagged every day)."
            ),
            "units": "1",
            "valid_min": 0.0,
            "valid_max": 1.0,
        }

    return coarsened


def _finalize(ds: xr.Dataset, src: str, window: int) -> xr.Dataset:
    """Cast to float32, fix chunks/encoding, and stamp provenance for writing."""
    ds = ds.astype("float32")

    # The decoded source carries int32 scale-factor encoding; clearing it makes
    # to_zarr write honest, uncompressed float32 instead of re-quantizing.
    for var in (*ds.data_vars, *ds.coords):
        ds[var].encoding = {}

    # One chunk per output step, full spatial domain per chunk (~16 MB at float32).
    spatial = {d: -1 for d in ds.dims if d != "time"}
    ds = ds.chunk({"time": 1, **spatial})

    ds.attrs["m2lines/source_dataset"] = src
    ds.attrs["m2lines/temporal_coarsening"] = (
        f"{window}-day consecutive simple mean (coarsen boundary='trim')"
    )
    ds.attrs["m2lines/ocean_emulators_git_hash"] = get_git_url_hash()
    ds.attrs["m2lines/date_created"] = datetime.datetime.now().isoformat()
    ds.attrs["m2lines/cli_args"] = " ".join(sys.argv)
    return ds


def main(
    src: str = DEFAULT_SRC,
    output_path: str = DEFAULT_OUTPUT,
    window: int = 5,
    cluster: str = os.environ.get("OCEAN_DUACS_CLUSTER", "coiled"),
    n_workers: int = int(os.environ.get("OCEAN_DUACS_WORKERS", "20")),
    dry_run: bool = False,
    small_run: bool = False,
    write_retries: int = 5,
) -> None:
    """Coarsen the daily DUACS product to a ``window``-day-average Zarr store.

    Args:
        src: Path to the daily (P1D) DUACS Zarr store.
        output_path: Destination for the coarsened (P5D) Zarr store.
        window: Number of consecutive days to average per output step (default 5).
        cluster: Dask cluster target ('coiled', 'local', 'off', ...). Defaults to
            'coiled'; override with the OCEAN_DUACS_CLUSTER env var or this flag.
        n_workers: Worker count for the cluster (Coiled/local). Override with
            OCEAN_DUACS_WORKERS.
        dry_run: If True, build the lazy result, print its structure, materialize a
            small corner to confirm the reduction runs, and write nothing.
        small_run: If True, restrict to the first ``window * 5`` days (5 output
            steps) before coarsening -- a quick end-to-end smoke of the write path.
        write_retries: Distributed-scheduler retry count for the final Zarr write,
            guarding against transient blosc/S3 chunk-read failures. Ignored without
            a cluster.
    """
    cluster_opts = {}
    if cluster in ("coiled", "local"):
        cluster_opts = {"n_workers": n_workers, "wait_for_workers": True}
    client = init_cluster(cluster, **cluster_opts)

    logger.info(f"opening daily DUACS source: {src}")
    # chunks={} keeps the native on-disk chunking (time=50); mask-and-scale (on by
    # default) decodes the int32 scale-factor encoding to float with NaN fill.
    ds = xr.open_zarr(src, chunks={})

    if small_run:
        n = window * 5
        logger.info(f"**small-run**: restricting to the first {n} days.")
        ds = ds.isel(time=slice(0, n))

    logger.info(f"coarsening to {window}-day means.")
    coarsened = coarsen_5day(ds, window=window)
    out = _finalize(coarsened, src=src, window=window)

    if dry_run:
        logger.info("**dry-run**: resulting dataset structure (nothing written):")
        logger.info("\n%s", out)
        # Print the full time axis (every window-center label), not the truncated
        # repr -- this is what gets pasted into the PR for review.
        logger.info("ds.time coordinate (%d steps):", out.sizes["time"])
        logger.info("\n%r", out.time)
        with np.printoptions(threshold=out.sizes["time"] + 1):
            logger.info("ds.time values:\n%s", out.time.values)
        # Sample a mid-grid (near-equatorial) window: the polar corners are
        # legitimately all-NaN for the absolute fields (no MDT reference under the
        # ice), which would make the smoke test look broken when it isn't.
        sample = out.isel(
            time=slice(0, 2),
            **{
                d: slice(out.sizes[d] // 2, out.sizes[d] // 2 + 64)
                for d in out.dims
                if d != "time"
            },
        )
        logger.info("materializing a small corner to validate the reduction...")
        sample = sample.compute()
        for var in CORE_VARS:
            if var in sample:
                da = sample[var]
                logger.info(
                    f"  {var}: min={float(da.min()):.4g} max={float(da.max()):.4g} "
                    f"mean={float(da.mean()):.4g}"
                )
        logger.info("dry run complete.")
        return

    logger.info(f"writing {window}-day-average dataset to {output_path}")
    delayed = out.to_zarr(
        output_path,
        mode="w",
        zarr_format=2,
        consolidated=True,
        encoding={var: {"compressor": None} for var in out.data_vars},
        compute=False,
    )
    # Reading blosc-compressed source chunks over S3 occasionally returns a
    # truncated buffer (numcodecs#810); retry the task rather than abort the job.
    if client is not None:
        client.compute(delayed, retries=write_retries, sync=True)
    else:
        delayed.compute()
    logger.info("zarr write complete.")


if __name__ == "__main__":
    fire.Fire(main, name="make_duacs_5day")
