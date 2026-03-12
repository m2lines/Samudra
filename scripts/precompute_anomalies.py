#!/usr/bin/env python
"""Precompute hfds_anomalies and append it to each OM4 zarr store.

When hfds_anomalies already exists in a zarr store, the training code
(`compute_anomalies` in utils/data.py) skips the runtime computation.
This avoids materialising ~20 GB (quarter-degree) of numpy arrays per
process — the dominant source of CPU-RAM OOM during multi-scale training.

Usage
-----
    # On torch, for all three scales:
    uv run scripts/precompute_anomalies.py /path/to/om4_quarterdeg_v2
    uv run scripts/precompute_anomalies.py /path/to/om4_halfdeg_v4
    uv run scripts/precompute_anomalies.py /path/to/om4_onedeg_v3

Each invocation:
  1. Opens  <root>/OM4.zarr, computes hfds_anomalies (= hfds − seasonal cycle)
  2. Appends the variable to <root>/OM4.zarr
  3. Appends its mean to   <root>/OM4_means.zarr
  4. Appends its std  to   <root>/OM4_stds.zarr

The script is idempotent: if hfds_anomalies already exists it will
overwrite it (with mode="a").
"""
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "xarray[io]",
#   "zarr<3",
#   "dask",
#   "numcodecs>=0.15",
#   "ocean-emulators",
# ]
#
# [tool.uv.sources]
# ocean-emulators = { path = "../" }
# ///

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

from ocean_emulators.utils.data import compute_anomalies

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

ANOMALIES_VARS = ("hfds_anomalies",)


def precompute(data_root: Path) -> None:
    data_path = data_root / "OM4.zarr"
    means_path = data_root / "OM4_means.zarr"
    stds_path = data_root / "OM4_stds.zarr"

    for p in [data_path, means_path, stds_path]:
        if not p.exists():
            logger.error(f"Expected {p} to exist")
            sys.exit(1)

    logger.info(f"Opening {data_path}")
    ds = xr.open_zarr(data_path)
    means = xr.open_zarr(means_path)
    stds = xr.open_zarr(stds_path)

    grid = dict(ds.sizes)
    grid.pop("time", None)
    logger.info(f"Grid: {grid}, time steps: {ds.sizes.get('time', '?')}")

    # compute_anomalies skips vars that already exist in ds, so if everything
    # is already present we can exit early.
    missing = [var for var in ANOMALIES_VARS if var not in ds]
    if not missing:
        logger.info("All anomaly variables already exist — nothing to do.")
        return

    logger.info(f"Computing missing anomalies: {missing} ...")
    ds, means, stds = compute_anomalies(ds, means, stds, tuple(missing))

    # Write each anomaly variable back to the zarr stores.
    for var in ANOMALIES_VARS:
        base_var = var.replace("_anomalies", "")
        anomaly = ds[var]

        logger.info(
            f"  {var}: mean={float(means[var].values):.6f}, "
            f"std={float(stds[var].values):.6f}"
        )

        # Preserve the source chunking so per-timestep reads stay efficient.
        source_encoding = ds[base_var].encoding
        chunks = source_encoding.get("chunks")
        if chunks is None:
            spatial = [ds.sizes[d] for d in ds[base_var].dims if d != "time"]
            chunks = tuple([1] + spatial)
            logger.warning(
                f"No chunk encoding found on {base_var}; defaulting to {chunks}"
            )

        logger.info(f"Appending {var} to {data_path} ...")
        anomaly.to_dataset(name=var).to_zarr(
            data_path, mode="a", encoding={var: {"chunks": chunks}}
        )

        logger.info(f"Appending {var} mean to {means_path} ...")
        xr.Dataset({var: means[var]}).to_zarr(means_path, mode="a")

        logger.info(f"Appending {var} std to {stds_path} ...")
        xr.Dataset({var: stds[var]}).to_zarr(stds_path, mode="a")

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_root",
        type=Path,
        help="Directory containing OM4.zarr, OM4_means.zarr, OM4_stds.zarr",
    )
    args = parser.parse_args()
    precompute(args.data_root)


if __name__ == "__main__":
    main()
