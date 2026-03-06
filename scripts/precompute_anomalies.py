#!/usr/bin/env python
"""Precompute hfds_anomalies and append it to each OM4 zarr store.

When hfds_anomalies already exists in a zarr store, the training code
(`compute_anomalies` in utils/data.py) skips the runtime computation.
This avoids materialising ~20 GB (quarter-degree) of numpy arrays per
process — the dominant source of CPU-RAM OOM during multi-scale training.

Usage
-----
    # On torch, for all three scales:
    python scripts/precompute_anomalies.py /path/to/om4_quarterdeg_v2
    python scripts/precompute_anomalies.py /path/to/om4_halfdeg_v4
    python scripts/precompute_anomalies.py /path/to/om4_onedeg_v3

Each invocation:
  1. Opens  <root>/OM4.zarr, computes hfds_anomalies (= hfds − seasonal cycle)
  2. Appends the variable to <root>/OM4.zarr
  3. Appends its mean to   <root>/OM4_means.zarr
  4. Appends its std  to   <root>/OM4_stds.zarr

The script is idempotent: if hfds_anomalies already exists it will
overwrite it (with mode="a").
"""

import argparse
import logging
import sys
from pathlib import Path

import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def precompute_anomalies(data_root: Path) -> None:
    data_path = data_root / "OM4.zarr"
    means_path = data_root / "OM4_means.zarr"
    stds_path = data_root / "OM4_stds.zarr"

    for p in [data_path, means_path, stds_path]:
        if not p.exists():
            logger.error(f"Expected {p} to exist")
            sys.exit(1)

    var = "hfds_anomalies"
    base_var = "hfds"

    logger.info(f"Opening {data_path}")
    ds = xr.open_zarr(data_path)

    if base_var not in ds:
        logger.error(f"{base_var} not found in {data_path}")
        sys.exit(1)

    grid = dict(ds.sizes)
    grid.pop("time", None)
    logger.info(f"Grid: {grid}, time steps: {ds.sizes.get('time', '?')}")

    # Compute climatology (small: 366 × lat × lon)
    logger.info("Computing daily climatology ...")
    climatology = ds[base_var].groupby("time.dayofyear").mean("time").compute()

    # Compute anomaly
    logger.info("Computing anomaly (this reads the full time series once) ...")
    day_of_year = ds[base_var]["time"].dt.dayofyear
    anomaly = (ds[base_var] - climatology.sel(dayofyear=day_of_year)).compute()
    anomaly = anomaly.drop_vars("dayofyear")
    anomaly.name = var

    anomaly_mean = float(anomaly.mean().values)
    anomaly_std = float(anomaly.std().values)
    logger.info(f"  mean={anomaly_mean:.6f}, std={anomaly_std:.6f}")

    # Write anomaly variable to the data store
    logger.info(f"Appending {var} to {data_path} ...")
    anomaly_ds = anomaly.to_dataset(name=var)
    anomaly_ds.to_zarr(data_path, mode="a")

    # Write mean
    logger.info(f"Appending {var} mean to {means_path} ...")
    means_ds = xr.Dataset({var: xr.DataArray(anomaly_mean)})
    means_ds.to_zarr(means_path, mode="a")

    # Write std
    logger.info(f"Appending {var} std to {stds_path} ...")
    stds_ds = xr.Dataset({var: xr.DataArray(anomaly_std)})
    stds_ds.to_zarr(stds_path, mode="a")

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "data_root",
        type=Path,
        help="Directory containing OM4.zarr, OM4_means.zarr, OM4_stds.zarr",
    )
    args = parser.parse_args()
    precompute_anomalies(args.data_root)


if __name__ == "__main__":
    main()
