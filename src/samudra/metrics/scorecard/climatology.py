# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

r"""Build a track's reference statistics once, from the training split of its data.

    python -m samudra.metrics.scorecard.climatology samudra_om4_v2/eval.yaml \\
        --scorecard.track=om4-v2026-09-onedeg \\
        --scorecard.climatology_location=/scratch/.../climatology.zarr

The store holds, per channel ``c``, the pentad climatology (``c``) and the
lag-1 coefficient of anomalies from it (``lag1_c``, for damped persistence).
The climatology is both a reference forecast and the seasonal cycle removed
before variance is compared, so every model on a track must use the same one.
It is written to a new local Zarr (v2) store and never overwrites; publishing
it to the registry is a separate, reviewed step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import xarray as xr

from samudra.config import EvalConfig
from samudra.metrics import skill
from samudra.metrics.scorecard.series import LAG1_PREFIX
from samudra.utils.location import LocalLocation
from samudra.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


def attrs(train: xr.Dataset) -> dict:
    first, last = (str(train["time"].values[i])[:10] for i in (0, -1))
    return {
        "climatology_period": f"{first} to {last}",
        "n_samples": int(train.sizes["time"]),
    }


def channel(train: xr.Dataset, name: str) -> xr.Dataset:
    """One channel's pentad climatology and lag-1 anomaly coefficient, in memory.

    Loading the channel whole and reducing it with NumPy is far faster than a
    lazy graph of one task per time step. `compute`, not `load`, so the
    channel is not kept in `train` after this returns (`load` fills it in place).
    """
    field = train[name]
    drop = [c for c in field.coords if c != "time"]
    field = field.drop_vars(drop).compute()
    clim = skill.pentad_climatology(field)
    phi = skill.anomaly_lag1(field, clim)
    return xr.Dataset(
        {name: clim.astype("float32"), LAG1_PREFIX + name: phi.astype("float32")}
    )


def build(train: xr.Dataset, channels: list[str]) -> xr.Dataset:
    """Reference statistics of each channel over all of `train`'s timestamps."""
    out = xr.merge([channel(train, c) for c in channels])
    out.attrs.update(attrs(train))
    return out


def write(train: xr.Dataset, channels: list[str], path: Path) -> None:
    """Write `build(train, channels)` to a new Zarr v2 store, one channel at a time.

    Memory holds one channel's training record at a time. Every part carries the
    attributes, because appending to a Zarr store replaces the root attributes
    with those of the appended part.
    """
    meta = attrs(train)
    for i, name in enumerate(channels):
        part = channel(train, name).assign_attrs(meta)
        part.to_zarr(path, mode="w-" if i == 0 else "a")
        logger.info("climatology: %s done", name)


def main() -> None:
    cfg = EvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    if cfg.scorecard is None:
        raise ValueError("This config has no `scorecard` block.")
    root = cfg.experiment.resolved_data_root
    target = root.resolve(cfg.scorecard.climatology_location)
    if not isinstance(target, LocalLocation):
        raise ValueError(
            f"Climatology target {target} is not local. Build it locally, then "
            "publish it to the registry by review."
        )
    path = Path(target.path)
    if path.exists():
        raise FileExistsError(f"{path} exists; the climatology is never overwritten.")
    bundle = cfg.data.build(root, include_inference=False)
    if len(bundle.train_sources) != 1:
        raise ValueError("The climatology is defined for a single training source.")
    train = bundle.train_sources[0].to_xarray_dataset()
    channels = list(bundle.data_layout.prognostic_var_names)
    logger.info("Computing climatology over %s", attrs(train)["climatology_period"])
    write(train, channels, path)
    logger.info("Wrote %s", path)


if __name__ == "__main__":
    main()
