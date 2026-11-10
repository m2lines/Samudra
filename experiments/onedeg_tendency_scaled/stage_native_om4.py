# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Stage the existing OM4 stores with the experiment's heat-flux anomaly."""

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml
import zarr

from samudra.config import TrainConfig
from samudra.utils.data import compute_anomalies

ANOMALY = "hfds_anomalies"
STORES = ("OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr")


def stage_store(source: Path, target: Path, anomaly: xr.DataArray) -> None:
    target.mkdir()
    metadata = json.loads((source / ".zmetadata").read_text())
    for entry in source.iterdir():
        if entry.name != ".zmetadata":
            (target / entry.name).symlink_to(entry.resolve())

    # Write only the new array. The source arrays and root metadata stay shared.
    group = zarr.open_group(str(target), mode="r")
    if ANOMALY in group:
        raise ValueError(f"Source already contains {ANOMALY}: {source}")
    array = zarr.open_array(
        str(target / ANOMALY),
        mode="w",
        shape=anomaly.shape,
        chunks=(1, *anomaly.shape[1:]) if anomaly.ndim else (),
        dtype=anomaly.dtype,
        fill_value=np.nan,
        compressor=None,
    )
    array[...] = anomaly.values
    array.attrs["_ARRAY_DIMENSIONS"] = list(anomaly.dims)
    for name in (".zarray", ".zattrs"):
        metadata["metadata"][f"{ANOMALY}/{name}"] = json.loads(
            (target / ANOMALY / name).read_text()
        )
    (target / ".zmetadata").write_text(json.dumps(metadata))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run_name")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", mmap=True, weights_only=True
    )
    if checkpoint["epoch"] != 6 or "scheduler" not in checkpoint:
        raise ValueError("Expected the epoch-six training checkpoint and scheduler")
    del checkpoint

    start = time.perf_counter()
    original = [xr.open_zarr(args.source / name, chunks=None) for name in STORES]
    data, means, stds = compute_anomalies(*original, (ANOMALY,))
    print(
        f"Calculated original heat-flux anomaly in {time.perf_counter() - start:.1f}s"
    )
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        staged = Path(temporary) / "om4"
        staged.mkdir()
        for name, dataset in zip(STORES, (data, means, stds), strict=True):
            stage_store(args.source / name, staged / name, dataset[ANOMALY])

        train = yaml.safe_load(args.config.read_text())
        original_name = train["experiment"]["name"]
        train["data"]["loading"] = {"type": "rust", "max_concurrent_reads": 8}
        train["experiment"]["data_root"] = str(args.output)
        train["experiment"]["name"] = args.run_name
        config = staged / "train.yaml"
        config.write_text(yaml.safe_dump(train))
        for name in ("eval.yaml", "viz.yaml", "metrics.yaml"):
            contents = (args.config.parent / name).read_text()
            contents = contents.replace(
                f"/runs/{original_name}", f"/runs/{args.run_name}"
            ).replace(f"name: {original_name}-", f"name: {args.run_name}-")
            (staged / name).write_text(contents)
        cfg = TrainConfig.from_yaml_and_cli(
            [str(config), f"--experiment.data_root={staged}"]
        )
        native = cfg.data.build(
            cfg.experiment.resolved_data_root, include_inference=False
        ).train_sources[0]
        python_cfg = TrainConfig.from_yaml_and_cli(
            [str(args.config), f"--experiment.data_root={staged}"]
        )
        reference = python_cfg.data.build(
            python_cfg.experiment.resolved_data_root, include_inference=False
        ).train_sources[0]

        channels = tuple(
            native.data_layout.prognostic_var_names
            + native.data_layout.boundary_var_names
        )
        indices = np.array([0, 321, 1240, native.time.size - 1])
        start = time.perf_counter()
        actual = native.read(indices, channels)
        native_seconds = time.perf_counter() - start
        start = time.perf_counter()
        expected = reference.read(indices, channels)
        python_seconds = time.perf_counter() - start
        np.testing.assert_array_equal(actual, expected)
        for key in ("mean", "std"):
            np.testing.assert_array_equal(
                getattr(native.statistics(channels), key),
                getattr(reference.statistics(channels), key),
            )
        np.testing.assert_array_equal(
            data[ANOMALY].sel(time=native.time.isel(time=indices)).values,
            actual[:, channels.index(ANOMALY)],
        )
        report = {
            "checkpoint_epoch": 6,
            "verified_channels": len(channels),
            "verified_training_indices": indices.tolist(),
            "rust_read_seconds": native_seconds,
            "python_read_seconds": python_seconds,
        }
        (staged / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        staged.rename(args.output)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
