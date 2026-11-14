# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Stage v2026-09 1° OM4 for the Rust loader (adds precomputed hfds_anomalies).

Adapted from #969's experiments/onedeg_tendency_scaled/stage_native_om4.py. The source arrays are symlinked; only
hfds_anomalies is written. Verifies that the Rust reader returns exactly what
the Python reader does, using the given train config and CLI overrides.

    python stage_rust_store.py PY_TRAIN_YAML RUST_TRAIN_YAML SOURCE_DIR OUTPUT_DIR
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]

from samudra.config import TrainConfig
from samudra.utils.data import compute_anomalies

ANOMALY = "hfds_anomalies"
STORES = ("OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr")
EXTRA = ("OM4_increment_rms_wet.zarr", "OM4_increment_rms_per_cell.zarr")


def stage_store(source: Path, target: Path, anomaly: xr.DataArray) -> None:
    target.mkdir()
    metadata = json.loads((source / ".zmetadata").read_text())
    for entry in source.iterdir():
        if entry.name != ".zmetadata":
            (target / entry.name).symlink_to(entry.resolve())
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
    config, rust_config, source, output = (Path(a) for a in sys.argv[1:])
    if output.exists():
        raise FileExistsError(output)

    start = time.perf_counter()
    raw_data, raw_means, raw_stds = (
        xr.open_zarr(source / name, chunks=None) for name in STORES
    )
    data, means, stds = compute_anomalies(raw_data, raw_means, raw_stds, (ANOMALY,))
    print(f"Computed {ANOMALY} in {time.perf_counter() - start:.1f}s", flush=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        staged = Path(temporary) / output.name
        staged.mkdir()
        for name, dataset in zip(STORES, (data, means, stds), strict=True):
            stage_store(source / name, staged / name, dataset[ANOMALY])
        for name in EXTRA:
            (staged / name).symlink_to((source / name).resolve())

        root = [f"--experiment.data_root={staged}"]
        native_cfg = TrainConfig.from_yaml_and_cli([str(rust_config), *root])
        assert native_cfg.data.loading.type == "rust", native_cfg.data.loading
        python_cfg = TrainConfig.from_yaml_and_cli([str(config), *root])
        assert python_cfg.data.loading.type == "cpu", python_cfg.data.loading
        native = native_cfg.data.build(
            native_cfg.experiment.resolved_data_root, include_inference=False
        ).train_sources[0]
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
        report = {
            "source": str(source),
            "verified_channels": len(channels),
            "verified_training_indices": indices.tolist(),
            "rust_read_seconds": native_seconds,
            "python_read_seconds": python_seconds,
        }
        (staged / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
        staged.rename(output)
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
