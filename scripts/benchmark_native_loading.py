# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare local loaders using identical schedules and prepared batch values."""

import argparse
import gc
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import torch

from samudra.config import (
    CpuDataLoadingConfig,
    DataConfig,
    Om4DataSourceConfig,
    RustDataLoadingConfig,
    TensorStoreDataLoadingConfig,
)
from samudra.datasets import ModelBatch, TorchTrainDataset
from samudra.train_data_loader import build_train_batch_loader
from samudra.utils.location import LocalLocation


def digest_batch(batch: ModelBatch) -> str:
    digest = hashlib.sha256()
    for step in batch.steps:
        for tensor in step:
            values = tensor.cpu().numpy().copy()
            # Normalize representations that compare equal numerically.
            values[np.isnan(values)] = np.nan
            values[values == 0] = 0
            digest.update(str(values.shape).encode())
            digest.update(values.tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["cpu", "rust", "tensorstore"])
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--read-threads", type=int, default=8)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=1300)
    parser.add_argument("--window-count", type=int, default=120)
    parser.add_argument("--hist", type=int, default=1)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.batches <= args.window_count or args.rounds < 1:
        parser.error("Require 0 < batches <= window-count and rounds >= 1")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.ones(1, device=device)
        synchronize()
    loading = (
        CpuDataLoadingConfig(num_workers=args.workers)
        if args.backend == "cpu"
        else {
            "rust": RustDataLoadingConfig,
            "tensorstore": TensorStoreDataLoadingConfig,
        }[args.backend](max_concurrent_reads=args.read_threads, prefetch_batches=2)
    )
    source_cfg = Om4DataSourceConfig.model_validate(
        {
            "train_time": {"start": "1958-01-01", "end": "2021-12-31"},
            "val_time": {"start": "2022-01-01", "end": "2022-12-31"},
            "data_location": "OM4.zarr",
            "data_means_location": "OM4_means.zarr",
            "data_stds_location": "OM4_stds.zarr",
            "prognostic_vars_key": "thermo_dynamic_all",
            "boundary_vars_key": "tau_hfds",
        }
    )
    start = time.perf_counter()
    container = DataConfig(sources=[source_cfg], loading=loading).build(
        LocalLocation(path=args.data_root.resolve())
    )
    source = container.train_sources[0]
    dataset = TorchTrainDataset(
        source,
        None,
        source.data_layout.prognostic_var_names,
        source.data_layout.boundary_var_names,
        hist=args.hist,
        steps=args.steps,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        concurrent_compute_=True,
    )
    if args.start_index < 0 or args.start_index + args.window_count > len(dataset):
        parser.error("Requested sampling window exceeds the training data")
    schedule = [
        [int(i)]
        for i in np.random.default_rng(871).choice(
            np.arange(args.start_index, args.start_index + args.window_count),
            size=args.batches,
            replace=False,
        )
    ]
    loader = build_train_batch_loader(
        [dataset],
        schedule,
        device,
        loading,
        pin_memory=device.type == "cuda",
        multiprocessing_context=multiprocessing.get_context("spawn")
        if loading.num_pytorch_workers()
        else None,
        worker_seed=0,
    )
    setup_seconds = time.perf_counter() - start
    measurements = []
    digests = []
    try:
        # Warm exactly the measured schedule and check every batch outside timing.
        synchronize()
        start = time.perf_counter()
        first_batch_seconds = None
        for batch in loader:
            synchronize()
            if first_batch_seconds is None:
                first_batch_seconds = time.perf_counter() - start
            digests.append(digest_batch(batch))
            del batch
        if args.reference:
            reference = json.loads(args.reference.read_text())
            if (
                schedule != reference["schedule"]
                or digests != reference["batch_digests"]
            ):
                raise AssertionError(
                    "Prepared batch values differ from the reference run"
                )
        gc.collect()
        for round_index in range(args.rounds):
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            count = 0
            for batch in loader:
                synchronize()
                count += 1
                del batch
            elapsed = time.perf_counter() - start
            assert count == args.batches
            measurement = {
                "round": round_index,
                "ms_per_batch": 1000 * elapsed / count,
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated(device)
                / 2**20
                if device.type == "cuda"
                else None,
            }
            measurements.append(measurement)
            print(json.dumps(measurement), flush=True)
    finally:
        loader.close()
    result = {
        "backend": args.backend,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "cpu",
        "data_root": str(args.data_root.resolve()),
        "hist": args.hist,
        "steps": args.steps,
        "channels": list(source.channels),
        "workers": loading.num_pytorch_workers(),
        "read_threads": args.read_threads,
        "setup_seconds": setup_seconds,
        "first_batch_seconds": first_batch_seconds,
        "schedule": schedule,
        "batch_digests": digests,
        "measurements": measurements,
        "median_ms_per_batch": float(
            np.median([m["ms_per_batch"] for m in measurements])
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "backend": args.backend,
                "median_ms_per_batch": result["median_ms_per_batch"],
                "verified_batches": len(digests) if args.reference else 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
