# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compare warm-cache Python and Rust raw-batch loading on local OM4."""

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from samudra.constants import build_om4_spec
from samudra.datasets import TorchTrainDataset
from samudra.rust_data import RustBatchDataset
from samudra.utils.data import DataSource
from samudra.utils.location import LocalLocation
from samudra.utils.train import collate_raw_train_data


def timed(callable_):
    start = time.perf_counter()
    result = callable_()
    return time.perf_counter() - start, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--prognostic-vars-key", default="thermo_dynamic_5")
    parser.add_argument("--boundary-vars-key", default="tau_hfds")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hist", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--max-concurrent-reads", type=int, default=8)
    args = parser.parse_args()
    data_root = args.data_root.resolve()

    spec = build_om4_spec(
        args.prognostic_vars_key,
        args.boundary_vars_key,
    )
    source = DataSource.from_locations(
        LocalLocation(path=data_root / "data.zarr"),
        LocalLocation(path=data_root / "means.nc"),
        LocalLocation(path=data_root / "stds.nc"),
        dataset_spec=spec,
        prognostic_var_names=spec.prognostic_var_names,
        boundary_var_names=spec.boundary_var_names,
        static_data_vars=None,
        use_dask=False,
    )
    dataset = TorchTrainDataset(
        src=source,
        dst=None,
        prognostic_var_names=spec.prognostic_var_names,
        boundary_var_names=spec.boundary_var_names,
        hist=args.hist,
        steps=args.steps,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=args.stride,
    )
    if args.batch_size > len(dataset):
        raise ValueError(
            f"batch size {args.batch_size} exceeds dataset length {len(dataset)}"
        )
    indices = np.linspace(
        0, len(dataset) - 1, num=args.batch_size, dtype=np.int64
    ).tolist()
    rust = RustBatchDataset(dataset, max_concurrent_reads=args.max_concurrent_reads)

    def python_load():
        return collate_raw_train_data([dataset[index] for index in indices])

    def rust_load():
        return rust.load_batch(indices)

    python_load()
    rust_load()
    python_times: list[float] = []
    rust_times: list[float] = []
    expected = actual = None
    for iteration in range(args.iterations):
        if iteration % 2:
            rust_time, actual = timed(rust_load)
            python_time, expected = timed(python_load)
        else:
            python_time, expected = timed(python_load)
            rust_time, actual = timed(rust_load)
        python_times.append(python_time)
        rust_times.append(rust_time)

    assert expected is not None and actual is not None
    for actual_step, expected_step in zip(actual.raw_data, expected.raw_data):
        for actual_tensor, expected_tensor in zip(actual_step, expected_step):
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                rtol=0,
                atol=0,
                equal_nan=True,
            )

    python_median = statistics.median(python_times)
    rust_median = statistics.median(rust_times)
    print(f"layout={'compact' if source.is_compact else 'flat'}")
    print(f"batch_size={args.batch_size}")
    print(f"python_median_ms={python_median * 1000:.3f}")
    print(f"rust_median_ms={rust_median * 1000:.3f}")
    print(f"speedup={python_median / rust_median:.3f}x")
    print(f"python_p95_ms={np.percentile(python_times, 95) * 1000:.3f}")
    print(f"rust_p95_ms={np.percentile(rust_times, 95) * 1000:.3f}")


if __name__ == "__main__":
    main()
