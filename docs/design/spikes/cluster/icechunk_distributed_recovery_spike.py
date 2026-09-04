# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise fork/merge, retry, split manifests, and GC on Icechunk storage."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import zarr  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec  # type: ignore[import-untyped]


def worker(input_path: Path, output_path: Path, time_index: int, fail: bool) -> None:
    session = pickle.loads(input_path.read_bytes())
    array = zarr.open_array(session.store, path="field", mode="r+")
    array[time_index] = np.float32(time_index + 1)
    if fail:
        raise OSError("simulated capacity failure after shard write")
    output_path.write_bytes(pickle.dumps(session))


def run_worker(
    script: Path,
    fork_path: Path,
    result_path: Path,
    time_index: int,
    *,
    fail: bool = False,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(script),
        "--worker",
        "--input",
        str(fork_path),
        "--output",
        str(result_path),
        "--time-index",
        str(time_index),
    ]
    if fail:
        command.append("--fail")
    return subprocess.run(command, check=False, text=True, capture_output=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--time-index", type=int)
    parser.add_argument("--fail", action="store_true")
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.input is None or args.output is None or args.time_index is None:
            parser.error("worker mode requires input, output, and time index")
        worker(args.input, args.output, args.time_index, args.fail)
        return

    split_config = ic.ManifestSplittingConfig.from_dict(
        {
            ic.ManifestSplitCondition.AnyArray(): {
                ic.ManifestSplitDimCondition.DimensionName("time"): 2
            }
        }
    )
    config = ic.RepositoryConfig(
        manifest=ic.ManifestConfig(splitting=split_config),
    )
    with tempfile.TemporaryDirectory(dir=args.temporary_root) as temporary:
        root = Path(temporary)
        storage = ic.local_filesystem_storage(str(root / "repo"))
        repo = ic.Repository.create(storage, config=config)
        repo.save_config()
        initial = repo.writable_session("main")
        array = zarr.create_array(
            initial.store,
            name="field",
            shape=(8, 4, 72, 72),
            chunks=(1, 4, 18, 18),
            shards=(1, 4, 72, 72),
            dtype="float32",
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
            dimension_names=("time", "k", "j", "i"),
        )
        array[:] = np.float32(0)
        initial_snapshot = initial.commit("initial zeros")

        coordinator = repo.writable_session("main")
        result_paths = []
        failed_worker_invisible = False
        script = Path(__file__).resolve()
        for time_index in range(8):
            fork_path = root / f"fork-{time_index}.pickle"
            result_path = root / f"result-{time_index}.pickle"
            fork_path.write_bytes(pickle.dumps(coordinator.fork()))
            if time_index == 3:
                failed = run_worker(
                    script, fork_path, result_path, time_index, fail=True
                )
                if failed.returncode == 0 or result_path.exists():
                    raise AssertionError("simulated failing worker published a result")
                before_retry = zarr.open_array(
                    repo.readonly_session("main").store, path="field", mode="r"
                )
                failed_worker_invisible = bool(np.all(before_retry[3] == 0))
            completed = run_worker(script, fork_path, result_path, time_index)
            if completed.returncode != 0:
                raise RuntimeError(completed.stderr)
            result_paths.append(result_path)

        forks = [pickle.loads(path.read_bytes()) for path in result_paths]
        coordinator.merge(*forks)
        snapshot = coordinator.commit(
            "merge disjoint worker shards",
            metadata={"completed_shards": [f"field/{index}" for index in range(8)]},
        )
        repo = ic.Repository.open(storage, config=config)
        actual = zarr.open_array(
            repo.readonly_session(snapshot_id=snapshot).store,
            path="field",
            mode="r",
        )[:]
        expected = np.stack(
            [np.full((4, 72, 72), index + 1, dtype="float32") for index in range(8)]
        )
        np.testing.assert_equal(actual, expected)
        initial_values = zarr.open_array(
            repo.readonly_session(snapshot_id=initial_snapshot).store,
            path="field",
            mode="r",
        )[:]
        np.testing.assert_equal(initial_values, np.zeros_like(expected))

        manifests = repo.list_manifest_files(snapshot)
        if len(manifests) < 4:
            raise AssertionError(f"expected split manifests, got {len(manifests)}")
        gc_summary = repo.garbage_collect(
            dt.datetime.now(dt.UTC) + dt.timedelta(seconds=1),
            dry_run=False,
        )
        np.testing.assert_equal(
            zarr.open_array(
                repo.readonly_session(snapshot_id=snapshot).store,
                path="field",
                mode="r",
            )[:],
            expected,
        )
        print(
            json.dumps(
                {
                    "icechunk": ic.__version__,
                    "workers": 8,
                    "initial_snapshot": str(initial_snapshot),
                    "merged_snapshot": str(snapshot),
                    "failed_worker_invisible": failed_worker_invisible,
                    "retry_succeeded": True,
                    "fork_merge_exact": True,
                    "initial_snapshot_immutable": True,
                    "manifest_files": len(manifests),
                    "garbage_collection": str(gc_summary),
                    "release_survived_gc": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
