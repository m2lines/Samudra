# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise Icechunk snapshot isolation and conflicting filesystem commits.

Run with:

    uv run --isolated --with 'icechunk==2.2.0' --with 'zarr==3.3.0' \
        --with numpy python docs/design/spikes/icechunk_recovery_spike.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import zarr  # type: ignore[import-untyped]
from zarr.codecs import BloscCodec  # type: ignore[import-untyped]


def open_repo(path: Path):  # noqa: ANN202
    return ic.Repository.open(ic.local_filesystem_storage(str(path)))


def child_write(repo_path: Path, marker: Path) -> None:
    repo = open_repo(repo_path)
    session = repo.writable_session("main")
    array = zarr.open_array(session.store, path="field", mode="r+")
    array[0, 0, 0, 0] = np.float32(2)
    marker.write_text("uncommitted shard written")
    time.sleep(600)


def values(repo, *, snapshot_id=None) -> tuple[float, float]:  # noqa: ANN001, ANN202
    session = (
        repo.readonly_session(snapshot_id=snapshot_id)
        if snapshot_id is not None
        else repo.readonly_session("main")
    )
    array = zarr.open_array(session.store, path="field", mode="r")
    return float(array[0, 0, 0, 0]), float(array[1, 0, 0, 0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--marker", type=Path)
    parser.add_argument("--temporary-root", type=Path)
    args = parser.parse_args()
    if args.child:
        if args.repo is None or args.marker is None:
            parser.error("--child requires --repo and --marker")
        child_write(args.repo, args.marker)
        return

    with tempfile.TemporaryDirectory(dir=args.temporary_root) as temporary:
        root = Path(temporary)
        repo_path = root / "repo"
        marker = root / "child-ready"
        repo = ic.Repository.create(ic.local_filesystem_storage(str(repo_path)))
        session = repo.writable_session("main")
        array = zarr.create_array(
            store=session.store,
            name="field",
            shape=(2, 4, 216, 216),
            chunks=(1, 4, 18, 18),
            shards=(1, 4, 72, 72),
            dtype="float32",
            compressors=[BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle")],
        )
        array[:] = np.float32(1)
        initial_snapshot = session.commit("initial")

        process = subprocess.Popen(
            [
                sys.executable,
                __file__,
                "--child",
                "--repo",
                str(repo_path),
                "--marker",
                str(marker),
            ]
        )
        deadline = time.monotonic() + 30
        while not marker.exists() and process.poll() is None:
            if time.monotonic() > deadline:
                process.kill()
                raise TimeoutError("child did not finish its uncommitted write")
            time.sleep(0.05)
        if process.poll() is not None:
            raise RuntimeError(f"child exited unexpectedly with {process.returncode}")
        process.terminate()
        process.wait(timeout=10)

        repo = open_repo(repo_path)
        assert values(repo) == (1.0, 1.0)
        assert values(repo, snapshot_id=initial_snapshot) == (1.0, 1.0)

        retry = repo.writable_session("main")
        retry_array = zarr.open_array(retry.store, path="field", mode="r+")
        retry_array[0, 0, 0, 0] = np.float32(2)
        retry_snapshot = retry.commit("retry killed write")
        assert values(repo) == (2.0, 1.0)
        assert values(repo, snapshot_id=initial_snapshot) == (1.0, 1.0)

        first = repo.writable_session("main")
        second = repo.writable_session("main")
        zarr.open_array(first.store, path="field", mode="r+")[0, 0, 0, 0] = 3
        zarr.open_array(second.store, path="field", mode="r+")[1, 0, 0, 0] = 4
        first_snapshot = first.commit("first concurrent session")
        conflict_type = None
        try:
            second.commit("stale concurrent session")
        except Exception as error:  # noqa: BLE001
            conflict_type = type(error).__name__
        if conflict_type is None:
            raise AssertionError("stale filesystem session committed without conflict")
        assert values(repo) == (3.0, 1.0)

        print(
            json.dumps(
                {
                    "icechunk": ic.__version__,
                    "initial_snapshot": str(initial_snapshot),
                    "retry_snapshot": str(retry_snapshot),
                    "first_concurrent_snapshot": str(first_snapshot),
                    "killed_uncommitted_write_invisible": True,
                    "initial_snapshot_immutable": True,
                    "retry_succeeded": True,
                    "stale_commit_conflict": conflict_type,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
