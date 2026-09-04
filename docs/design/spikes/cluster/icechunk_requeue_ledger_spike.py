# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise a persisted completion ledger across a real Slurm requeue."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pickle
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import icechunk as ic  # type: ignore[import-not-found]
import numpy as np
import zarr  # type: ignore[import-untyped]


@contextmanager
def locked_ledger(root: Path) -> Iterator[dict[str, Any]]:
    lock_path = root / "ledger.lock"
    ledger_path = root / "ledger.json"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
        yield ledger
        temporary = ledger_path.with_suffix(".json.new")
        temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
        temporary.replace(ledger_path)


def claim(root: Path, task: str, owner: str) -> bool:
    with locked_ledger(root) as ledger:
        state = ledger["tasks"][task]
        if state["status"] != "pending":
            return False
        state.update(status="claimed", owner=owner)
        return True


def complete(root: Path, task: str, owner: str, artifact: Path) -> None:
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    with locked_ledger(root) as ledger:
        state = ledger["tasks"][task]
        if state != {"status": "claimed", "owner": owner}:
            raise RuntimeError(f"invalid completion state for {task}: {state}")
        state.update(status="complete", sha256=digest, artifact=str(artifact))


def write_artifact(root: Path, task_index: int, owner: str) -> None:
    task = str(task_index)
    if not claim(root, task, owner):
        raise RuntimeError(f"could not claim task {task}")
    artifact = root / "artifacts" / f"task-{task}.npy"
    artifact.parent.mkdir(exist_ok=True)
    np.save(artifact, np.full((4, 24, 24), task_index + 1, dtype="float32"))
    complete(root, task, owner, artifact)


def repository(root: Path) -> ic.Repository:
    storage = ic.local_filesystem_storage(str(root / "repository"))
    return ic.Repository.open(storage)


def first_attempt(root: Path, owner: str) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    with locked_ledger(root) as ledger:
        if ledger:
            raise RuntimeError("first attempt found an existing ledger")
        ledger.update(
            tasks={str(index): {"status": "pending"} for index in range(4)},
            attempts=[owner],
        )

    storage = ic.local_filesystem_storage(str(root / "repository"))
    repo = ic.Repository.create(storage)
    repo.save_config()
    session = repo.writable_session("main")
    array = zarr.create_array(
        session.store,
        name="field",
        shape=(4, 4, 24, 24),
        chunks=(1, 4, 12, 12),
        shards=(1, 4, 24, 24),
        dtype="float32",
        dimension_names=("time", "k", "j", "i"),
    )
    array[:] = np.float32(0)
    initial_snapshot = session.commit("initial zeros")
    with locked_ledger(root) as ledger:
        ledger["initial_snapshot"] = str(initial_snapshot)

    write_artifact(root, 0, owner)
    duplicate_claim_rejected = not claim(root, "0", "duplicate-worker")
    if not claim(root, "1", owner):
        raise RuntimeError("could not create a stale in-progress claim")

    dead_coordinator = repo.writable_session("main")
    dead_array = zarr.open_array(dead_coordinator.store, path="field", mode="r+")
    dead_array[0] = np.load(root / "artifacts" / "task-0.npy")
    (root / "dead-coordinator.pickle").write_bytes(
        pickle.dumps(dead_coordinator.fork())
    )
    visible = zarr.open_array(
        repo.readonly_session("main").store, path="field", mode="r"
    )[:]
    if np.any(visible):
        raise AssertionError("uncommitted coordinator changes became visible")
    return {
        "attempt": 0,
        "duplicate_active_claim_rejected": duplicate_claim_rejected,
        "uncommitted_coordinator_invisible": True,
        "request_requeue": True,
    }


def resumed_attempt(root: Path, owner: str) -> dict[str, Any]:
    repo = repository(root)
    visible = zarr.open_array(
        repo.readonly_session("main").store, path="field", mode="r"
    )[:]
    if np.any(visible):
        raise AssertionError("dead coordinator changed the branch head")

    with locked_ledger(root) as ledger:
        ledger["attempts"].append(owner)
        recovered = []
        for task, state in ledger["tasks"].items():
            if state["status"] == "claimed" and state["owner"] != owner:
                state.clear()
                state["status"] = "pending"
                recovered.append(task)

    for task_index in range(1, 4):
        write_artifact(root, task_index, owner)
    duplicate_complete_claim_rejected = not claim(root, "0", owner)

    coordinator = repo.writable_session("main")
    output = zarr.open_array(coordinator.store, path="field", mode="r+")
    with locked_ledger(root) as ledger:
        for task_index in range(4):
            state = ledger["tasks"][str(task_index)]
            if state["status"] != "complete":
                raise AssertionError(f"task {task_index} is not complete")
            artifact = Path(state["artifact"])
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != state["sha256"]:
                raise AssertionError(f"artifact checksum mismatch for {task_index}")
            output[task_index] = np.load(artifact)
    snapshot = coordinator.commit("reconstructed from completion ledger")
    actual = zarr.open_array(
        repo.readonly_session(snapshot_id=snapshot).store, path="field", mode="r"
    )[:]
    expected = np.stack(
        [np.full((4, 24, 24), index + 1, dtype="float32") for index in range(4)]
    )
    np.testing.assert_equal(actual, expected)
    return {
        "attempt": int(os.environ.get("SLURM_RESTART_COUNT", "1")),
        "stale_claims_recovered": recovered,
        "duplicate_complete_claim_rejected": duplicate_complete_claim_rejected,
        "dead_coordinator_snapshot_invisible": True,
        "reconstructed_commit_exact": True,
        "release_snapshot": str(snapshot),
        "ledger": json.loads((root / "ledger.json").read_text()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--attempt", type=int, required=True)
    args = parser.parse_args()
    owner = f"{os.environ.get('SLURM_JOB_ID', 'local')}:{args.attempt}"
    result = (
        first_attempt(args.root, owner)
        if args.attempt == 0
        else resumed_attempt(args.root, owner)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
