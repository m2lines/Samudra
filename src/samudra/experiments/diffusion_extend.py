# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Create an explicitly budget-extended continuation without changing its parent."""

import argparse
import copy
import json
from pathlib import Path

import torch

from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest


def extend(parent, output, *, updates, hours):
    """Preserve optimizer, RNG, update count and best selection; only extend limits."""
    parent, output = Path(parent), Path(output)
    if output.exists():
        raise FileExistsError("Continuation destination must be new")
    signature = json.loads((parent / "protocol.json").read_text())
    complete = json.loads((parent / "OBSERVATION_COMPLETE.json").read_text())
    hashes = {name: digest(parent / name) for name in ("last.pt", "best.pt")}
    if (
        not complete["state"]["complete"]
        or hashes["best.pt"] != complete["best_checkpoint_sha256"]
    ):
        raise ValueError("Parent observation stage is incomplete or changed")
    last = torch.load(parent / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(parent / "best.pt", map_location="cpu", weights_only=False)
    if (
        last["state"] != complete["state"]
        or last["protocol"] != best["protocol"]
        or last["protocol"]["signature"] != signature
    ):
        raise ValueError("Parent checkpoint and completion contracts differ")
    if updates <= signature["max_updates"] or hours * 3600 <= max(
        signature["max_seconds"], last["state"]["elapsed"]
    ):
        raise ValueError("Both continuation ceilings must increase")
    old_protocol = copy.deepcopy(last["protocol"])
    signature = dict(signature, max_updates=updates, max_seconds=hours * 3600)
    protocol = dict(
        old_protocol, signature=signature, max_updates=updates, max_seconds=hours * 3600
    )
    last["protocol"] = protocol
    best["protocol"] = protocol
    last["state"] = dict(last["state"], complete=False)
    output.mkdir(parents=True)
    atomic_torch(last, output / "last.pt")
    atomic_torch(best, output / "best.pt")
    atomic_json(signature, output / "protocol.json")
    receipt = dict(
        parent=str(parent.resolve()),
        parent_hashes=hashes,
        parent_protocol=old_protocol,
        continuation_protocol=protocol,
        initial_state=last["state"],
        prepared_hashes={name: digest(output / name) for name in hashes},
        semantics="Budget migration only: weights, optimizer, RNG, update count, elapsed fitting time and incumbent best score preserved. Parent files untouched. Resume uses last weights, selection retains best weights.",
    )
    atomic_json(receipt, output / "CONTINUATION.json")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--hours", type=float, required=True)
    args = parser.parse_args()
    result = extend(args.parent, args.output, updates=args.updates, hours=args.hours)
    print(
        json.dumps(
            dict(
                step=result["initial_state"]["step"],
                best=result["initial_state"]["best"],
                elapsed=result["initial_state"]["elapsed"],
            )
        )
    )


if __name__ == "__main__":
    main()
