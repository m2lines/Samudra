#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Inspect completed checkpoint bytes and verify frozen components on CPU."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from samudra.experiments.surface_adaptation import state_fingerprint


def file_record(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024**2):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def audit(directory):
    marker = json.loads((directory / "COMPLETE.json").read_text())
    initial = json.loads((directory / "initialization.json").read_text())
    best_path = directory / "adapt-best.pt"
    last_path = directory / "adapt-last.pt"
    selected = torch.load(best_path, map_location="cpu", weights_only=False)
    if not math.isclose(
        selected["state"]["best"], marker["selected_ts_mse"], rel_tol=1e-6
    ):
        raise ValueError("Selected checkpoint state does not match evaluation marker")
    result = dict(
        selected=file_record(best_path),
        last=file_record(last_path),
        completion=marker,
        audit_script=file_record(Path(__file__)),
    )
    result["selected"]["state"] = selected["state"]
    for component in ["initializer", "evolution"]:
        prefix = component + "."
        weights = {
            name[len(prefix) :]: tensor
            for name, tensor in selected["model"].items()
            if name.startswith(prefix)
        }
        if not weights:
            raise ValueError(f"Missing component {component}")
        result["selected"][component + "_fingerprint"] = state_fingerprint(
            SimpleNamespace(state_dict=lambda: weights)
        )
    frozen = {"initializer": "evolution", "evolution": "initializer"}.get(marker["arm"])
    if frozen is not None:
        if (
            result["selected"][frozen + "_fingerprint"]
            != initial["initial_" + frozen + "_fingerprint"]
        ):
            raise ValueError(f"Selected checkpoint changed frozen {frozen}")
    result["frozen_component_verified"] = frozen
    del selected
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    if not last["state"]["complete"]:
        raise ValueError("Last checkpoint has not completed its training phase")
    result["last"]["state"] = last["state"]
    (directory / "checkpoint-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        directory.name,
        "PASS",
        "selected step",
        result["selected"]["state"]["step"],
        "last step",
        result["last"]["state"]["step"],
        "frozen",
        frozen,
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    for directory in args.directories:
        audit(directory)


if __name__ == "__main__":
    main()
