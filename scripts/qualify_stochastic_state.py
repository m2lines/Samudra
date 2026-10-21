#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify real-data restart, gradients, ensemble exports and matched weights."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.observation_pilot import atomic_json, digest


def fingerprint(state):
    result = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--code-id", required=True)
    args = parser.parse_args()
    records = {}
    initial_hash = None
    for arm in ("deterministic", "initial", "fresh", "ar1"):
        output = args.root / "smoke" / arm
        initial = torch.load(
            output / "step-00000.pt", map_location="cpu", weights_only=False
        )
        current_hash = fingerprint(initial["model"])
        if initial_hash is not None and initial_hash != current_hash:
            raise ValueError("Different initial parameter tensors between arms")
        initial_hash = current_hash
        del initial
        saved = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if saved["step"] != 8 or saved["counts"] != {"om4": 4, "observation": 4}:
            raise ValueError(f"Wrong qualification exposure: {arm}")
        if (
            saved["contract"]["producer"] != args.code_id
            or not saved["optimizer"]["state"]
        ):
            raise ValueError(f"Missing optimizer/source provenance: {arm}")
        events = [
            json.loads(line)
            for line in (output / "events.jsonl").read_text().splitlines()
        ]
        if [event["step"] for event in events] != list(range(1, 9)):
            raise ValueError(f"Restart skipped or repeated an update: {arm}")
        required = ["encoder", "processor", "decoder"]
        if arm != "deterministic":
            required.append("initial_noise")
        if arm in {"fresh", "ar1"}:
            required.append("transition_noise")
        for task in ("observation", "om4"):
            for name in required:
                values = [
                    e["gradient_norms"][name] for e in events if e["task"] == task
                ]
                if not all(np.isfinite(values)) or max(values) <= 0:
                    raise ValueError(f"Missing finite gradient: {arm}/{task}/{name}")
        evaluation = output / "evaluation/screen-obs4"
        receipt = json.loads((evaluation / "COMPLETE.json").read_text())
        if receipt["inputs"]["checkpoint_sha256"] != digest(output / "last.pt"):
            raise ValueError("Evaluation belongs to a different checkpoint")
        for filename, checksum in receipt["files"].items():
            if digest(evaluation / filename) != checksum:
                raise ValueError(f"Corrupt evaluation output: {filename}")
        with np.load(evaluation / "annual/members-2018-01-01.npz") as data:
            members = data["fields"]
            if not np.isfinite(members).all():
                raise ValueError(f"Nonfinite annual rollout: {arm}")
            spread = float(members.std(axis=0).mean())
            equal = np.array_equal(members, np.broadcast_to(members[:1], members.shape))
            if (arm == "deterministic") != equal:
                raise ValueError(f"Unexpected ensemble diversity: {arm}")
        records[arm] = dict(
            counts=saved["counts"],
            checkpoint_sha256=digest(output / "last.pt"),
            annual_mean_pointwise_spread_mixed_units=spread,
            timing_seconds={
                task: float(
                    np.median([e["seconds"] for e in events[2:] if e["task"] == task])
                )
                for task in ("om4", "observation")
            },
        )
        del saved
    atomic_json(
        dict(
            qualified=True,
            producer=args.code_id,
            initial_parameters=initial_hash,
            arms=records,
        ),
        args.root / "QUALIFIED.json",
    )


if __name__ == "__main__":
    main()
