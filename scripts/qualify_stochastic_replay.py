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
    for arm in ("deterministic", "initial", "fresh"):
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
        if not saved["contract"]["replay"] or saved["replay_examples"] <= 0:
            raise ValueError(f"Replay did not run: {arm}")
        pool = saved["replay_pool"]
        if (pool["per_lead"], pool["max_age"], pool["max_months"]) != (8, 250, 11):
            raise ValueError(f"Replay policy changed: {arm}")
        if not saved["replay_rng"] or not any(pool["entries"].values()):
            raise ValueError(f"Missing serialized replay state: {arm}")
        for entries in pool["entries"].values():
            for entry in entries:
                age = saved["step"] - entry["origin_step"]
                if not 0 <= age <= 250 or entry["latent"].ndim != 6:
                    raise ValueError(f"Invalid replay state age or shape: {arm}")
        ages = [age for event in events for age in event["replay_age_updates"]]
        if not ages or min(ages) < 0 or max(ages) > 250:
            raise ValueError(f"Replay exposure age out of bounds: {arm}")
        if saved["native_replay_examples"] <= 0 or not saved["native_replay_rng"]:
            raise ValueError(f"Native replay did not run or serialize RNG: {arm}")
        native_pool = saved["native_replay_pool"]
        if (
            native_pool["per_lead"],
            native_pool["max_age"],
            native_pool["max_lead"],
            native_pool["chunk"],
        ) != (8, 250, 72, 6):
            raise ValueError(f"Native replay policy changed: {arm}")
        if not any(native_pool["entries"].values()):
            raise ValueError(f"Missing serialized native replay state: {arm}")
        for entries in native_pool["entries"].values():
            for entry in entries:
                age = saved["step"] - entry["origin_step"]
                if not 0 <= age <= 250 or entry["latent"].ndim != 6:
                    raise ValueError(f"Invalid native replay state age or shape: {arm}")
        native_ages = [
            age for event in events for age in event["native_replay_age_updates"]
        ]
        if not native_ages or min(native_ages) < 0 or max(native_ages) > 250:
            raise ValueError(f"Native replay exposure age out of bounds: {arm}")
        longlead = json.loads((output / "replay-longlead.json").read_text())
        if longlead["max_end_days"] < 365 or longlead["observed_ocean_resets"]:
            raise ValueError(f"Long-lead qualification failed: {arm}")
        if len(longlead["windows"]) != 11 or not all(
            np.isfinite(window["loss"]) for window in longlead["windows"]
        ):
            raise ValueError(f"Invalid long-lead losses: {arm}")
        if longlead["native_max_end_days"] < 360:
            raise ValueError(f"Native long-lead qualification failed: {arm}")
        if len(longlead["native_windows"]) != 11 or not all(
            np.isfinite(window["loss"]) for window in longlead["native_windows"]
        ):
            raise ValueError(f"Invalid native long-lead losses: {arm}")
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
            replay_examples=saved["replay_examples"],
            native_replay_examples=saved["native_replay_examples"],
            native_replay_max_end_days=longlead["native_max_end_days"],
            fresh_examples=saved["fresh_examples"],
            replay_max_end_days=longlead["max_end_days"],
            longlead_sha256=digest(output / "replay-longlead.json"),
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
