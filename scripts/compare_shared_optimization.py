#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Frozen engineering gates for the shared-ERA5 optimization migration."""

import argparse
import json
from pathlib import Path

import torch

from samudra.experiments.observation_metrics import PROTOCOL


def norm_difference(left, right):
    numerator = sum(
        (left[key].float() - right[key].float()).square().sum().item() for key in left
    )
    denominator = sum(right[key].float().square().sum().item() for key in left)
    return (numerator / max(denominator, 1e-30)) ** 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--output", default="COMPARISON.json")
    args = parser.parse_args()
    base = (
        args.baseline_root or args.root / "baseline"
    ) / "qualification/A-shared-era5-8k"
    opt = (
        args.candidate_root or args.root / "optimized"
    ) / "qualification/A-shared-era5-8k"
    left = json.loads((base / "QUALIFIED.json").read_text())
    right = json.loads((opt / "QUALIFIED.json").read_text())
    records = []
    traces_exact = True
    for a, b in zip(left["records"], right["records"], strict=True):
        exact = all(
            a[key] == b[key]
            for key in (
                "source",
                "sample_trace",
                "draw_trace",
                "generator_sha256",
                "obs_updates",
                "model_updates",
            )
        )
        traces_exact &= exact
        records.append(
            dict(
                step=a["step"],
                source=a["source"],
                traces_exact=exact,
                loss_original=a["loss"],
                loss_optimized=b["loss"],
                loss_relative=abs(a["loss"] - b["loss"]) / abs(a["loss"]),
                seconds_original=a["seconds"],
                seconds_optimized=b["seconds"],
            )
        )
    a = torch.load(base / "first-gradients.pt", weights_only=True)
    b = torch.load(opt / "first-gradients.pt", weights_only=True)
    grad = norm_difference(b, a)
    del a, b
    a = torch.load(base / "first-update.pt", weights_only=False, map_location="cpu")
    b = torch.load(opt / "first-update.pt", weights_only=False, map_location="cpu")
    # Reconstruct the original fresh initializer exactly, without data/model work.
    from samudra.experiments.monthly_latent import MonthlyLatentOcean

    torch.manual_seed(1729)
    initial = MonthlyLatentOcean().state_dict()
    delta = {key: a["model"][key] - initial[key] for key in initial}
    difference = {key: b["model"][key] - a["model"][key] for key in initial}
    update = (
        sum(v.float().square().sum().item() for v in difference.values())
        / sum(v.float().square().sum().item() for v in delta.values())
    ) ** 0.5
    del a, b, initial, delta, difference
    validation = {}
    validation_ok = True
    for weights in ("raw", "ema"):
        checks = {}
        for key in PROTOCOL["integrated"]:
            old = left["validation_" + weights]["metrics"][key]
            new = right["validation_" + weights]["metrics"][key]
            relative = abs(new - old) / max(abs(old), 1e-30)
            checks[key] = dict(
                original=old,
                optimized=new,
                absolute_difference=new - old,
                relative_difference=relative,
            )
            validation_ok &= relative <= 0.01
        validation[weights] = checks
    passed = (
        traces_exact
        and grad < 0.01
        and update < 0.05
        and max(r["loss_relative"] for r in records) < 0.02
        and validation_ok
        and left["exact_serialized_update"]
        and right["exact_serialized_update"]
    )
    result = dict(
        passed=bool(passed),
        traces_exact=traces_exact,
        first_gradient_relative_l2=grad,
        first_update_relative_l2=update,
        records=records,
        validation=validation,
        exact_serialized_update_original=left["exact_serialized_update"],
        exact_serialized_update_optimized=right["exact_serialized_update"],
        peak_cuda_original=left["peak_cuda_bytes"],
        peak_cuda_optimized=right["peak_cuda_bytes"],
        gate="exact draws; gradient<1%; update<5%; all16loss<2%; each5RMSEcomponent raw/EMA<=1%; exact serialization",
    )
    (args.root / args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if not passed:
        raise RuntimeError("Optimization engineering gate failed; do not promote")


if __name__ == "__main__":
    main()
