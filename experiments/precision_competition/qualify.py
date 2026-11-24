# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Run representative GPU cases before releasing the first-wave array."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from .benchmark import Config, atomic_json, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU qualification requires CUDA")
    base = Config(
        steps=30, eval_every=30, validation_size=128, test_size=128, train_seconds=180
    )
    cases = [
        base,
        replace(base, width=512, heads=8),
        replace(base, width=256, heads=4, depth=16),
        replace(base, precision="fp64"),
        replace(base, precision="bf16"),
        replace(base, objective="ce", bins=4096),
        replace(
            base,
            main="forecast",
            auxiliary="evolution",
            path="residual",
            auxiliary_weight=1.0,
        ),
    ]
    receipts = []
    for config in cases:
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        result = train(config, args.output / config.run_id, "cuda", "disabled")
        if result.get("interrupted"):
            raise SystemExit(75)
        receipts.append(
            {
                "id": config.run_id,
                "config": result["config"],
                "step": result["step"],
                "seconds_per_update": result["training_seconds"] / result["step"],
                "wall_seconds": time.monotonic() - started,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "device": torch.cuda.get_device_name(),
                "test": result["test"],
            }
        )
        print(json.dumps({"qualification_case": receipts[-1]}), flush=True)
    atomic_json(args.output / "qualification.json", receipts)
    print("QUALIFICATION_PASSED", flush=True)


if __name__ == "__main__":
    main()
