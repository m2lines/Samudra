# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""GPU smoke qualification of averaging, normalized loss and checkpoint forks."""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path

import torch

from .benchmark import atomic_json
from .followup import FollowupConfig, train
from .followup_suite import checkpoint_digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scales", type=Path, required=True)
    args = parser.parse_args()
    scale = json.loads(args.scales.read_text())
    c = FollowupConfig(steps=30, eval_every=30, validation_size=128, test_size=128)
    configs = [
        c,
        replace(
            c, main_weight=0.0, auxiliary_weight=1.0, aux_loss_scale=scale["transport"]
        ),
    ]
    receipts = []
    for config in configs:
        result = train(config, args.output / config.run_id)
        receipts.append(result)
    path = args.output / c.run_id / "latest.pt"
    fork = replace(
        c,
        stage="competition",
        auxiliary_weight=1.0,
        schedule="cosine",
        parent_checkpoint=str(path.resolve()),
        parent_sha256=checkpoint_digest(path),
        main_loss_scale=scale["main"],
        aux_loss_scale=scale["transport"],
    )
    receipts.append(train(fork, args.output / fork.run_id))
    for result in receipts:
        for metrics in [result["test"], *result["ema_test"].values()]:
            assert math.isfinite(metrics["rmse"]) and math.isfinite(
                metrics["auxiliary_rmse"]
            )
    summary = {
        "cases": len(receipts),
        "finite": True,
        "device": torch.cuda.get_device_name(),
        "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "training_seconds": [r["training_seconds"] for r in receipts],
    }
    atomic_json(args.output / "qualification.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
