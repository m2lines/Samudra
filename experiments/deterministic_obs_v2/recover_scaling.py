# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Audit completed single-rank checkpoints after receipt-only finalization failure."""

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from samudra.observations.qualify import equal_tree
from samudra.observations.qualify_scaling import state_dict_digest
from samudra.observations.shared_era5 import digest


def recover(output):
    records = [
        json.loads(line) for line in (output / "scaling.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((output / "manifest.json").read_text())

    def read(name):
        saved = torch.load(
            output / name, map_location="cpu", weights_only=False, mmap=True
        )
        equal_tree(saved["manifest"], manifest)
        if len(saved["rank_rng"]) != 1:
            raise ValueError("CPU recovery supports only the single-rank receipt")
        return saved

    expected = read("async/ckpt.pt")
    for name in ("sync", "sync-repeat"):
        actual = read(f"{name}/ckpt.pt")
        equal_tree(actual, expected)
        del actual
    del expected
    gc.collect()
    main = read("steady/main.pt")
    endpoint = read("cooldown/endpoint.pt")
    if (
        main["updates"] != 128
        or main["cooled"]
        or int(main["ema"]["num_updates"]) != 128
    ):
        raise ValueError("Uncooled main checkpoint differs")
    if (
        endpoint["updates"] != 130
        or not endpoint["cooled"]
        or int(endpoint["ema"]["num_updates"]) != 2
    ):
        raise ValueError("Cooldown endpoint differs")
    if endpoint["pending_plan"] is not None:
        raise ValueError("Completed endpoint still has a pending update")
    for values in (main["model"], endpoint["model"]):
        if not all(torch.isfinite(value).all() for value in values.values()):
            raise ValueError("Nonfinite checkpoint model value")
    for step in (128, 130):
        validation_rows = [
            r
            for r in records
            if r.get("updates") == step and "timing/validation_seconds" in r
        ]
        if len(validation_rows) != 1:
            raise ValueError("Missing or duplicate annual validation")
        metrics = validation_rows[0]
        for key, value in (
            ("scored_channels", 29),
            ("forecast_steps", 73),
            ("complete_months", 12),
        ):
            if metrics[f"rollout_val/365d/{key}"] != value:
                raise ValueError("Incomplete annual validation")
        if not all(
            math.isfinite(v) for k, v in metrics.items() if k.startswith("rollout_val/")
        ):
            raise ValueError("Nonfinite annual score")
    metadata_paths = list(output.glob("wandb/run-*/files/wandb-metadata.json"))
    if len(metadata_paths) != 1:
        raise ValueError("Ambiguous training hardware metadata")
    hardware = json.loads(metadata_paths[0].read_text())
    report = {
        "purpose": "CPU audit recovered after single-GPU receipt finalization failure",
        "original_job_id": "21805440",
        "original_job_state": "FAILED 1:0 after training and validation",
        "world_size": 1,
        "global_examples": 8,
        "gpu": hardware["gpu"],
        "exact_prefetch_and_resume": True,
        "uncooled_main_checkpoint": True,
        "cooldown_ema_updates": int(endpoint["ema"]["num_updates"]),
        "final_model_sha256": state_dict_digest(endpoint["model"]),
        "exposure": endpoint["exposure"],
        "training_records": records,
        "checkpoint_sha256": {
            name: digest(output / name)
            for name in ("steady/main.pt", "cooldown/endpoint.pt")
        },
        "source_manifest_sha256": digest(output / "manifest.json"),
        "recovery_limits": [
            "Original job failed; this audit does not change its scheduler status",
            "Pre-cooldown main-file hash and framework peak allocator statistics were not persisted",
            "Utilization is assessed separately from the recorded GPU telemetry",
        ],
    }
    (output / "RECOVERED.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "training_records"}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    recover(args.output)
