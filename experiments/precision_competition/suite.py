# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Predeclared staged screen; each joint arm has an exact solo counterpart."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from .benchmark import Config


def make_suite(seeds=(11, 22, 33)):
    entries = {}

    def add(config, family):
        key = config.run_id
        if key not in entries:
            entries[key] = {"id": key, "families": [], "config": asdict(config)}
        if family not in entries[key]["families"]:
            entries[key]["families"].append(family)

    def pair(config, family, dependent=True):
        add(
            replace(
                config, auxiliary_weight=0.0, relation="independent", auxiliary_start=0
            ),
            family,
        )
        add(replace(config, auxiliary_weight=1.0, relation="independent"), family)
        if dependent:
            add(replace(config, auxiliary_weight=1.0, relation="dependent"), family)

    for seed in seeds:
        base = Config(seed=seed)
        add(replace(base, main_weight=0.0, auxiliary_weight=1.0), "auxiliary_alone")
        # Fixed head-width growth, fixed total-width repartition, and depth.
        for width, depth, heads in [
            (128, 4, 2),
            (256, 4, 4),
            (512, 4, 8),
            (256, 2, 4),
            (256, 8, 4),
            (256, 16, 4),
            (256, 4, 2),
            (256, 4, 8),
        ]:
            pair(replace(base, width=width, depth=depth, heads=heads), "capacity")
        for main in ("affine", "nonlinear"):
            pair(replace(base, main=main), "copy_vs_calculation")
        for alpha in (0.1, 0.01, 0.001):
            pair(
                replace(base, main="forecast", auxiliary="evolution", alpha=alpha),
                "coupled_forecast",
            )
        for bins in (16, 64, 256, 1024, 4096):
            pair(
                replace(base, objective="ce", bins=bins), "output_bins", dependent=False
            )
        for objective in ("quantized_mse", "tolerance"):
            pair(
                replace(base, objective=objective, bins=256),
                "bounded_precision",
                dependent=False,
            )
        for precision in ("fp64", "bf16"):
            pair(replace(base, precision=precision), "numerics", dependent=False)
        for weight in (0.1, 10.0):
            add(replace(base, auxiliary_weight=weight), "loss_weight")
        for path in ("retrieval", "residual"):
            pair(replace(base, path=path), "copy_path", dependent=False)
        pair(replace(base, filter="before"), "filter_order")
        pair(replace(base, velocity_fine=0.0), "velocity_bandwidth")
        pair(replace(base, final_norm=False), "final_normalization", dependent=False)
        add(replace(base, auxiliary_weight=1.0, auxiliary_start=1000), "switch_on")
        # Zero-block learned embedding/readout is the inexpensive local control.
        add(replace(base, depth=0), "zero_block")
    add(replace(Config(), path="wire", steps=0), "exact_wire")
    return list(entries.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    entries = make_suite()
    # IDs are derived from config at runtime; don't duplicate opaque hash strings
    # throughout the human-reviewable manifest.
    serialized = [{k: v for k, v in entry.items() if k != "id"} for entry in entries]
    args.output.write_text(json.dumps(serialized, indent=2) + "\n")
    print(
        json.dumps(
            {
                "runs": len(entries),
                "seeds": [11, 22, 33],
                "maximum_training_gpu_hours": sum(
                    e["config"]["train_seconds"]
                    for e in entries
                    if e["config"]["path"] != "wire"
                )
                / 3600,
                "maximum_steps": max(e["config"]["steps"] for e in entries),
            }
        )
    )


if __name__ == "__main__":
    main()
