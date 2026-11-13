# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Prepare a bounded, training-only support audit and six discrete pilot configs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .data import CHANNELS
from .multitask import ComparisonSource


def prepare(template, output):
    output.mkdir(parents=True, exist_ok=True)
    source = ComparisonSource(
        template["source"], 4, template.get("reference_root"), arm="all_fields"
    )
    try:
        origins = source.origins["train"]
        times = origins[np.linspace(0, len(origins) - 1, 32, dtype=int)] + 1
        extrema = []
        for c, name in enumerate(CHANNELS[:77]):
            mask = source.channel_wet[c]
            bound = 0.0
            for t in times:
                values = np.asarray(source.zarr[name][int(t)], dtype=np.float32)[mask]
                normalized = (values - source.mean[c]) / source.std[c]
                if not np.isfinite(normalized).all():
                    raise ValueError(f"Nonfinite training support: {name}/{t}")
                bound = max(bound, float(np.abs(normalized).max()))
            extrema.append(bound)
            print(json.dumps(dict(channel=name, max_abs=bound)), flush=True)
        bounds = np.maximum(
            template["vocabulary_bounds"], np.asarray(extrema) * 1.1
        ).tolist()
        receipt = dict(
            source=source.receipt(),
            target_indices=times.tolist(),
            dates=source.date_strings[times].tolist(),
            max_abs=extrema,
            bounds=bounds,
            scope="32 training-only dates, all wet cells; 10% margin; not exhaustive temporal support",
        )
        (output / "support.json").write_text(json.dumps(receipt, indent=2) + "\n")
    finally:
        source.close()
    for decoder, bins, coarse in (
        ("flat2048", 2048, 0),
        ("flat8192", 8192, 0),
        ("coarse128-fine64", 8192, 128),
    ):
        for task in ("temperature", "all-fields"):
            config = dict(template)
            config.update(
                arm="all_fields",
                temperature_loss_only=task == "temperature",
                discrete_readout_only=True,
                crop=17,
                world_size=1,
                microbatch=32,
                accumulation=8,
                samples_per_frame=2048,
                max_updates=2000,
                max_allocated_seconds=7200,
                schedule_updates=2000,
                warmup=100,
                save_seconds=120,
                validation_dates=4,
                validation_points_per_depth=16,
                validate_updates=500,
                validate_at=[1, 50, 100, 200, 500, 750, 1000],
                vocabulary_bounds=bounds,
                vocabulary_note=receipt["scope"],
                retain_validation_weights=True,
                support_audit="support.json",
                seed=20261110,
            )
            config["model"] = dict(
                template["model"],
                width=512,
                layers=4,
                heads=8,
                bins=bins,
                coarse_bins=coarse,
                checkpoint_blocks=False,
            )
            name = f"{task}-{decoder}"
            (output / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
            qualification = dict(
                config,
                max_updates=4,
                schedule_updates=4,
                warmup=1,
                validate_at=[1, 2, 4],
                validation_dates=2,
                validation_points_per_depth=1,
                wandb=False,
                qualification=True,
            )
            (output / f"qual-{name}.json").write_text(
                json.dumps(qualification, indent=2) + "\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.template.read_text()), args.output)


if __name__ == "__main__":
    main()
