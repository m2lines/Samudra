#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify fixed monthly endpoints and plot longer-training/replay comparisons."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    normalization = json.loads(a.normalization.read_text())["normalization"]
    curves, endpoints, exposure, inputs = {}, {}, {}, {}
    origin_cohort = None
    for name in ["A", "E", "A-extend", "E-extend", "A-replay", "E-replay"]:
        original = name in ["A", "E"]
        directory = a.original / name if original else a.root / "production" / name
        events_path = directory / "events.jsonl"
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        end = 5000 if "extend" in name else 3000
        first = 2001 if "extend" in name else 1
        if [e["step"] for e in events] != list(range(first, end + 1)):
            raise ValueError("Missing or duplicate updates: " + name)
        complete_path = directory / "TRAIN_COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        expected_producer = "1c9817d44" if original else "ed020de4b"
        if complete["step"] != end or complete["producer"] != expected_producer:
            raise ValueError("Endpoint producer or step differs: " + name)
        inputs[str(events_path)] = hashlib.sha256(events_path.read_bytes()).hexdigest()
        records = []
        for event in events:
            if "validation_rmse_blend" not in event:
                continue
            path = directory / f"validation-{event['step']:05d}.json"
            payload = json.loads(path.read_text())
            if payload["members"] != 8 or payload["sampler_steps"] != 32:
                raise ValueError("Different diffusion readout")
            if origin_cohort is None:
                origin_cohort = payload["origins"]
            if payload["origins"] != origin_cohort or len(origin_cohort) != 9:
                raise ValueError("Different validation origins")
            metrics = payload["metrics"]
            reconstructed = float(
                np.mean([metrics[k] / normalization[k] for k in normalization])
            )
            if not np.isclose(
                reconstructed, event["validation_rmse_blend"], atol=1e-12, rtol=0
            ):
                raise ValueError("Fixed climatology normalization differs")
            records.append(
                dict(
                    step=event["step"],
                    blend=reconstructed,
                    surface_blend=float(
                        np.mean(
                            [
                                metrics[k] / normalization[k]
                                for k in ["sst_rmse", "velocity_rmse", "eke_rmse"]
                            ]
                        )
                    ),
                    metrics=metrics,
                )
            )
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if len(records) != 12 or records[-1]["step"] != end:
            raise ValueError("Incomplete validation curve")
        curves[name] = records
        endpoints[name] = dict(
            **records[-1],
            raw_blend=events[-1].get("validation_raw_rmse_blend"),
            counts=complete["counts"],
            selected_step=min(records, key=lambda r: r["blend"])["step"],
            wandb=(directory / "wandb-url.txt").read_text().strip(),
        )
        if not original:
            obs = [e for e in events if e["source"] == "obs"]
            exposure[name] = dict(
                fresh_examples=complete["fresh_examples"],
                replay_examples=complete["replay_examples"],
                maximum_start_lead_days=max(
                    e["replay_start_lead_days_max"] for e in obs
                ),
                maximum_age_updates=max(e["replay_age_max"] for e in obs),
                observation_curve=[
                    {
                        k: e[k]
                        for k in [
                            "step",
                            "replay_start_lead_days_mean",
                            "replay_start_lead_days_max",
                            "replay_age_max",
                            "latent_rms_mean",
                        ]
                    }
                    for e in obs[::25]
                ],
            )
    colors = dict(
        A="#1f77b4",
        E="#ff7f0e",
        **{
            "A-extend": "#2ca02c",
            "E-extend": "#d62728",
            "A-replay": "#9467bd",
            "E-replay": "#8c564b",
        },
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fields = ["blend", "surface_blend", "ohc_0_700_rmse", "ohc_700_2000_rmse"]
    titles = [
        "Five-component RMSE blend",
        "Surface-only RMSE blend",
        "Monthly OHC 0–700 m RMSE (GJ/m²)",
        "Monthly OHC 700–2,000 m RMSE (GJ/m²)",
    ]
    for ax, field, title in zip(axes.flat, fields, titles, strict=True):
        for name, records in curves.items():
            if "extend" in name:
                records = curves[name[0]][:8] + records
            ys = [
                r[field] if field in r else r["metrics"][field] / 1e9 for r in records
            ]
            ax.plot(
                [r["step"] for r in records],
                ys,
                label=name + (" (2k+1k)" if name in ["A", "E"] else ""),
                color=colors[name],
                linestyle="--" if name in ["A", "E"] else "-",
                marker=".",
                linewidth=1.5,
            )
        ax.set_title(title)
        ax.set_xlabel("Optimizer updates; cooldown starts at 2k or 4k")
        ax.set_xlim(1500, 5000)
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylim(1, 1.6)
    axes[0, 1].set_ylim(0.85, 1.1)
    axes[1, 0].set_ylim(0.65, 1.4)
    axes[1, 1].set_ylim(0.5, 1.1)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "validation-curves.png", dpi=140)
    plt.close(fig)
    result = dict(
        normalization=normalization,
        origins=origin_cohort,
        curves=curves,
        endpoints=endpoints,
        exposure=exposure,
        checksum_file="checksums.txt",
        qualification_gpu_hours=1759 / 3600,
        production_gpu_hours=22609 * 4 / 3600,
    )
    (a.out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    inputs.update(json.loads((a.root / "ENDPOINT_SHA256.json").read_text()))
    (a.out / "checksums.txt").write_text(
        "".join(
            f"{sha}  {path}  # pragma: allowlist secret\n"
            for path, sha in sorted(inputs.items())
        )
    )
    for name, endpoint in endpoints.items():
        print(name, json.dumps(endpoint))


if __name__ == "__main__":
    main()
