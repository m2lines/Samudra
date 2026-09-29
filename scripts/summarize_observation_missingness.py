#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Build auditable comparison tables and curves from verified metrics bundles."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from samudra.experiments.observation_metrics import selection_score

NAMES = {
    "legacy-scratch": "Small legacy scratch",
    "masked-scratch": "Small masked scratch",
    "masked-sequential": "Small masked sequential",
    "masked-mixed-finish": "Small masked mixed-finish",
    "conditioned-mixed-finish": "Small conditioned mixed-finish",
    "conditioned-mixed": "Small conditioned mixed",
}


def completion_summary(records):
    result: dict[str, Any] = {}
    for pattern in ("natural", "blocks", "polar_caps"):
        selected = [r for r in records if r["pattern"] == pattern]
        result[pattern] = {}
        for method in ("model", "climatology", "normalized_zero"):
            result[pattern][method] = {}
            for key in selected[0][method]:
                values = [r[method][key] for r in selected]
                weight = sum(v["area_weight"] for v in values)
                result[pattern][method][key] = {
                    "count": sum(v["count"] for v in values),
                    "area_weight": weight,
                    "rmse": float(
                        np.sqrt(
                            sum(
                                v["area_weight"] * v["rmse"] ** 2
                                for v in values
                                if v["rmse"] is not None
                            )
                            / weight
                        )
                    )
                    if weight
                    else None,
                    "bias": sum(
                        v["area_weight"] * v["bias"]
                        for v in values
                        if v["bias"] is not None
                    )
                    / weight
                    if weight
                    else None,
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = json.loads(gzip.decompress(args.bundle.read_bytes()))
    files = data["files"]
    result = {
        "bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
        "status": data["status"],
        "models": {},
    }
    common_control = None
    common_keys = None
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for arm, label in NAMES.items():
        if data["status"][arm] != "verified complete":
            continue
        keys = files[arm + "/selection-reference.json"]["spectral_keys"]
        control = files[arm + "-selected-monthly/seasonal-climatology.json"]
        if common_control is None:
            common_control, common_keys = control, keys
        else:
            assert common_keys == keys
            # Controls may have small device-dependent arithmetic differences.
            for k, value in control["metrics"].items():
                np.testing.assert_allclose(
                    value, common_control["metrics"][k], rtol=1e-5
                )
        best = files[arm + "/best.json"]
        rows = {}
        choices = {
            "selected": ("selected-monthly", "selected"),
            "raw_endpoint": ("fixed-endpoint-monthly", "fixed-budget"),
            "persistence": ("selected-monthly", "selected-inferred-persistence"),
            "anomaly_persistence": (
                "selected-monthly",
                "selected-inferred-anomaly-persistence",
            ),
        }
        if "scratch" in arm:
            choices["raw_obs_8000"] = ("fixed8k-monthly", "fixed-budget")
        for name, (directory, stem) in choices.items():
            path = f"{arm}-{directory}/{stem}.json"
            record = files[path]
            metrics = record if "spectra" in record else record["metrics"]
            score = selection_score(metrics, control, keys)
            rows[name] = {
                "composite": score,
                "metrics": metrics["metrics"],
                "source": path,
                "spectral_error_dex": float(
                    np.mean([metrics["spectra"][k]["error_dex"] for k in keys])
                ),
            }
        annual = {}
        for origin in ("2015-01-01", "2018-01-01", "2021-01-01"):
            annual[origin] = files[f"{arm}-selected-annual/{origin}.json"]["leads"][
                "365"
            ]
        validations = sorted(
            (v for k, v in files.items() if k.startswith(arm + "/validation-")),
            key=lambda v: v["global_step"],
        )
        for ax, coordinate in zip(axes, ("total", "observation"), strict=True):
            x = [
                v["global_step"]
                if coordinate == "total"
                else v["task_counts"]["observation"]
                for v in validations
            ]
            ax.plot(x, [v["score"] for v in validations], label=label, linewidth=1)
            ax.scatter(
                [
                    best["global_step"]
                    if coordinate == "total"
                    else best["task_counts"]["observation"]
                ],
                [best["score"]],
                s=25,
            )
        result["models"][arm] = {
            "name": label,
            "selected": best,
            "held_out": rows,
            "completion": completion_summary(
                files[arm + "-completion/best.json"]["records"]
            ),
            "state_interventions": files[arm + "-state-diagnostics/best.json"][
                "interventions"
            ],
            "selected_om4_retention": next(
                v["om4_retention"]
                for v in validations
                if v["global_step"] == best["global_step"]
            ),
            "day365_by_origin": annual,
            "day365_mean": {
                k: float(np.mean([v[k] for v in annual.values()]))
                for k in next(iter(annual.values()))
            },
            "validation_curve": [
                {
                    k: v[k]
                    for k in ("global_step", "task_counts", "score", "om4_retention")
                }
                for v in validations
            ],
        }
    for ax, xlabel in zip(
        axes,
        (
            "Total optimizer updates (OM4 + observations)",
            "Observation optimizer updates",
        ),
        strict=True,
    ):
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Integrated + spectral validation score (lower is better)")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(
        "Validation history; dots mark selected checkpoints. Incomplete arms omitted."
    )
    fig.savefig(args.output / "validation-curves.png", dpi=160)
    plt.close(fig)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
