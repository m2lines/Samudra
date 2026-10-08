#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Pair final test metrics and pool deterministic/probabilistic scores on identical support."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from samudra.experiments.diffusion_ab_figures import save_png
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_pilot import digest


def load(path):
    return json.loads(path.read_text())


def cooldown(root, output):
    sources = ("cooldown-start", "cooldown-midpoint", "final-validation")
    rows = []
    reference = load(root / f"{sources[0]}-steps32/COMPLETE.json")
    for label in sources:
        folder = root / f"{label}-steps32"
        receipt = load(folder / "COMPLETE.json")
        assert receipt["training_contract"] == reference["training_contract"]
        assert receipt["files"].keys() == reference["files"].keys()
        for name, sha in receipt["files"].items():
            assert digest(folder / name) == sha
            with (
                np.load(folder / name) as a,
                np.load(root / f"{sources[0]}-steps32" / name) as b,
            ):
                for key in (
                    "mask",
                    "lat",
                    "lon",
                    "monthly_reference",
                    "day30_surface_reference",
                ):
                    np.testing.assert_array_equal(a[key], b[key])
        summary = load(root / f"{label}-assets/summary.json")
        spectra = load(root / f"{label}-assets/spectra.json")
        high_frequency = {}
        for name in ("thetao_0", "zos", "thetao_9", "so_9"):
            subset = [r for r in spectra if r["channel"] == name]
            high_frequency[name] = {
                field: float(np.array([r[field] for r in subset])[..., -4:].mean())
                for field in ("reference", "members", "mean")
            }
        rows.append(
            dict(step=receipt["step"], summary=summary, high_frequency=high_frequency)
        )
    (output / "cooldown-comparison.json").write_text(
        json.dumps(
            dict(
                checkpoints=rows,
                verification="All nine full target arrays, coordinates, masks and training contracts match; individual array hashes verified",
                limitation="More training plus LR cooldown; not an equal-exposure LR ablation",
            ),
            indent=2,
        )
        + "\n"
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    x = [r["step"] for r in rows]
    axes[0].plot(x, [r["summary"]["score"] for r in rows], marker="o")
    axes[0].set(ylabel="Validation composite (lower is better)")
    for task, marker in (("surface", "o"), ("interior", "s")):
        axes[1].plot(
            x,
            [r["summary"]["calibration"][task]["spread_rmse"] for r in rows],
            marker=marker,
            label=task,
        )
    axes[1].set(ylabel="Ensemble spread / mean RMSE")
    axes[1].axhline(1, color="black", linewidth=0.7)
    axes[1].legend()
    for name, marker in zip(
        ("thetao_0", "zos", "thetao_9", "so_9"), ("o", "s", "^", "x"), strict=True
    ):
        start = rows[0]["high_frequency"][name]["members"]
        axes[2].plot(
            x,
            [r["high_frequency"][name]["members"] / start for r in rows],
            marker=marker,
            label=name,
        )
    axes[2].set(ylabel="Member high-frequency power / pre-cooldown")
    axes[2].legend()
    for ax in axes:
        ax.set(xlabel="Total updates", xticks=x)
        ax.grid(alpha=0.2)
    save_png(fig, output / "cooldown.png", dpi=150)
    plt.close(fig)


def final_test(root, baseline):
    receipt = load(root / "final-test-steps32/COMPLETE.json")
    with gzip.open(baseline, "rt") as stream:
        control = json.load(stream)
    assert (
        receipt["training_contract"]["reference"]
        == control["validation_reference_sha256"]
    )
    d = control["models"]["obs08000"]
    point_root = root / "deterministic-point-reference"
    point_receipt = load(point_root / "COMPLETE.json")
    assert point_receipt["lineage"] == d["lineage"]
    assert receipt["split"] == "test"
    assert (
        receipt["counts"]
        == d["lineage"]["task_counts"]
        == {"om4": 8000, "observation": 8000}
    )
    assert receipt["step"] == d["lineage"]["global_step"] == 16000
    assert (
        digest(point_root / "point-statistics.json")
        == point_receipt["point_statistics_sha256"]
    )
    assert point_receipt["previous_surface_max_absolute_difference"] == 0
    assert (
        receipt["training_contract"]["observation_manifest"]
        == point_receipt["data_manifest_sha256"]
    )
    points = load(point_root / "point-statistics.json")
    members = load(root / "final-test-steps32/calibration.json")
    assert len(points) == len(members) == 96
    pooled = {}
    for task in ("surface", "interior"):
        for point, ensemble in zip(points, members, strict=True):
            assert point["origin"] == ensemble["origin"]
            for key in ("weight", "cells"):
                np.testing.assert_array_equal(point[task][key], ensemble[task][key])
        weight = sum(np.array(row[task]["weight"]).sum() for row in points)
        pooled[task] = dict(
            deterministic={
                key: float(
                    sum(np.array(row[task][key]).sum() for row in points) / weight
                )
                for key in ("mean_squared_error", "absolute_error")
            },
            diffusion={
                key: float(
                    sum(np.array(row[task][key]).sum() for row in members) / weight
                )
                for key in (
                    "mean_squared_error",
                    "fair_crps",
                    "empirical_crps",
                    "ensemble_variance",
                )
            },
        )
    score = selection_score(
        receipt["metrics"], control["reporting_climatology"], control["spectral_keys"]
    )
    return dict(
        heldout_reporting_score=dict(
            diffusion=score, deterministic=d["forecast"]["composite"]
        ),
        validation_normalized_test_score=dict(
            diffusion=receipt["score"],
            deterministic=d["validation_normalized_test_score"],
        ),
        pooled_standardized=pooled,
        diffusion_metrics=receipt["metrics"],
        deterministic=d,
        reporting_climatology=control["reporting_climatology"],
        spectral_keys=control["spectral_keys"],
        verification="96 origins, per-origin/channel/lead wet weights and finite-cell counts identical; deterministic day5..30 exports reproduced exactly; manifests and frozen validation reference identical",
        score_scope=control["score_definition"]
        + " No held-out score was used for checkpoint selection; validation-normalized test scores are retained separately.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cooldown(args.root, args.output)
    if args.baseline:
        result = final_test(args.root, args.baseline)
        (args.output / "heldout-comparison.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
