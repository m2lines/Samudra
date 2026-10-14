#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify and report the scratch observation-2k checkpoint."""

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from report_diffusion_interventions import (  # type: ignore[import-not-found]
    COMPONENTS,
    ORIGINS,
    endpoint,
    manifests,
    pool,
    read,
    sha,
    table,
)

from samudra.experiments.diffusion_scratch_evaluate import verify_output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--comparison", type=Path)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--presentation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    signature = read(a.evaluation / "input.json")
    receipt = read(a.evaluation / "COMPLETE.json")
    assert signature["counts"] == {"om4": 5561, "observation": 2000}
    assert signature["processor_prediction"] in ("direct", "residual")
    label = signature["processor_prediction"].capitalize()
    assert receipt["inputs"] == signature
    for section in ("validation", "native", "annual"):
        assert verify_output(a.evaluation, section, signature)
        assert (
            sha(a.evaluation / section / "COMPLETE.json")
            == receipt["receipts"][section]
        )
    with gzip.open(a.presentation, "rt") as f:
        source = json.load(f)
    assert source["configuration"]["origins"] == ORIGINS
    climate = source["metrics"]["Training seasonal climatology"]
    expected = manifests(source["models"]["U-global"]["inputs"])
    roots = {
        f"{label} scratch: 2k obs / 5,561 OM4": a.evaluation / "annual",
        "Historical deterministic: 8k / 8k": a.baseline / "global-endpoint-annual",
        "Historical diffusion: 8k / 8k": a.baseline / "final-annual-steps32",
    }
    if a.comparison:
        other = read(a.comparison / "input.json")
        assert other["counts"] == signature["counts"]
        for section in ("validation", "native", "annual"):
            assert verify_output(a.comparison, section, other)
        roots[
            f"{other['processor_prediction'].capitalize()} scratch: 2k obs / 5,561 OM4"
        ] = a.comparison / "annual"
    rows = []
    curves = []
    for arm, directory in roots.items():
        inputs = read(directory / "input.json")
        assert manifests(inputs) == expected
        records = {o: read(directory / f"{o}.json") for o in ORIGINS}
        for day in (30, 365):
            norm = pool([endpoint(climate[o], o, day) for o in ORIGINS])
            for origin in [*ORIGINS, "pooled"]:
                selected = ORIGINS if origin == "pooled" else [origin]
                values = pool([endpoint(records[o], o, day) for o in selected])
                rows.append(
                    dict(
                        arm=arm,
                        origin=origin,
                        day=day,
                        score=sum(values[k] / norm[k] for k in COMPONENTS) / 4,
                        **values,
                    )
                )
        for day in (5, 15, 30, 90, 180, 365):
            values = pool([records[o]["leads"][str(day)] for o in ORIGINS])
            curves.append(dict(arm=arm, day=day, **values))
    table(a.output / "blended-components.csv", rows)
    table(a.output / "surface-leads.csv", curves)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    markers = ["o", "s", "^", "D"][: len(roots)]
    for arm, marker in zip(roots, markers, strict=True):
        selected = [r for r in rows if r["arm"] == arm and r["origin"] == "pooled"]
        axes[0].plot(
            [r["day"] for r in selected],
            [r["score"] for r in selected],
            marker=marker,
            label=arm,
        )
        selected = [r for r in curves if r["arm"] == arm]
        for ax, key in zip(axes[1:], ("sst_rmse", "adt_rmse"), strict=True):
            ax.plot(
                [r["day"] for r in selected], [r[key] for r in selected], marker=marker
            )
    for ax, title, unit in zip(
        axes,
        ("Blended error: 30 / 365 days", "SST error", "SSH error"),
        ("Climatology-normalized RMSE", "RMSE (degrees C)", "RMSE (m)"),
        strict=True,
    ):
        ax.set(title=title, xlabel="Lead (days)", ylabel=unit, yscale="log")
        ax.grid(alpha=0.25)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=1)
    fig.savefig(a.output / "error-growth.png", dpi=160)
    fig.savefig(a.output / "error-growth.pdf")
    plt.close(fig)
    latents: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    for index, origin in enumerate(ORIGINS):
        obs = read(a.evaluation / "annual" / f"latent-{origin}.json")
        native = read(a.evaluation / "native" / f"{origin[:4]}.json")
        for task, days, values, marker in (
            ("Observation path", obs["leads_days"], obs["latent_rms"], "o"),
            ("OM4 path", list(range(0, 366, 5)), native["latent_rms"], "s"),
        ):
            assert len(days) == len(values)
            ax.plot(
                days,
                values,
                color=f"C{index}",
                marker=marker,
                markevery=12,
                markersize=4,
                label=f"{origin[:4]} {task}",
            )
            latents.extend(
                dict(origin=origin, task=task, day=d, latent_rms=v)
                for d, v in zip(days, values, strict=True)
            )
    ax.set(
        xlabel="Lead (days)",
        ylabel="Latent RMS (log scale)",
        yscale="log",
        title=f"{label} checkpoint: observation and OM4 latent evolution",
    )
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(a.output / "latent-growth.png", dpi=160)
    fig.savefig(a.output / "latent-growth.pdf")
    plt.close(fig)
    table(a.output / "latent-growth.csv", latents)
    calibration = []
    stats = read(a.evaluation / "validation/calibration.json")
    for group, section, nchan, indices in (
        ("SST", "surface", 2, [0]),
        ("SSH", "surface", 2, [1]),
        ("T", "interior", 28, list(range(14))),
        ("S", "interior", 28, list(range(14, 28))),
    ):
        totals = {
            key: sum(
                np.asarray(record[section][key]).reshape(-1, nchan)[:, indices].sum()
                for record in stats
            )
            for key in (
                "weight",
                "mean_squared_error",
                "ensemble_variance",
                "fair_crps",
                "coverage_90",
            )
        }
        calibration.append(
            dict(
                group=group,
                spread_rmse_ratio=float(
                    np.sqrt(totals["ensemble_variance"] / totals["mean_squared_error"])
                ),
                coverage90=totals["coverage_90"] / totals["weight"],
                fair_crps=totals["fair_crps"] / totals["weight"],
            )
        )
    table(a.output / "validation-calibration.csv", calibration)
    print("validation calibration", calibration)
    for path in a.output.iterdir():
        if path.suffix in (".csv", ".png", ".pdf"):
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
    for r in rows:
        if r["origin"] == "pooled":
            print(r)
    assert all(np.isfinite(r["latent_rms"]) for r in latents)


if __name__ == "__main__":
    main()
