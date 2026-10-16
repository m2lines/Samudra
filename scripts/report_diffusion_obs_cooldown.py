#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify and reduce the frozen residual observation-only cooldown endpoints."""

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

LABELS = {
    "residual-obs4000": "Parent raw: 4,000 obs",
    "cooldown-raw": "Cooldown raw: 4,500 obs",
    "cooldown-ema": "Cooldown EMA: 4,500 obs",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--presentation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    signatures = {}
    audit = {}
    for name in LABELS:
        root = a.root / name
        signature = read(root / "input.json")
        receipt = read(root / "COMPLETE.json")
        assert receipt["inputs"] == signature
        assert signature["counts"] == {
            "om4": 7721,
            "observation": 4000 if name == "residual-obs4000" else 4500,
        }
        assert signature["members"] == 8 and signature["sampling_steps"] == 32
        for section in ("validation", "native", "annual"):
            assert verify_output(root, section, signature)
            assert sha(root / section / "COMPLETE.json") == receipt["receipts"][section]
        signatures[name] = signature
        audit[name] = {
            "input_sha256": sha(root / "input.json"),
            "complete_sha256": sha(root / "COMPLETE.json"),
            "checkpoint_sha256": signature["checkpoint_sha256"],
            "counts": signature["counts"],
            "section_receipts": receipt["receipts"],
        }
    with gzip.open(a.presentation, "rt") as f:
        source = json.load(f)
    climate = source["metrics"]["Training seasonal climatology"]
    expected = manifests(source["models"]["U-global"]["inputs"])
    roots = {label: a.root / name / "annual" for name, label in LABELS.items()}
    roots["Historical deterministic: 8k obs / 8k OM4"] = (
        a.baseline / "global-endpoint-annual"
    )
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    latents: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for label, root in roots.items():
        assert manifests(read(root / "input.json")) == expected
        records = {o: read(root / f"{o}.json") for o in ORIGINS}
        for day in (30, 365):
            norm = pool([endpoint(climate[o], o, day) for o in ORIGINS])
            for origin in [*ORIGINS, "pooled"]:
                selected = ORIGINS if origin == "pooled" else [origin]
                values = pool([endpoint(records[o], o, day) for o in selected])
                rows.append(
                    dict(
                        arm=label,
                        origin=origin,
                        day=day,
                        score=sum(values[k] / norm[k] for k in COMPONENTS) / 4,
                        **values,
                    )
                )
        for day in (5, 15, 30, 90, 180, 365):
            curves.append(
                dict(
                    arm=label,
                    day=day,
                    **pool([records[o]["leads"][str(day)] for o in ORIGINS]),
                )
            )
    for name, label in LABELS.items():
        root = a.root / name
        val = read(root / "validation/COMPLETE.json")
        validation.append(
            dict(
                arm=label,
                blended_spectral_validation_score=val["score"],
                **val["metrics"]["metrics"],
            )
        )
        for o in ORIGINS:
            obs = read(root / "annual" / f"latent-{o}.json")
            nat = read(root / "native" / f"{o[:4]}.json")
            for task, days, values in (
                ("observation", obs["leads_days"], obs["latent_rms"]),
                ("OM4", list(range(0, 366, 5)), nat["latent_rms"]),
            ):
                assert len(days) == len(values) and np.isfinite(values).all()
                latents.extend(
                    dict(arm=label, origin=o, task=task, day=d, latent_rms=v)
                    for d, v in zip(days, values, strict=True)
                )
            for day, metrics in nat["metrics"].items():
                for metric, values in metrics.items():
                    for channel, value in enumerate(np.asarray(values).reshape(-1)):
                        native.append(
                            dict(
                                arm=label,
                                origin=o,
                                day=int(day),
                                metric=metric,
                                channel=channel,
                                value=value,
                            )
                        )
        stats = read(root / "validation/calibration.json")
        for group, section, nchan, indices in (
            ("SST", "surface", 2, [0]),
            ("SSH", "surface", 2, [1]),
            ("T", "interior", 28, list(range(14))),
            ("S", "interior", 28, list(range(14, 28))),
        ):
            totals = {
                key: sum(
                    np.asarray(r[section][key]).reshape(-1, nchan)[:, indices].sum()
                    for r in stats
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
                    arm=label,
                    group=group,
                    spread_rmse_ratio=np.sqrt(
                        totals["ensemble_variance"] / totals["mean_squared_error"]
                    ),
                    coverage90=totals["coverage_90"] / totals["weight"],
                    fair_crps=totals["fair_crps"] / totals["weight"],
                )
            )
    for filename, data in (
        ("blended-components", rows),
        ("surface-leads", curves),
        ("latent-growth", latents),
        ("validation-calibration", calibration),
        ("native-channel-metrics", native),
        ("validation", validation),
    ):
        table(a.output / f"{filename}.csv", data)
    native_path = a.output / "native-channel-metrics.csv"
    with (a.output / "native-channel-metrics.csv.gz").open("wb") as target:
        with gzip.GzipFile(filename="", mode="wb", fileobj=target, mtime=0) as stream:
            stream.write(native_path.read_bytes())
    native_path.unlink()
    Path(str(native_path) + ".license").unlink(missing_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for label, marker in zip(roots, ("o", "s", "^", "D"), strict=True):
        r = [x for x in rows if x["arm"] == label and x["origin"] == "pooled"]
        axes[0].plot(
            [x["day"] for x in r], [x["score"] for x in r], marker=marker, label=label
        )
        r = [x for x in curves if x["arm"] == label]
        for ax, key in zip(axes[1:], ("sst_rmse", "adt_rmse"), strict=True):
            ax.plot([x["day"] for x in r], [x[key] for x in r], marker=marker)
    for ax, title, unit in zip(
        axes,
        ("Blended RMSE", "SST error", "SSH error"),
        ("Climatology-normalized RMSE", "RMSE (degrees C)", "RMSE (m)"),
        strict=True,
    ):
        ax.set(title=title, xlabel="Lead (days)", ylabel=unit)
        ax.grid(alpha=0.25)
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    for ext in ("png", "pdf"):
        fig.savefig(a.output / f"error-growth.{ext}", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for label, marker in zip(LABELS.values(), ("o", "s", "^"), strict=True):
        for ax, task in zip(axes, ("observation", "OM4"), strict=True):
            for index, o in enumerate(ORIGINS):
                r = [
                    x
                    for x in latents
                    if x["arm"] == label and x["task"] == task and x["origin"] == o
                ]
                ax.plot(
                    [x["day"] for x in r],
                    [x["latent_rms"] for x in r],
                    color=f"C{list(LABELS.values()).index(label)}",
                    alpha=0.6,
                    marker=marker,
                    markevery=24,
                    label=label if index == 0 else None,
                )
            ax.set(title=f"{task} path", xlabel="Lead (days)", ylabel="Latent RMS")
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    for ext in ("png", "pdf"):
        fig.savefig(a.output / f"latent-growth.{ext}", dpi=140)
    plt.close(fig)
    (a.output / "verification.json").write_text(json.dumps(audit, indent=2) + "\n")
    for path in a.output.iterdir():
        if path.suffix in (".csv", ".gz", ".json", ".png", ".pdf"):
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
    print(json.dumps([r for r in rows if r["origin"] == "pooled"], indent=2))


if __name__ == "__main__":
    main()
