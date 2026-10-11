#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare saved historical and reproduced validation on one frozen RMSE scale."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = [
    "sst_rmse",
    "velocity_rmse",
    "eke_rmse",
    "ohc_0_700_rmse",
    "ohc_700_2000_rmse",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())
    reference = json.loads(args.reference.read_text())["control"]["metrics"]
    rows, evidence = [], []
    contract = None
    for source in sources:
        path = Path(source["bundle"])
        bundle = json.loads(gzip.decompress(path.read_bytes()))
        evidence.append(
            dict(
                wave=source["wave"],
                path=source.get("archive", str(path)),
                sha256=sha(path),
            )
        )
        for model, run in bundle.items():
            manifest = run["manifest"]
            current = {
                k: manifest[k]
                for k in ("data_manifest_sha256", "grid_sha256", "statistics_sha256")
            }
            current.update(
                {
                    k: manifest["arguments"][k]
                    for k in (
                        "global_observations",
                        "strict_velocity_support",
                        "surface_policy",
                        "surface_fill",
                    )
                }
            )
            if contract is None:
                contract = current
            if current != contract or manifest["protocol"]["integrated"] != COMPONENTS:
                raise ValueError("Observation/metric contract changed: " + model)
            seen = set()
            for item in run["events"]:
                event = item["data"]
                step = event["global_step"]
                if step in seen:
                    raise ValueError("Duplicate validation step")
                seen.add(step)
                metrics = event["metrics"]["metrics"]
                ratios = [metrics[k] / reference[k] for k in COMPONENTS]
                score = float(np.mean(ratios))
                if not np.isfinite(ratios).all():
                    raise ValueError("Nonfinite validation")
                if manifest["arguments"].get(
                    "selection_metric"
                ) == "rmse-only" and not np.isclose(
                    score, event["score"], rtol=0, atol=1e-12
                ):
                    raise ValueError("RMSE reference changed")
                rows.append(
                    dict(
                        wave=source["wave"],
                        model=model,
                        seed=manifest["arguments"]["seed"],
                        budget=manifest["arguments"]["joint_steps"]
                        + manifest["arguments"]["om4_updates"],
                        update=step,
                        observation_updates=event["task_counts"]["observation"],
                        rmse_blend=score,
                        **{k: metrics[k] for k in COMPONENTS},
                    )
                )
    rows.sort(key=lambda r: (r["wave"], r["model"], r["update"]))
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "validation.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    series: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        series.setdefault((row["wave"], row["model"]), []).append(row)
    minima = []
    for (wave, model), events in series.items():
        best = min(events, key=lambda r: r["rmse_blend"])
        final_steps = set(range(best["budget"] - 400, best["budget"] + 1, 100))
        final_events = [e for e in events if e["update"] in final_steps]
        final_mean = (
            float(np.mean([e["rmse_blend"] for e in final_events]))
            if {e["update"] for e in final_events} == final_steps
            else None
        )
        minima.append(
            dict(
                wave=wave,
                model=model,
                budget=best["budget"],
                seed=best["seed"],
                best_update=best["update"],
                best_rmse=best["rmse_blend"],
                last_update=events[-1]["update"],
                last_rmse=events[-1]["rmse_blend"],
                final_five_scheduled_mean=final_mean,
            )
        )
    with (out / "minima.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(minima[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(minima)
    panels = [
        (
            "Matched 4k schedules",
            lambda w, m, e: (
                e[0]["budget"] == 4000
                and m
                in {
                    "U-global",
                    "U-global-repro4k",
                    "U-multitask-early-fine",
                    "U-multitask-early-fine-latent",
                }
            ),
        ),
        (
            "Early fine: 4k versus stretched 6k",
            lambda w, m, e: m == "U-multitask-early-fine",
        ),
        (
            "Early fine + latent: 4k versus stretched 6k",
            lambda w, m, e: m == "U-multitask-early-fine-latent",
        ),
        ("U-global controls and seeds", lambda w, m, e: m.startswith("U-global")),
    ]
    for axis, zoom, stem in [
        ("update", False, "validation-full"),
        ("update", True, "validation-late"),
        ("observation_updates", True, "validation-observation-exposure"),
    ]:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
        for ax, (title, include) in zip(axes.flat, panels, strict=True):
            for (wave, model), events in series.items():
                if not include(wave, model, events):
                    continue
                label = f"{model} / {events[0]['budget'] // 1000}k / {wave}"
                color = (
                    "tab:orange"
                    if model.endswith("fine-latent")
                    else "tab:blue"
                    if model.endswith("fine")
                    else "0.5"
                    if model.endswith("seed1730")
                    else "black"
                )
                style = (
                    ":"
                    if wave == "original 4k"
                    else "--"
                    if events[0]["budget"] == 6000
                    else "-"
                )
                ax.plot(
                    [e[axis] for e in events],
                    [e["rmse_blend"] for e in events],
                    label=label,
                    color=color,
                    linestyle=style,
                    lw=1.35,
                    marker="o",
                    markersize=2,
                )
            ax.set(
                title=title,
                xlabel="Total optimizer updates"
                if axis == "update"
                else "Observation updates",
                ylabel="Validation RMSE blend (lower is better)",
            )
            if zoom:
                ax.set_ylim(0.80, 1.15)
                ax.set_xlim(left=2000 if axis == "update" else 500)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=7)
        fig.suptitle(
            "Same five RMSE components and frozen validation reference; no spectra"
            + (" · late-stage zoom" if zoom else "")
        )
        fig.savefig(out / (stem + ".png"), dpi=120)
        fig.savefig(out / (stem + ".pdf"))
        plt.close(fig)
    (out / "provenance.json").write_text(
        json.dumps(
            dict(
                sources=evidence,
                reference_sha256=sha(args.reference),
                script_sha256=sha(__file__),
                observation_contract=contract,
                events=len(rows),
                rescore="All original checkpoints rescored from saved physical validation components; historical weights are not reselected or re-evaluated on test data.",
                caveat="Historical U-global ran on Beta; historical early arms include recorded recovery attempts. New reproductions and the 6k wave share the Torch producer. Same seed is a repeatability check, not a new independent seed.",
            ),
            indent=2,
        )
        + "\n"
    )
    print(f"Verified and plotted {len(rows)} events across {len(series)} series")


if __name__ == "__main__":
    main()
