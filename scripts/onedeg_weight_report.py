# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Summarize the frozen carry experiment: python -m scripts.onedeg_weight_report."""

import json
from pathlib import Path

import matplotlib
import numpy as np

from scripts.onedeg_weight_edit import save_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path(".LOCAL/weight-edit-original")
DEST = Path("experiments/onedeg-weight-edits")
TAG = "carry_balanced_l15_0.5"
KEYS = (
    "rmse_normalized",
    "coastal_rmse_normalized",
    "gradient_rmse_normalized_per_km",
    "coastal_gradient_rmse_normalized_per_km",
)


def read(name):
    return json.loads((SOURCE / name).read_text())


def deep(row):
    return (
        row["channel"].startswith(("thetao_", "so_"))
        and int(row["channel"].split("_")[1]) >= 15
    )


def rms(rows, key):
    return float(np.sqrt(np.mean([r[key] ** 2 for r in rows])))


def compare(candidate, baseline):
    lookup = {(r.get("window"), r["channel"], r["lead_days"]): r for r in baseline}
    result = {}
    for label, predicate in (
        ("all", lambda r: True),
        ("edited_deep_tracers", deep),
        ("other", lambda r: not deep(r)),
    ):
        selected = [r for r in candidate if predicate(r)]
        reference = [
            lookup[(r.get("window"), r["channel"], r["lead_days"])] for r in selected
        ]
        result[label] = {
            key: {
                "candidate": rms(selected, key),
                "baseline": rms(reference, key),
                "ratio": rms(selected, key) / rms(reference, key),
            }
            for key in KEYS
        }
    ratios = [
        (
            r["rmse_normalized"]
            / lookup[(r.get("window"), r["channel"], r["lead_days"])][
                "rmse_normalized"
            ],
            r,
        )
        for r in candidate
    ]
    ratio, worst = max(ratios, key=lambda pair: pair[0])
    result["worst_channel_window_lead"] = {
        "ratio": ratio,
        "channel": worst["channel"],
        "window": worst.get("window"),
        "lead_days": worst["lead_days"],
    }
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    manifest = read(f"{TAG}_edit_manifest.json")
    result = {
        "status": "Experimental partial mitigation; not intended to merge or replace the production checkpoint",
        "candidate": TAG,
        "aggregation": "Square root of equally weighted mean squared per-channel/window/lead RMSE; each per-channel spatial metric is wet-cell area weighted. Global means normalized channels, not a conserved physical quantity.",
        "edited_depths_m": [3100, 4000, 5000, 6000],
        "edit_manifest": manifest,
        "changed_tensor_values": sum(
            v["changed_values"] for v in manifest["changes"].values()
        ),
        "original_parameter_count": 84066474,
        "data": read("data_manifest.json"),
        "precision_check": read("precision_check.json"),
        "export_validation": read("export_validation.json"),
        "single_call": {},
        "rollout": {},
        "limitations": [
            "Four seasonal windows per year, three years; no confidence interval or exhaustive validation.",
            "2015 inputs selected the feature basis; 2015 targets screened candidates. 2017 checked selection; 2019 was opened only after freezing the final candidate.",
            "The one 60-day rollout starts in January 2015 and is a discovery diagnostic, not an independent rollout test.",
            "Persistence outperforms the candidate in the selected deep tracers. This edit damps spurious change; it does not establish improved deep-ocean dynamics.",
            "Some unedited channels regress. No guarantee of heat/salt conservation, stability, positivity, or removal of all boundary artifacts.",
            "Full-record heat-flux anomaly climatology matches the original training convention and includes forcing observations from evaluation years.",
            "No backpropagation, optimizer, or target regression; target feedback still guided candidate selection.",
            "Low-rank projections touch many dense tensor entries; this is not a sparse or small-norm edit.",
        ],
        "sources": {
            "weight_space_article": "https://rikiyat.github.io/blog/weight-space-programming/",
            "samudra2_paper": "https://m2lines.github.io/Samudra/samudra2/",
            "training_run": "https://forge.coreweave.com/wandb/ocean_emulators/default/runs/3o112ot4",
        },
    }
    for split in ("discovery", "selection", "test"):
        result["single_call"][split] = compare(
            read(f"{TAG}_metrics_{split}.json"), read(f"baseline_metrics_{split}.json")
        )
    result["reservation_only_control"] = compare(
        read("carry_balanced_l15_0_metrics_discovery.json"),
        read("baseline_metrics_discovery.json"),
    )
    result["broader_depth_control"] = compare(
        read("carry_balanced_l10_0.5_metrics_discovery.json"),
        read("baseline_metrics_discovery.json"),
    )
    rollout = read(f"rollout_{TAG}_metrics.json")
    for day in range(5, 61, 5):
        baseline = [
            r for r in rollout if r["candidate"] == "baseline" and r["lead_days"] == day
        ]
        result["rollout"][str(day)] = {
            label: compare(
                [
                    r
                    for r in rollout
                    if r["candidate"] == label and r["lead_days"] == day
                ],
                baseline,
            )
            for label in (TAG, "external_blend_control", "persistence")
        }
    save_json(DEST / "results.json", result)
    save_json(DEST / "decoder_screen.json", read("screen_discovery.json"))
    # Standalone scientific figure: preserve the global regression alongside the deep improvement.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    splits = ("discovery", "selection", "test")
    for group, label in (
        ("edited_deep_tracers", "Edited deep tracers"),
        ("all", "All channels"),
    ):
        ratios = [
            result["single_call"][s][group]["rmse_normalized"]["ratio"] for s in splits
        ]
        axes[0].plot([2015, 2017, 2019], ratios, "o-", label=label)
    axes[0].set(
        title="Single-call RMSE", xlabel="Year", ylabel="Edited / original RMSE"
    )
    axes[0].axhline(1, color="gray", linestyle="--")
    axes[0].legend(fontsize=8)
    for ax, group, title in zip(
        axes[1:],
        ("edited_deep_tracers", "all"),
        ("Deep tracers: 2015 rollout", "All channels: 2015 rollout"),
        strict=True,
    ):
        for label, display in (
            ("baseline", "Original"),
            (TAG, "Weight edit"),
            ("external_blend_control", "External blend control"),
            ("persistence", "Persistence"),
        ):
            values = [
                rms(
                    [
                        r
                        for r in rollout
                        if r["candidate"] == label
                        and r["lead_days"] == day
                        and (group == "all" or deep(r))
                    ],
                    "rmse_normalized",
                )
                for day in range(5, 61, 5)
            ]
            ax.plot(range(5, 61, 5), values, label=display)
        ax.set(title=title, xlabel="Lead (days)", ylabel="Normalized RMSE")
        ax.legend(fontsize=7)
    fig.savefig(DEST / "metrics.svg", metadata={"Date": None})
    plt.close(fig)
    for filename in ("results.json", "decoder_screen.json", "metrics.svg"):
        (DEST / (filename + ".license")).write_text(
            "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
        )


if __name__ == "__main__":
    main()
