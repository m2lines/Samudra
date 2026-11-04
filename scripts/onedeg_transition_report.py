# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Build review artifacts from the completed, frozen transition benchmark."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.onedeg_weight_edit import save_json, sha256

LABELS = {
    "original": "Original",
    "carry": "Carry weight edit",
    "reservation_only": "Reservation only",
    "external_blend": "External 50% blend",
    "deep_persistence": "Deep persistence",
}
METRICS = [
    "increment_rmse_normalized",
    "increment_rms_ratio",
    "increment_spatial_correlation",
    "increment_gradient_rmse_normalized_per_km",
    "pred_tail_area_fraction",
    "ref_tail_area_fraction",
]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(".LOCAL/transition-benchmark-v1/results")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/onedeg-weight-edits")
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    frame = pd.read_csv(args.source / "transitions.csv")
    assert len(frame) == 443520
    assert not frame.duplicated(
        ["candidate", "window", "mode", "transition", "channel", "region"]
    ).any()
    deep = frame[
        frame.channel.isin(
            [f"{v}_{d}" for v in ("thetao", "so") for d in range(15, 19)]
        )
    ]
    phases = (
        deep.groupby(["candidate", "split", "mode", "phase", "region"])[METRICS]
        .mean()
        .reset_index()
    )
    phases.to_csv(args.output / "transition_phases.csv", index=False)
    acceptance = json.loads((args.source / "acceptance.json").read_text())
    summaries = json.loads((args.source / "window_summaries.json").read_text())
    selected = []
    for candidate in LABELS:
        for mode in ("teacher_forced", "autoregressive"):
            rows = [
                r["metrics"]
                for r in summaries
                if r["candidate"] == candidate
                and r["mode"] == mode
                and r["split"] == "new_local_test"
            ]
            metrics = {
                k: float(np.mean([r[k] for r in rows]))
                if all(r[k] is not None for r in rows)
                else None
                for k in rows[0]
                if k != "per_channel_state_rmse"
            }
            # Independently check a reported group metric against the raw CSV.
            raw = deep[
                (deep.candidate == candidate)
                & (deep["mode"] == mode)
                & (deep.split == "new_local_test")
                & (deep.region == "wet")
            ]
            logged_rmse = metrics["deep/wet/increment_rmse_normalized"]
            assert logged_rmse is not None
            np.testing.assert_allclose(
                raw.increment_rmse_normalized.mean(),
                logged_rmse,
                rtol=1e-12,
            )
            selected.append({"candidate": candidate, "mode": mode, "metrics": metrics})
    save_json(
        args.output / "transition_results.json",
        {
            "status": "Completed; carry and external blend pass relative screening, but excessive deep tendencies and low correlation remain",
            "wandb_url": "https://wandb.ai/ocean_emulators/default/runs/r65wrzux",
            "source_csv_sha256": sha256(args.source / "transitions.csv"),
            "rows": len(frame),
            "displayed_split": "new_local_test: four seasonal 2021 windows; new to local selection, previously included in aggregate long-rollout evaluation",
            "aggregation": "Equal-weight average over 12 transitions, eight deep tracer channels, and four windows. RMS ratios are averages of per-transition/per-channel spatial RMS ratios, not a ratio of pooled RMS values. Correlations are spatial increment correlations. Tail rates are area weighted against channel/region thresholds calibrated using 2015 OM4 increments.",
            "acceptance_caveat": "Frozen v1 gates measure improvement relative to the original model; passing does not require absolute agreement with OM4 increment magnitude or correlation. Gates are not changed retrospectively.",
            "results": selected,
        },
    )
    shutil.copyfile(
        args.source / "acceptance.json", args.output / "transition_acceptance.json"
    )
    display = deep[
        (deep.split == "new_local_test") & (deep["mode"] == "autoregressive")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), layout="constrained")
    colors = dict(zip(LABELS, ("#333333", "#007fa3", "#b86b22", "#8d50ab", "#5c8c54")))
    for candidate, label in LABELS.items():
        wet = display[(display.candidate == candidate) & (display.region == "wet")]
        curve = wet.groupby("transition").increment_rms_ratio.mean()
        axes[0].plot(
            curve.index * 5,
            curve.values,
            label=label,
            color=colors[candidate],
            linewidth=1.7,
        )
    axes[0].axhline(1, color="#999999", linestyle="--")
    axes[0].set(
        xlabel="Lead (nominal days)",
        ylabel="Mean predicted / OM4 increment RMS",
        title="Deep changes remain too large",
    )
    axes[0].legend(fontsize=8)
    phase_names = ["initial_first_lead", "between_calls", "within_pair"]
    for offset, candidate in enumerate(("original", "carry", "external_blend")):
        part = display[(display.candidate == candidate) & (display.region == "wet")]
        correlation = (
            part.groupby("phase")
            .increment_spatial_correlation.mean()
            .reindex(phase_names)
        )
        axes[1].bar(
            np.arange(3) + (offset - 1) * 0.25,
            correlation.values,
            width=0.24,
            color=colors[candidate],
            label=LABELS[candidate],
        )
    axes[1].set(
        xticks=np.arange(3),
        xticklabels=["Initial", "Between calls", "Within pair"],
        ylabel="Mean spatial increment correlation",
        title="Spatial agreement remains weak",
        ylim=(0, 0.1),
    )
    axes[1].legend(fontsize=8)
    coastal = display[display.region == "coast"]
    values = (
        coastal.groupby("candidate")
        .pred_tail_area_fraction.mean()
        .reindex(list(LABELS))
    )
    axes[2].bar(np.arange(5), values * 100, color=[colors[k] for k in LABELS])
    reference = coastal.ref_tail_area_fraction.mean() * 100
    axes[2].axhline(
        reference, color="black", linestyle="--", label=f"OM4: {reference:.2f}%"
    )
    axes[2].set(
        xticks=np.arange(5),
        xticklabels=["Original", "Carry", "Reserve", "Blend", "Persist"],
        ylabel="Mean coastal area exceeding threshold (%)",
        title="Excessive changes are reduced, not removed",
    )
    axes[2].legend(fontsize=8)
    fig.suptitle(
        "2021 short AR rollouts · temperature/salinity at 3,100–6,000 m · four seasonal windows"
    )
    plt.rcParams["svg.hashsalt"] = "samudra-transition-benchmark-v1"
    svg = args.output / "transition_diagnostics.svg"
    fig.savefig(svg, metadata={"Date": None})
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    fig.savefig(args.source / "transition_diagnostics.png", dpi=150)
    plt.close(fig)
    for name in (
        "transition_phases.csv",
        "transition_results.json",
        "transition_acceptance.json",
        "transition_diagnostics.svg",
    ):
        (args.output / (name + ".license")).write_text(
            "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"
        )
    print(
        f"Validated {len(frame):,} rows; {sum(r['passes_all'] for r in acceptance)}/{len(acceptance)} candidate/split/mode combinations pass."
    )


if __name__ == "__main__":
    main()
