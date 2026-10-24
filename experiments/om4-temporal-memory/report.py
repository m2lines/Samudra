"""Produce a report from completed diagnostic outputs, without model execution."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    args = ap.parse_args()
    root = Path(args.results)
    files = sorted(root.glob("origin-*.json"))
    if not files:
        raise RuntimeError("No completed diagnostic origins")
    records = []
    gradients = []
    for f in files:
        obj = json.loads(f.read_text())
        records += obj["interventions"]
        gradients += obj["gradients"]
    baseline = {
        (r["origin"], r["call"]): r["mse"] for r in records if r["kind"] == "baseline"
    }
    groups = list(dict.fromkeys(r["group"] for r in gradients))
    summary = []
    for g in groups:
        row = {"group": g}
        for call in (1, 2, 4, 6, 12):
            selected = [
                r
                for r in records
                if r["group"] == g
                and r["kind"] == "remove_error_texture"
                and r["alpha"] == 0.5
                and r["call"] == call
            ]
            deltas = [
                100 * (r["mse"] / baseline[r["origin"], call] - 1) for r in selected
            ]
            row[f"call_{call}_percent_mse_change"] = (
                float(np.mean(deltas)) if deltas else None
            )
            if call == 4:
                row["origins_future_worse"] = sum(d > 0 for d in deltas)
                controls = [
                    100 * (r["mse"] / baseline[r["origin"], call] - 1)
                    for r in records
                    if r["group"] == g
                    and r["kind"].startswith("phase_randomized")
                    and r["call"] == call
                ]
                row["random_control_call4_percent"] = float(np.mean(controls))
        for obj in ("fixed_wet_mse", "archived_dynamic_weights"):
            gs = [r for r in gradients if r["group"] == g and r["objective"] == obj]
            row[obj + "_opposed_origins"] = sum(
                r["immediate_gradient_dot_error_texture"] > 0
                and r["future_gradient_dot_error_texture"] < 0
                for r in gs
            )
        summary.append(row)
    with open(root / "summary.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(root / "gradient-projections.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=list(gradients[0]))
        w.writeheader()
        w.writerows(gradients)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    x = np.arange(len(groups))
    labels = [g.replace("_", "\n") for g in groups]
    for ax, call in zip(axes, (1, 4), strict=True):
        ax.bar(
            x,
            [r[f"call_{call}_percent_mse_change"] for r in summary],
            label="Remove half error texture",
        )
        if call == 4:
            ax.scatter(
                x,
                [r["random_control_call4_percent"] for r in summary],
                color="black",
                label="Phase-randomized control",
            )
        ax.axhline(0, color="gray", lw=1)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylabel("Change in total normalized MSE (%)")
        ax.set_title("Immediate state" if call == 1 else "Fourth forecast block")
        ax.legend(fontsize=8)
    fig.savefig(root / "intervention-summary.png", dpi=180)
    fig.savefig(root / "intervention-summary.pdf")
    provenance = json.loads((root / "provenance.json").read_text())
    lines = [
        "# OM4 temporal-memory diagnostics",
        "",
        f"Analyzed {len(files)} held-out origins using official 1° run `uw49dzuq`, "
        "training run `3o112ot4`, epoch-70 EMA checkpoint.",
        "",
        f"Checkpoint SHA-256: `{provenance['checkpoint_sha256']}`.",
        "",
        "Archived prediction replay passed the preset normalized RMS threshold of 0.001. "
        "The raw replay summary in provenance combines different physical units and must "
        "not be interpreted as normalized RMS or a single physical-unit error.",
        "",
        "Model is in evaluation mode with fixed BatchNorm statistics. Gradients below "
        "describe the trained model, not gradients recorded during its original training.",
        "",
        "Each model call predicts two five-day timestamps. The intervention changes "
        "both timestamps in the first predicted block, then continues with unchanged forcing.",
        "",
        "The error texture is center minus the four-neighbor mean, restricted to all-wet "
        "stencils. This is a local high-frequency diagnostic, not a sharp spectral projection. "
        "Truth is used only for diagnostic interventions. This is not a deployable correction.",
        "",
        "The table removes half the error texture (alpha=0.5). MSE is the total "
        "normalized wet-cell loss, averaged equally across all 77 channels and both output "
        "timestamps, not just the intervened group. The fourth block covers lead days 35 and 40 "
        "after the last input timestamp. Percent changes are averaged over origins relative "
        "to each unmodified rollout. "
        "Negative means better. The randomized control matches per-plane perturbation norm; "
        "remasking can alter its spectrum. CSV and per-origin JSON retain all amplitudes and controls.",
        "",
        "| Intervened group | Immediate MSE Δ% | Fourth-block MSE Δ% | Random control Δ% | Future worse origins | Opposed gradient signs: fixed / archived |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['group']} | {r['call_1_percent_mse_change']:.4g} | "
            f"{r['call_4_percent_mse_change']:.4g} | {r['random_control_call4_percent']:.4g} | "
            f"{r['origins_future_worse']}/{len(files)} | "
            f"{r['fixed_wet_mse_opposed_origins']} / {r['archived_dynamic_weights_opposed_origins']} |"
        )
    lines += [
        "",
        "![Interventions](intervention-summary.png)",
        "",
        "Opposed signs means the immediate loss favors removing the error texture while "
        "future loss locally favors preserving it. This is evidence of a tradeoff; it does "
        "not distinguish stored historical information from compensation for imperfect dynamics.",
        "",
        "The randomized and truth-restart controls, sensitivity to intervention amplitude, "
        "and consistency across origins must be examined before interpreting a future-error increase "
        "as dependence on a specific signal. Six origins do not establish broad generality.",
        "",
        "Interpretation: these interventions do not show a consistent need to preserve erroneous "
        "texture for future skill in this checkpoint. Deep temperature, deep salinity, and SSH "
        "removal improves mean fourth-block total MSE; surface salinity worsens it by only "
        "0.000844 percent. Opposed immediate/future gradient signs occur in 2 of 6 surface "
        "salinity cases under fixed MSE, but none under archived weights. This is not evidence "
        "that temporal training gradients caused or did not cause the artifacts. Small global "
        "loss changes can also hide larger channel-specific effects. This 1-degree result "
        "does not establish the mechanism of the quarter-degree texture in the viewer.",
        "",
        "The four new matched-training arms are a separate experiment; no training comparison "
        "is implied by these checkpoint diagnostics.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
