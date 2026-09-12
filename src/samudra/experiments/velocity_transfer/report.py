# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Create a brief scientific report from a completed, matched velocity campaign."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments.velocity_transfer.compare import compare

SEEDS = (15, 16, 17)
METHODS = ("samudra", "persistence", "seasonal_climatology", "damped_linear")
KEYS = ["anchor", "target", "lead_step", "region"]
REGIONS = (
    "global_outside_5deg",
    "gulf_stream",
    "kuroshio",
    "southern_ocean",
    "north_pacific_gyre",
)


def read_complete_scores(path, split, variant, seed):
    manifest = json.loads((path / "manifest.json").read_text())
    if (
        not manifest["complete"]
        or manifest["split"] != split
        or manifest["config"]["variant"] != variant
        or manifest["config"]["seed"] != seed
        or manifest["linear_damping"] != 0.25
    ):
        raise ValueError(f"Incomplete or mismatched evaluation: {path}")
    frame = pd.read_csv(path / "scores.csv")
    if frame.duplicated(KEYS + ["method"]).any() or set(frame.method) != set(METHODS):
        raise ValueError(f"Duplicate forecasts or missing baseline: {path}")
    if set(frame.region) != set(REGIONS) or set(frame.lead_step) != {1, 2, 4, 6}:
        raise ValueError(f"Missing region or lead: {path}")
    numeric = frame.select_dtypes(include="number")
    if (
        not np.isfinite(numeric.to_numpy()).all()
        or (frame.weight <= 0).any()
        or (frame.weighted_squared_error < 0).any()
        or (frame.valid_cell_count <= 0).any()
    ):
        raise ValueError(f"Invalid evaluation values: {path}")
    reference = frame[frame.method == "samudra"].set_index(KEYS).sort_index()
    anchors = reference.index.get_level_values("anchor").unique()
    if len(anchors) != manifest["anchors"] or len(reference) != len(anchors) * 20:
        raise ValueError(f"Incomplete date/region/lead coverage: {path}")
    for method in METHODS[1:]:
        baseline = frame[frame.method == method].set_index(KEYS).sort_index()
        columns = ["weight", "valid_cell_count", "actual_lead_days"]
        if not baseline.index.equals(reference.index) or not np.allclose(
            baseline[columns], reference[columns], rtol=1e-7, atol=1e-8
        ):
            raise ValueError(f"Unmatched baseline cohort: {path}, {method}")
    # Recover bias numerators before pooling dates; never average date RMSEs.
    frame["weighted_bias_u"] = frame.bias_u_m_per_s * frame.weight
    frame["weighted_bias_v"] = frame.bias_v_m_per_s * frame.weight
    frame["seed"] = seed
    frame["variant"] = variant
    frame["split"] = split
    return frame


def collect(root):
    campaign = json.loads((root / "campaign.json").read_text())
    if campaign.get("stage") != "complete" or campaign.get("failure"):
        raise ValueError("Campaign is not complete; refusing a final scientific report")
    spent = campaign["allocated_gpu_hours_at_last_gate"]
    if not np.isfinite(spent) or not 0 < spent <= campaign["budget_gpu_hours"]:
        raise ValueError("Missing or over-budget allocation accounting")
    winner = campaign["selected_transfer_arm"]
    if winner not in {f"D{i}" for i in range(1, 6)}:
        raise ValueError("Invalid transfer arm")
    audit_path = root / "data-audit.json"
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    frames, comparisons = [], {}
    reference_baselines: dict[str, pd.DataFrame] = {}
    for split in ("validation", "test"):
        paths: dict[str, list[Path]] = {variant: [] for variant in ("D0", winner)}
        for variant in paths:
            for seed in SEEDS:
                path = root / f"{split}-confirm-{variant}-s{seed}"
                frame = read_complete_scores(path, split, variant, seed)
                if audit is not None:
                    expected = audit["duacs"]["splits"][split]
                    dates = pd.to_datetime(frame.anchor)
                    if (
                        dates.nunique() != expected["windows"]
                        or dates.min() != pd.Timestamp(expected["first_anchor"])
                        or dates.max() != pd.Timestamp(expected["last_anchor"])
                    ):
                        raise ValueError(
                            "Evaluation does not cover the audited date cohort"
                        )
                baseline = (
                    frame[frame.method != "samudra"]
                    .set_index(KEYS + ["method"])
                    .sort_index()
                )
                columns = [
                    "weighted_squared_error",
                    "weight",
                    "valid_cell_count",
                    "actual_lead_days",
                ]
                if split in reference_baselines:
                    reference = reference_baselines[split]
                    if not baseline.index.equals(reference.index) or not np.allclose(
                        baseline[columns], reference[columns], rtol=1e-7, atol=1e-8
                    ):
                        raise ValueError("Baselines differ across arms or seeds")
                else:
                    reference_baselines[split] = baseline
                frames.append(frame)
                paths[variant].append(path)
        # Recompute from the CSVs rather than trusting potentially stale summaries.
        comparisons[split] = compare(paths["D0"], paths[winner])
    scores = pd.concat(frames, ignore_index=True)
    group = ["split", "variant", "seed", "region", "lead_step", "method"]
    metrics = scores.groupby(group)[
        ["weighted_squared_error", "weight", "weighted_bias_u", "weighted_bias_v"]
    ].sum()
    metrics["vector_rmse_m_per_s"] = np.sqrt(
        metrics.weighted_squared_error / metrics.weight
    )
    metrics["bias_u_m_per_s"] = metrics.weighted_bias_u / metrics.weight
    metrics["bias_v_m_per_s"] = metrics.weighted_bias_v / metrics.weight
    coverage = scores.groupby(group).agg(
        forecast_dates=("anchor", "nunique"),
        minimum_valid_cells=("valid_cell_count", "min"),
        maximum_valid_cells=("valid_cell_count", "max"),
        minimum_lead_days=("actual_lead_days", "min"),
        maximum_lead_days=("actual_lead_days", "max"),
    )
    metrics = metrics.join(coverage)
    return campaign, scores, metrics.reset_index(), comparisons


def write_report(root: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    campaign, scores, metrics, comparisons = collect(root)
    winner = campaign["selected_transfer_arm"]
    test = metrics[metrics.split == "test"]
    pooled = test.groupby(["variant", "region", "lead_step", "method"])[
        ["weighted_squared_error", "weight"]
    ].sum()
    pooled["rmse"] = np.sqrt(pooled.weighted_squared_error / pooled.weight)

    def rmse(variant, step, method="samudra"):
        return pooled.loc[(variant, "global_outside_5deg", step, method), "rmse"]

    primary = next(
        row
        for row in comparisons["test"]
        if row["region"] == "global_outside_5deg" and row["nominal_lead_days"] == 10
    )
    improvement = primary["improvement_percent"]
    interval = (primary["paired_block_ci_low"], primary["paired_block_ci_high"])
    text = [
        "# Shared original Samudra: DUACS velocity transfer",
        "",
        f"The selected transfer arm **{winner}** changed held-out 10-day DUACS vector RMSE "
        f"by **{improvement:+.2f}% improvement** relative to DUACS-only D0 "
        f"(paired seed/calendar-quarter bootstrap 95% interval "
        f"{interval[0]:+.2f}% to {interval[1]:+.2f}%). "
        + (
            "The point estimate meets the provisional 3% improvement target."
            if improvement >= 3
            else "The point estimate does not meet the provisional 3% improvement target."
        ),
        "",
        "## Method",
        "",
        "One original ConvNeXt U-Net shares all backbone weights across sources, with "
        "small source-specific input/output adapters. Four five-day velocity histories "
        "predict the next map recursively. Local DUACS geostrophic velocities are "
        "coarsened from 1/8° to 1/4°; OM4 geostrophic velocities are derived from SSH "
        "on the 1° global grid and 1/4° regional crops. No Perceiver or LLC data are used.",
        "",
        "The six-arm screen compares D0 (DUACS only), D1 (+1° global OM4), D2 "
        "(+1/4° regional OM4), D3 (both), D4 (D3 without explicit position/spacing "
        "channels), and D5 (40% OM4 pretraining, then DUACS fine-tuning). Selection "
        "uses validation only. D0 and the selected arm are retrained at equal "
        "128 GPU-hour target budgets with seeds 15/16/17.",
        "",
        campaign.get("screen_hardware_note", ""),
        "",
        "Training ends 2018-09-30; validation spans 2019-04-01–2020-09-30; test "
        "spans 2021-04-01–2022-12-31 subject to archive coverage. Each full history "
        "and forecast window lies within its split. Normalization and climatology "
        "use training dates only. All methods are scored on identical valid cells "
        "outside ±5° latitude. RMSE is the square root of pooled area-weighted "
        "squared vector error, not an average of per-date RMSEs.",
        "",
        "## Held-out forecast skill",
        "",
        "Global vector RMSE in m/s; lower is better. Leads are nominal days.",
        "",
        f"| Lead | Persistence | Climatology | Damped linear | D0 | {winner} |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for step in (1, 2, 4, 6):
        values = [rmse("D0", step, method) for method in METHODS[1:]]
        values += [rmse("D0", step), rmse(winner, step)]
        text.append(f"| {step * 5} | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    text += [
        "",
        "Regional transfer improvement relative to D0 (positive is better):",
        "",
        "| Region | Lead | RMSE improvement | Paired 95% interval |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in comparisons["test"]:
        if row["nominal_lead_days"] in (10, 30):
            text.append(
                f"| {row['region']} | {row['nominal_lead_days']} | "
                f"{row['improvement_percent']:+.2f}% | "
                f"[{row['paired_block_ci_low']:+.2f}, {row['paired_block_ci_high']:+.2f}]% |"
            )
    text += [
        "",
        "Per-seed global 10-day RMSE:",
        "",
        f"| Seed | D0 | {winner} |",
        "| ---: | ---: | ---: |",
    ]
    for seed in SEEDS:
        selected = test[
            (test.seed == seed)
            & (test.region == "global_outside_5deg")
            & (test.lead_step == 2)
            & (test.method == "samudra")
        ].set_index("variant")
        text.append(
            f"| {seed} | {selected.loc['D0', 'vector_rmse_m_per_s']:.4f} | "
            f"{selected.loc[winner, 'vector_rmse_m_per_s']:.4f} |"
        )
    held_out = scores[
        (scores.split == "test") & (scores.variant == "D0") & (scores.seed == 15)
    ]
    text += [
        "",
        "## Coverage, compute, and interpretation",
        "",
        f"Test coverage: {held_out.anchor.nunique()} forecast dates per seed, "
        f"{held_out.anchor.min()} through {held_out.anchor.max()}; "
        f"{primary['quarter_blocks']} calendar-quarter blocks. Actual lead durations "
        "are retained in the score CSVs. Monthly climatology is fitted on training "
        "data; linear extrapolation uses a fixed damping coefficient of 0.25.",
        "",
        f"Slurm-accounted allocation: **{campaign['allocated_gpu_hours_at_last_gate']:.2f} "
        f"GPU-hours**, against the {campaign['budget_gpu_hours']:,} GPU-hour ceiling. "
        "Training uses the locally merged Rust loader and immutable branch code layers.",
        "GPU family accounting: "
        + json.dumps(
            campaign.get("allocated_gpu_hours_by_type_at_last_gate", {}), sort_keys=True
        )
        + ". Hours are reported by hardware type; no cross-hardware performance equivalence is assumed.",
        "",
        "Uncertainty resamples paired seeds and calendar-quarter date blocks. "
        "There are only three seeds and fewer than two test years; overlapping "
        "forecast dates are not independent, and these intervals do not establish "
        "interannual generalization. Regional and longer-lead regressions must be "
        "considered alongside the global point estimate.",
        "",
        "DUACS maps are retrospective reprocessed, centered five-day averages, so "
        "these are not operational issuance-vintage forecasts. The equatorial band "
        "is excluded. OM4 resolutions are regriddings of one trajectory; D1 versus "
        "D2 changes both resolution and extent. D4 removes explicit geometry "
        "channels but retains geographically informative masks and mean currents. "
        "Velocity skill does not establish SSH forecast skill. A negative result "
        "here does not rule out transfer with different tasks or architectures.",
        "",
        "## Reproducibility",
        "",
        f"Training code: `{campaign['code_commit']}`. Runtime/local Rust merge: "
        f"`{campaign['runtime_commit']}`. The accompanying `summary.json` records "
        "selection scores, comparisons, provenance, and accounting; "
        "`metrics-by-seed.csv` includes physical RMSE and bias for every baseline, "
        "region, lead, seed, and split. Raw per-date scores remain in the campaign "
        "evaluation directories.",
        "",
    ]
    output.mkdir(parents=True)
    metrics.to_csv(output / "metrics-by-seed.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps({"campaign": campaign, "comparisons": comparisons}, indent=2) + "\n"
    )
    (output / "report.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.campaign_root, args.output)


if __name__ == "__main__":
    main()
