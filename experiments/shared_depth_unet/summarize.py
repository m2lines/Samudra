# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Generate a compact report from completed local runs; no model execution."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads((args.input / "results.json").read_text())
    assert len(rows) == 12, (
        "All four arms and three seeds must finish before conclusions"
    )
    arms = ["independent", "shared", "shared_context", "independent_context"]
    labels = ["Independent", "Shared", "Shared + context", "Independent + context"]
    by = {
        arm: sorted([r for r in rows if r["arm"] == arm], key=lambda r: r["seed"])
        for arm in arms
    }
    for runs in by.values():
        assert [r["seed"] for r in runs] == [0, 1, 2]
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ["results.json", "provenance.json", "checks.json", "loss_scales.json"]:
        shutil.copy2(args.input / name, args.output / name)

    def values(arm, lead, metric):
        return np.array([r["test"][str(lead)][metric] for r in by[arm]])

    def fmt(a):
        return f"{a.mean():.4f} ± {a.std(ddof=1):.4f}"

    table = [
        "| Arm | Parameters | Coarse RMSE, step 1 | Fine RMSE, step 1 | Fine RMSE, step 5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, label in zip(arms, labels):
        table.append(
            f"| {label} | {by[arm][0]['parameters']:,} | {fmt(values(arm, 1, 'coarse'))} | {fmt(values(arm, 1, 'fine'))} | {fmt(values(arm, 5, 'fine'))} |"
        )
    ref = rows[0]["test"]
    table.append(
        f"| Persistence | — | {ref['1']['persistence_coarse']:.4f} | {ref['1']['persistence_fine']:.4f} | {ref['5']['persistence_fine']:.4f} |"
    )
    paired = []
    for shared, independent in [
        ("shared", "independent"),
        ("shared_context", "independent_context"),
    ]:
        for lead in [1, 5]:
            changes = 100 * (
                values(shared, lead, "fine") / values(independent, lead, "fine") - 1
            )
            paired.append(
                f"- {shared} versus {independent}, step {lead}: mean paired fine RMSE change {changes.mean():+.2f}%; individual seeds {', '.join(f'{x:+.2f}%' for x in changes)}."
            )
    diagnostics = [
        "| Arm | Detail RMSE, step 1 | Detail power / truth, step 1 | Detail power / truth, step 5 | Fine RMSE: 80 / 112 / 128 patch |",
        "|---|---:|---:|---:|---|",
    ]
    for arm, label in zip(arms, labels):
        ratios = [
            np.mean(
                (
                    values(arm, k, "detail_prediction_rms")
                    / values(arm, k, "detail_target_rms")
                )
                ** 2
            )
            for k in [1, 5]
        ]
        ext = [
            np.mean([r["extent_test"][str(size)]["fine"] for r in by[arm]])
            for size in [80, 128]
        ]
        diagnostics.append(
            f"| {label} | {fmt(values(arm, 1, 'fine_detail'))} | {ratios[0]:.3f} | {ratios[1]:.3f} | {ext[0]:.4f} / {values(arm, 1, 'fine').mean():.4f} / {ext[1]:.4f} |"
        )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout="constrained")
    colors = plt.get_cmap("tab10")(np.arange(10))
    leads = [1, 3, 5]
    for i, (arm, label) in enumerate(zip(arms, labels)):
        for ax, metric, title in [
            (axes[0, 0], "coarse", "Coarse global forecast"),
            (axes[0, 1], "fine", "Fine regional forecast"),
        ]:
            means = [values(arm, k, metric).mean() for k in leads]
            std = [values(arm, k, metric).std(ddof=1) for k in leads]
            ax.errorbar(
                leads,
                means,
                yerr=std,
                marker="o",
                label=label,
                color=colors[i],
                capsize=3,
            )
            ax.set(title=title, xlabel="Autoregressive steps", ylabel="RMSE")
        ratios = [
            np.mean(
                (
                    values(arm, k, "detail_prediction_rms")
                    / values(arm, k, "detail_target_rms")
                )
                ** 2
            )
            for k in leads
        ]
        axes[1, 0].plot(leads, ratios, "o-", label=label, color=colors[i])
        histories = [
            json.loads(
                (args.input / f"{arm}_seed{r['seed']}" / "history.json").read_text()
            )
            for r in by[arm]
        ]
        xs = [p["step"] for p in histories[0]]
        ys = np.array([[p["val_score"] for p in hist] for hist in histories])
        axes[1, 1].plot(xs, ys.mean(0), color=colors[i], label=label)
    for ax, metric in [(axes[0, 0], "coarse"), (axes[0, 1], "fine")]:
        ax.plot(
            leads,
            [ref[str(k)][f"persistence_{metric}"] for k in leads],
            "k--",
            label="Persistence",
        )
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].axhline(1, color="black", linestyle="--")
    axes[1, 0].set(
        title="Detail power retained",
        xlabel="Autoregressive steps",
        ylabel="Prediction / truth power",
    )
    axes[1, 1].set(
        title="Validation selection objective",
        xlabel="Training updates",
        ylabel="Mean normalized MSE",
    )
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Shared-depth U-Net: four arms × three seeds\nSynthetic 2D vorticity; error bars show seed SD",
        fontsize=13,
    )
    fig.savefig(args.output / "comparison.png", dpi=140)
    plt.close(fig)
    runtime = sum(r["seconds"] for r in rows)
    checks = json.loads((args.input / "checks.json").read_text())
    report = f"""<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Local shared-depth U-Net experiment

A paired 2×2 test of shared versus independent coarse/fine parameters, with and without current coarse-global context. This is an experiment/results record, not intended to merge. These synthetic results do not establish ocean forecasting skill or cross-source transfer.

## Results

Mean ± sample standard deviation over three training seeds; lower RMSE is better. Units are the synthetic vorticity units (initial spatial SD approximately one). Every run uses the same held-out test examples.

{chr(10).join(table)}

![Forecast, detail-power, and learning-curve comparison](comparison.png)

{chr(10).join(paired)}

These paired seed differences are descriptive, not confidence intervals over independent datasets. All seeds share one fixed train/validation/test dataset.

## Interpretation

Sharing without context improves one-step fine RMSE modestly across all three seeds, while coarse skill is nearly unchanged. Its five-step advantage does not persist. Sharing with context improves fine RMSE against the corresponding independent-context control in every seed at both one and five steps, using roughly 37% fewer parameters. The shared-context arm also has the lowest mean one-step coarse RMSE, although differences between coarse results are small compared with seed variation.

The one-step shared-context detail power ratio is only about 0.63, so the best model still smooths fine structure. Its mean five-step ratio near one conceals seed values of about 0.80, 1.39, and 0.81; this does not establish calibrated amplitudes or correct placement of fine structures. Validation curves are still improving for the shared-context arm at the training-budget boundary. These are fixed-budget optimization/transfer results, not converged rankings. A larger or longer run, new data seeds, and a fine-data-budget sweep would be needed to establish robust transfer or sample efficiency.

## Detail and spatial extent

{chr(10).join(diagnostics)}

Detail is the residual after 4×4 area averaging and piecewise-constant prolongation, not a sharp Fourier band. Its power ratio detects smoothing that RMSE alone can reward. Extent tests score the same physical 32×32 center, changing patch size and halo together (24, 40, 48 fine cells). The 128-cell test supplies extra surrounding data; it is not evidence of transfer to new grid spacing. Five-step rollouts predict the whole regional patch and coarse globe autoregressively without refreshing either from truth. Local boundary errors can propagate into the scored center during rollout.

## Architecture and controls

- One physical input channel: vorticity. Fine input: 128×256 full field; coarse global input: 32×64; training patch: 112×112, scoring only central 32×32.
- Fine-only detail width 4; shared-resolution width 8; bottleneck width 12. Each core has one 2× down/up pair. Ratio between fine and shared grids is 4 in each direction.
- Uses the repository's `ConvNeXtBlock` with expansion factor 2, dilation 1, no normalization, and pointwise linear projections. A subclass changes only padding: both axes periodic for the synthetic global domain, zero exterior for regional patches. This is an experimental staged network built from Samudra blocks, not the unchanged production `Samudra.forward_once`.
- Coarse input is exactly a 4×4 area average. Fine input is split into that same coarse field and its zero-block-mean detail residual. The detail encoder reduces to the shared physical grid; no availability-mask channels are used.
- Fine prediction combines a coarse tendency with a projected zero-block-mean fine correction. Coarse examples omit the detail path. Sharing ties the entire coarse route (input adapter, U-Net core, tendency head); independent routes start as exact copies.
- Context is cropped from the CURRENT coarse global feature map, then linearly projected into the local core input. It never uses future truth. Fine loss backpropagates through this context in both context arms, so “independent” means untied weights, not isolated optimization.
- All arms start from identical corresponding weights, see identical batches/crops within seed, and take 2,000 updates with batch size 4. Each update includes both coarse and fine losses. This matches task exposure and widths, not parameter count or FLOPs.
- Loss is the equal mean of coarse and interior-fine MSE divided by their respective training-set persistence MSE. AdamW, weight decay 1e-4, learning rate cosine decay .001 → .0002, gradient clipping 5.
- Select the best checkpoint using the mean normalized one-step validation MSE every 100 updates. Test is evaluated after selection, using 64 fixed examples; validation uses 32 fixed examples. All final comparisons use these selected checkpoints.

## Data and numerical checks

48 training, 12 validation, and 12 test trajectories generated with independent seeds 12001/12002/12003. Ten frames per trajectory; train times 0–8, evaluation starts 0–4 to allow five-step rollouts. Splits are by whole simulation, not overlapping spatial patches.

Dealiased pseudo-spectral incompressible 2D vorticity evolution on [0,2π) × [0,4π), viscosity .002, RK4 timestep .02 and forecast interval .4. Initial fields combine low and intermediate Fourier modes; no forcing. This is a synthetic torus, not a spherical ocean.

- Halving simulator timestep: one-forecast-interval RMSE {checks["solver_timestep_halving_rmse"]:.3g} on two independent verification trajectories.
- Mean vorticity conservation and decreasing enstrophy checked.
- Aligned patch restriction equals cropping the restricted global field; detail has zero block mean.
- All four models start at persistence and have finite backward passes; parameter identity checked for tied versus untied routes.
- Full-domain versus aligned cropped core interior maximum discrepancy: {checks["crop_core_max_abs_error"]:.3g}.
- Saved checkpoints are reloaded before held-out evaluation. Final metrics, environment metadata, and source/data hashes are in the adjacent JSON files.

## Cost and scope

Training plus periodic validation across 12 runs took {runtime:.1f} seconds ({runtime / 60:.2f} minutes) on the local NVIDIA GB10, excluding data generation and final tests. Maximum reported PyTorch allocated GPU memory was {max(r["peak_cuda_bytes"] for r in rows) / 2**20:.1f} MiB, including the resident dataset; this is not total driver/process memory.

This tests whether sharing is feasible and useful under one small synthetic training distribution. It does not test fine-data sample efficiency, heterogeneous sources, native LLC geometry, conservation across assembled tiles, or long-term ocean stability. Parameter savings and performance changes should be reported together. The 100-update pilot was bring-up only and is excluded from these tables.

## Reproduction and artifacts

Run from the repository root with `PYTHONPATH=src` and a working PyTorch CUDA environment:

```bash
python experiments/shared_depth_unet/run.py --output experiments/shared_depth_unet/outputs/reproduction --steps 2000 --seeds 0 1 2 --eval-every 100
python experiments/shared_depth_unet/summarize.py --input experiments/shared_depth_unet/outputs/reproduction --output experiments/shared_depth_unet/reproduced_results
```

`outputs/main/` holds the generated dataset, all twelve checkpoints, training histories, and exact run source snapshot. It is ignored by Git. Checked-in `provenance.json` contains the configuration, runtime metadata, base commit, and source, imported block, and dataset SHA256 hashes. The four hash strings are explicitly recorded as non-secret false positives in the scanner baseline. The final script adds lint/type cleanups and a guard against reusing incompatible run directories. Exact executed source can be reconstructed with the adjacent `executed-source.patch`:

```bash
patch -o /tmp/shared-depth-executed.py experiments/shared_depth_unet/run.py < experiments/shared_depth_unet/results/executed-source.patch
```

The reconstructed file SHA256 matches `provenance.json`. `final_verification.json` records bit-identical checkpoint forward outputs for all four arms (seed 0) between the executed and final source. Dataset and checkpoint binaries remain local; they are not uploaded to W&B or committed to Git.
"""
    (args.output / "REPORT.md").write_text(report)
    print("\n".join(table))
    print("\n".join(paired))
    print(f"Training seconds: {runtime:.1f}")


if __name__ == "__main__":
    main()
