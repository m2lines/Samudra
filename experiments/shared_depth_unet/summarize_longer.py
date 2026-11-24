# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Report the doubled-update controls only after all six fits finish."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "outputs/longer"
DEST = BASE / "results_longer"


def main():
    original = json.loads((BASE / "outputs/main/results.json").read_text())
    longer = json.loads((SOURCE / "results.json").read_text())
    assert len(longer) == 6
    groups = {
        "Independent + context, 2k": [
            r for r in original if r["arm"] == "independent_context"
        ],
        "Shared + context, 2k": [r for r in original if r["arm"] == "shared_context"],
        "Independent + context, 4k continuation": [
            r for r in longer if r["variant"] == "continue_floor"
        ],
        "Independent + context, 4k cosine": [
            r for r in longer if r["variant"] == "cosine_4000"
        ],
    }
    for key in groups:
        groups[key].sort(key=lambda r: r["seed"])
        assert [r["seed"] for r in groups[key]] == [0, 1, 2]
    DEST.mkdir(exist_ok=True)
    for name in [
        "results.json",
        "provenance.json",
        "gradient_paths.json",
        "COMPLETE.json",
    ]:
        shutil.copy2(SOURCE / name, DEST / name)

    def vals(key, lead, metric):
        return np.array([r["test"][str(lead)][metric] for r in groups[key]])

    def fmt(v):
        return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}"

    table = [
        "| Model / budget | Coarse RMSE, step 1 | Fine RMSE, step 1 | Fine RMSE, step 5 | Selected updates |",
        "|---|---:|---:|---:|---|",
    ]
    for key, rows in groups.items():
        table.append(
            f"| {key} | {fmt(vals(key, 1, 'coarse'))} | {fmt(vals(key, 1, 'fine'))} | {fmt(vals(key, 5, 'fine'))} | {', '.join(str(r['best_step']) for r in rows)} |"
        )
    comparison = []
    for key in list(groups)[2:]:
        for lead in [1, 5]:
            change = 100 * (
                vals(key, lead, "fine") / vals("Shared + context, 2k", lead, "fine") - 1
            )
            comparison.append(
                f"- {key}, lead {lead}, versus shared + context at 2k: mean paired change {change.mean():+.2f}%; seeds {', '.join(f'{v:+.2f}%' for v in change)}. Positive means independent is worse."
            )
    gradients = json.loads((SOURCE / "gradient_paths.json").read_text())
    gt = [
        "| Arm | Fine-loss gradient norm in global core | With context detached |",
        "|---|---:|---:|",
    ]
    for key, v in gradients.items():
        gt.append(
            f"| {key} | {v['fine_loss_global_core_gradient_norm']:.6f} | {v['with_context_detached_gradient_norm']:.6f} |"
        )
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    colors = ["tab:red", "tab:green", "tab:orange", "tab:blue"]
    for key, color in zip(groups, colors):
        rows = groups[key]
        histories = []
        for r in rows:
            directory = (
                SOURCE / f"{r['variant']}_seed{r['seed']}"
                if "variant" in r
                else BASE / f"outputs/main/{r['arm']}_seed{r['seed']}"
            )
            histories.append(json.loads((directory / "history.json").read_text()))
        x = [v["step"] for v in histories[0]]
        y = np.array([[v["val_score"] for v in h] for h in histories])
        axs[0].plot(x, y.mean(0), color=color, label=key)
        means = [vals(key, k, "fine").mean() for k in [1, 3, 5]]
        sd = [vals(key, k, "fine").std(ddof=1) for k in [1, 3, 5]]
        axs[1].errorbar(
            [1, 3, 5], means, yerr=sd, marker="o", color=color, label=key, capsize=3
        )
    axs[0].set(
        title="Validation objective (mean of 3 seeds)",
        xlabel="Optimizer updates",
        ylabel="Mean normalized MSE",
    )
    axs[0].axvline(2000, color="gray", linestyle=":", linewidth=1)
    axs[1].set(
        title="Held-out regional forecast (mean ± seed SD)",
        xlabel="Autoregressive steps",
        ylabel="Fine RMSE",
    )
    axs[0].legend(fontsize=7)
    for ax in axs:
        ax.grid(alpha=0.2)
    fig.savefig(DEST / "comparison.png", dpi=130)
    plt.close(fig)
    continuation = [r for r in longer if r["variant"] == "continue_floor"]
    report = f"""<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Does independent + context catch up with twice the updates?

Follow-up to the original four-arm local vorticity study. Experiment/results record, not intended to merge.

## Held-out results

Mean ± sample SD across the same three training seeds. Same dataset, architecture, validation objective, test examples, and per-update coarse/fine exposures as the original study. Each 4k run receives twice the original task exposure and approximately twice its training compute. The shared 2k reference is unchanged.

{chr(10).join(table)}

{chr(10).join(comparison)}

![Learning curves and regional rollout errors](comparison.png)

## Reading the result

The 4k controls improve on the original independent-context mean, with the stretched cosine schedule recovering most of the original one-step fine-error gap. Its mean fine RMSE remains 0.3311 versus 0.3141 for shared at 2k, and its coarse mean essentially matches the shared reference (0.2958 versus 0.2966). The cosine control beats the shared fine reference in one seed and trails in two, at both one and five steps. Seed variation is substantial; this does not establish a statistically reliable remaining gap.

The original 2k result therefore supports a fixed-budget advantage, not an irreducible advantage of sharing. More independent training recovers much of the difference. The gradient audit independently confirms that shared coarse blocks receive direct fine-example updates, but this longer-training comparison does not prove that those gradients, rather than other effects of tying parameters, cause the accuracy difference.

## What the two longer controls change

- **4k continuation:** Replay the original 2,000 updates with the same example schedule and learning-rate schedule, keeping the resulting AdamW state. Continue for 2,000 more updates at the final learning-rate floor, 0.0002. The original study saved selected model weights but not final optimizer state, so replay is necessary for a genuine optimizer continuation.
- **4k cosine:** Train from the same initial weights and same 4,000-example schedule, but stretch the cosine learning-rate decay from 0.001 to 0.0002 across 4,000 updates. This changes the early learning rate and therefore is a fresh budget-matched control rather than an exact continuation.
- Prefix and extension schedules are generated separately and concatenated. Calling the original schedule generator with 4,000 instead of 2,000 changes the prefix's time/crop draws, so doing that would not preserve the original examples.
- The continuation replays select the same 2k checkpoints as the original. Maximum validation-objective discrepancy over all seeds/checkpoints is {max(r["prefix_max_validation_difference"] for r in continuation):.3g}. Maximum output discrepancy on a fixed validation batch from those selected checkpoints is {max(r["prefix_max_prediction_difference"] for r in continuation):.3g}. GPU execution is numerically close, not bitwise identical; the initial strict-bitwise bring-up attempt stopped before any continuation and is excluded from this comparison.
- Checkpoint selection examines one-step validation loss every 100 updates, including the original first 2,000 updates. The held-out test is evaluated only on the selected model. Three- and five-step predictions are autoregressive with no truth refresh.

## Why sharing is not simply extra forward passes

Both context arms execute a global U-Net core once and a regional U-Net core once per update. Both take one AdamW optimizer step on the mean of coarse and fine losses. The shared arm uses the same parameter tensors in both executions; the independent arm has two parameter sets. Fine gradients are summed into shared weights from both regional and global-context paths before that single optimizer step.

We backpropagated the fine loss alone through each original seed-zero selected checkpoint on the same fixed batch. We then repeated it with the global context tensor detached, keeping the local prediction graph intact:

{chr(10).join(gt)}

Independent global-core weights receive fine gradients only through context; detaching context removes them. Shared global-core weights also participate directly in the regional computation, so fine gradients remain after detaching context. Gradient magnitudes are diagnostic of connectivity, not proof that one path causes the accuracy gain, and are not comparable measures of learning quality between separately trained models.

## Interpretation limits

This tests whether extra optimization/task exposure closes the observed gap at a specified budget. It does not isolate the causal benefit of fine-to-coarse transfer from regularization, optimization conditioning, or shared feature semantics. There is no shared-at-4k arm here. Both independent-context and shared-context already learn from fine examples via the context path; sharing adds direct regional gradients and also exposes the regional weights directly to coarse supervision.

The three seeds share one synthetic dataset, so these are descriptive seed comparisons, not confidence intervals over ocean data. Neither 4k schedule proves asymptotic convergence. The old limitations about fine-detail smoothing, patch boundaries, paired data, and long-rollout stability remain.

## Artifacts and checks

All six fits completed, totaling {sum(r["seconds"] for r in longer):.1f} seconds of training/validation on the GB10, excluding final held-out tests and the stopped prefix-check bring-up. Full checkpoints, final optimizer states, and per-run histories remain locally under `outputs/longer/`. Adjacent JSON files contain raw metrics, gradient checks, and provenance, including the original dataset hash. The executed source is the checked-in `longer.py` at its recorded SHA256.

Reproduce from the repository root in a PyTorch CUDA environment with `PYTHONPATH=src`, after generating the original study's `outputs/main/` artifacts:

```bash
python experiments/shared_depth_unet/longer.py
python experiments/shared_depth_unet/summarize_longer.py
```

Use a fresh `outputs/longer/` directory. The script refuses to overwrite completed fits.
"""
    (DEST / "REPORT.md").write_text(report)
    for p in DEST.iterdir():
        if p.suffix in [".json", ".png"]:
            Path(str(p) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
    print("\n".join(table))
    print("\n".join(comparison))


if __name__ == "__main__":
    main()
