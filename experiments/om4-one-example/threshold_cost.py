"""Display threshold controls by measured work, including early stopping differences."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initial-steps", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    histories = {}
    for arm in ("default", "lowered"):
        all_rows = [json.loads(line) for line in (args.root / arm / "history.jsonl").read_text().splitlines()]
        rows = [row for row in all_rows if row["step"] >= args.initial_steps and "curvature_candidates" in row]
        assert rows[0]["step"] == args.initial_steps and rows[-1]["step"] == 2500
        first = rows[0]
        histories[arm] = [row | {
            "extra_closures": row["closure_evaluations"] - first["closure_evaluations"],
            "extra_seconds": row["seconds"] - first["seconds"],
        } for row in rows]
    control = histories["default"][-1]
    comparison = {}
    for budget in ("extra_closures", "extra_seconds"):
        treatment = max((r for r in histories["lowered"] if r[budget] <= control[budget]), key=lambda r: r[budget])
        comparison[budget] = {
            "control_budget": control[budget], "lowered_budget": treatment[budget],
            "lowered_step": treatment["step"],
            "control_over_lowered_state_mse": control["eval_mse"] / treatment["eval_mse"],
            "control_over_lowered_highpass_mse": control["highpass_mse"] / treatment["highpass_mse"],
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cost.json").write_text(json.dumps({
        "scope": "Intermediate training-log measurements; final maps independently verified separately. Last logged lowered-threshold point at or below the default endpoint budget, not interpolated or exactly equal compute. Closure counts exclude periodic diagnostic gradients; measured time includes periodic checkpoints.",
        "within_default_budget": comparison,
        "endpoints": {arm: {key: rows[-1][key] for key in ("extra_closures", "extra_seconds", "curvature_candidates", "curvature_accepted")}
                      for arm, rows in histories.items()},
    }, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for arm, rows in histories.items():
        x = [r["extra_closures"] for r in rows]
        for ax, key, label in zip(axes[:2], ("eval_mse", "highpass_mse"), ("Wet-cell state MSE", "Interior high-pass error MSE")):
            ax.semilogy(x, [r[key] for r in rows], label=arm)
            ax.set_ylabel(label)
        axes[2].plot(x[1:], [r["curvature_accepted"] / max(r["curvature_candidates"], 1) for r in rows[1:]], label=arm)
    axes[2].set_ylabel("Fraction of curvature updates accepted")
    axes[2].set_ylim(-.03, 1.03)
    for ax in axes:
        ax.set_xlabel("Additional L-BFGS closure evaluations")
        ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Same starting state and objective; only curvature acceptance threshold differs")
    fig.savefig(args.output / "cost.png", dpi=160)
    plt.close(fig)
    print(comparison)


if __name__ == "__main__":
    main()
