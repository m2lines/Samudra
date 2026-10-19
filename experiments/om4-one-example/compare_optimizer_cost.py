"""Compare same-checkpoint continuations by gradient evaluations and wall time."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    configurations = [
        ("adam-spatial/lr_0.00001", "Adam 1e-5, spatial loss", "adam"),
        ("adam-spatial/lr_0.0001", "Adam 1e-4, spatial loss", "adam"),
        ("lbfgs-spatial/weight_0", "L-BFGS, MSE only", "lbfgs"),
        ("lbfgs-spatial/weight_100", "L-BFGS, spatial loss", "lbfgs"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    summary = {}
    initial = None
    initial_arrays = None
    for relative, label, optimizer in configurations:
        directory = args.root / relative
        assert (directory / "COMPLETE").exists()
        history = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
        assert len({row["step"] for row in history}) == len(history), "Audit resume duplicates first"
        start = [history[0][key] for key in ("eval_mse", "highpass_mse")]
        if initial is None:
            initial = start
        assert start == initial
        with np.load(directory / "maps-0.npz") as maps:
            if initial_arrays is None:
                initial_arrays = {key: maps[key].copy() for key in maps.files}
            assert all(np.array_equal(maps[key], value) for key, value in initial_arrays.items())
        # Each saved diagnostic adds one backward pass outside the optimizer.
        # L-BFGS func_evals already counts all line-search closure evaluations.
        evaluations = [
            row["logical_gradient_evaluations"] if optimizer == "adam"
            else row["closure_evaluations"] + index + 1
            for index, row in enumerate(history)
        ]
        seconds = [row["seconds"] for row in history]
        for column, key in enumerate(("eval_mse", "highpass_mse")):
            values = [row[key] for row in history]
            axes[0, column].loglog(evaluations, values, label=label, marker=".", markersize=2)
            axes[1, column].semilogy(seconds, values, label=label)
        last = history[-1]
        summary[label] = {
            "state_mse": last["eval_mse"], "highpass_mse": last["highpass_mse"],
            "gradient_evaluations_including_measurements": evaluations[-1],
            "measured_seconds": seconds[-1],
            "training_objective": last["training_objective"],
            "objective_gradient_norm": last["objective_gradient_norm"],
        }
    for row in range(2):
        axes[row, 0].set_ylabel("Normalized wet-cell state MSE")
        axes[row, 1].set_ylabel("Interior high-pass error MSE")
        for column in range(2):
            axes[row, column].set_xlabel("Gradient evaluations" if row == 0 else "Measured seconds")
            axes[row, column].grid(alpha=.2)
    axes[0, 1].legend(fontsize=8)
    figure.suptitle("Same absolute-output checkpoint; fixed BatchNorm; all 77 outputs\n"
                   "Spatial loss = state MSE + 100 × curvature-error MSE")
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "optimizer-cost.png", dpi=170)
    plt.close(figure)
    (args.output / "optimizer-cost.json").write_text(json.dumps({
        "endpoints": summary,
        "notes": "Counts include diagnostic backward passes and L-BFGS line searches. "
                 "No duplicate measurement steps were observed. Wall time includes diagnostics and checkpoint writes. "
                 "Optimizer update counts alone are not comparable; final evaluation budgets differ.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
