"""Summarize independently verified coastal-loss pairs without pooling contexts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", default=["seed15_oct", "seed16_oct", "seed15_jan"])
    args = parser.parse_args()
    result = {}
    for context in args.contexts:
        directory = args.report / ("lbfgs-coastal" if context == "seed15_oct" else f"coastal-confirm/{context}")
        maps = json.loads((directory / "verification.json").read_text())["arms"]
        objective = json.loads((directory / "objective-verification.json").read_text())["arms"]
        control, treatment = maps["interior"], maps["wet_neighbors"]
        for arm, values in maps.items():
            for key in ("state_mse", "highpass_mse"):
                np.testing.assert_allclose(values[key], objective[arm][key], rtol=1e-10)
        state_ratios = np.array(treatment["per_channel_mse"]) / control["per_channel_mse"]
        result[context] = {
            "state_mse_reduction": control["state_mse"] / treatment["state_mse"],
            "boundary_state_contribution_reduction": objective["interior"]["boundary_state_mse_contribution"] / objective["wet_neighbors"]["boundary_state_mse_contribution"],
            "interior_state_contribution_ratio": objective["wet_neighbors"]["interior_state_mse_contribution"] / objective["interior"]["interior_state_mse_contribution"],
            "interior_highpass_mse_ratio": treatment["highpass_mse"] / control["highpass_mse"],
            "channels_with_lower_state_mse": int((state_ratios < 1).sum()),
            "per_channel_state_ratio_min_max": [float(state_ratios.min()), float(state_ratios.max())],
            "arms": {
                arm: {
                    **{key: objective[arm][key] for key in ("state_mse", "highpass_mse", "boundary_state_mse_contribution", "interior_state_mse_contribution", "closure_evaluations", "measured_seconds")},
                    "worst_global_hp_error_over_target_rms": max(values["per_channel_hp_error_over_target_rms"]),
                    "state_mse_contribution_fraction_by_group": {
                        group: float(np.array(values["per_channel_mse"])[indices].sum() / np.sum(values["per_channel_mse"]))
                        for group, indices in {
                            "U": slice(0, 19), "V": slice(19, 38), "T": slice(38, 57),
                            "S": slice(57, 76), "SSH": slice(76, 77),
                            "T_shallowest_two": slice(38, 40),
                        }.items()
                    },
                    "quiet_patch": {field: values["selected_fields"][field]["patch"] for field in ("T1850", "S1850", "SSH")},
                } for arm, values in maps.items()
            },
        }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({
        "scope": "Matched independent single-example fits; fixed continuation budget, not converged minima or held-out forecasting. Ratios compare added wet-boundary term with the unchanged interior-only term. Raw-map CPU float64 verification is prerequisite.",
        "contexts": result,
    }, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
    x = np.arange(len(result))
    for offset, (arm, label) in enumerate((("interior", "Interior penalty"), ("wet_neighbors", "+ wet-boundary penalty"))):
        for ax, metric, title in zip(axes[:2], ("state_mse", "highpass_mse"), ("All wet-cell state MSE", "Wet-interior texture MSE")):
            ax.bar(x + (offset - .5) * .36, [r["arms"][arm][metric] for r in result.values()], width=.36, label=label)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_title(title)
            ax.set_ylabel("Normalized squared error · lower is better")
        axes[2].bar(x + (offset - .5) * .36, [r["arms"][arm]["quiet_patch"]["S1850"]["hp_error_over_increment_rms"] for r in result.values()], width=.36, label=label)
    axes[2].axhline(1, color="black", ls="--", lw=1)
    axes[2].set_title("Quiet deep-salinity limitation")
    axes[2].set_ylabel("Texture error RMS / true increment texture RMS")
    for ax in axes:
        ax.set_xticks(x, [name.replace("_", " ") for name in result])
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Boundary coverage improves state fit; check interior texture and tiny increments separately")
    fig.savefig(args.output / "comparison.png", dpi=160)
    plt.close(fig)
    print({context: {key: value for key, value in entry.items() if key != "arms"} for context, entry in result.items()})


if __name__ == "__main__":
    main()
