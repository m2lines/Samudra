"""Summarize matched objective continuations across seed/date fitting contexts."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_history(directory):
    assert (directory / "COMPLETE").exists()
    rows = [json.loads(line) for line in (directory / "history.jsonl").read_text().splitlines()]
    assert rows[-1]["step"] == 1000
    assert len({row["step"] for row in rows}) == len(rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--october-data", type=Path, required=True)
    parser.add_argument("--january-data", type=Path, required=True)
    args = parser.parse_args()
    output = args.report / "spatial-confirmation"
    output.mkdir(parents=True, exist_ok=True)
    contexts = [
        ("seed15_oct", "Oct 20, 2014 · seed 15", args.results / "lbfgs-spatial", args.report / "lbfgs-spatial", args.october_data),
        ("seed16_oct", "Oct 20, 2014 · seed 16", args.results / "spatial-confirm-lbfgs/seed16_oct", output / "seed16_oct", args.october_data),
        ("seed15_jan", "Jan 23, 2015 · seed 15", args.results / "spatial-confirm-lbfgs/seed15_jan", output / "seed15_jan", args.january_data),
    ]
    summary = {}
    texture_figure, texture_axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, layout="constrained")
    for context_index, (name, title, root, verified_report, data) in enumerate(contexts):
        verification = json.loads((verified_report / "verification.json").read_text())["arms"]
        protocols = [json.loads((root / arm / "protocol.json").read_text()) for arm in ("weight_0", "weight_100")]
        assert protocols[0]["source"] == protocols[1]["source"]
        assert all(protocol["parameters"] == 84066474 for protocol in protocols)
        assert [protocol["weight"] for protocol in protocols] == [0, 100]
        assert protocols[0]["mode"] == protocols[1]["mode"]
        histories = [read_history(root / arm) for arm in ("weight_0", "weight_100")]
        for key in ("eval_mse", "highpass_mse"):
            assert histories[0][0][key] == histories[1][0][key]
        with np.load(root / "weight_0/maps-0.npz") as a, np.load(root / "weight_100/maps-0.npz") as b:
            assert set(a.files) == set(b.files)
            assert all(np.array_equal(a[key], b[key]) for key in a.files)
        maps = []
        for arm in ("weight_0", "weight_100"):
            with np.load(root / arm / "maps-1000.npz") as saved:
                maps.append({key: saved[key].copy() for key in saved.files})
        assert np.array_equal(maps[0]["truth"], maps[1]["truth"])
        assert np.array_equal(maps[0]["mask"], maps[1]["mask"])
        target = maps[0]["truth"][0]
        mask = maps[0]["mask"]
        meta = json.loads((data / "metadata.json").read_text())
        std, mean = np.asarray(meta["stds"][:77]), np.asarray(meta["means"][:77])
        base, spatial = verification["weight_0"], verification["weight_100"]
        summary[name] = {
            "title": title,
            "state_mse_improvement": base["state_mse"] / spatial["state_mse"],
            "highpass_mse_improvement": base["highpass_mse"] / spatial["highpass_mse"],
            "arms": {},
        }
        for index, (arm, label, color) in enumerate((("weight_0", "MSE only", "C0"), ("weight_100", "MSE + spatial error", "C1"))):
            metrics = verification[arm]
            ratios = np.asarray(metrics["per_channel_hp_error_over_target_rms"])
            history = histories[index]
            summary[name]["arms"][arm] = {
                "state_mse": metrics["state_mse"], "highpass_mse": metrics["highpass_mse"],
                "max_global_hp_error_over_target_rms": float(ratios.max()),
                "selected_fields": metrics["selected_fields"],
                "gradient_evaluations_including_measurements": history[-1]["closure_evaluations"] + len(history),
                "measured_seconds": history[-1]["seconds"],
            }
            texture_axes[context_index].semilogy(np.arange(77), ratios, label=label, color=color)
        ax = texture_axes[context_index]
        ax.axhline(1, color="gray", linestyle=":", label="Error equals target structure")
        ax.set_xticks([9, 28, 47, 66, 76], ["U", "V", "T", "S", "SSH"])
        for edge in (18.5, 37.5, 56.5, 75.5):
            ax.axvline(edge, color="gray", alpha=.15)
        ax.set_title(title)
        ax.set_xlabel("All 77 outputs; depths increase within each group")
        ax.grid(axis="y", alpha=.2)
        figure, axes = plt.subplots(3, 5, figsize=(16, 9), layout="constrained")
        patch = (slice(50, 82), slice(200, 232))
        for row, (channel, label) in enumerate(((51, "Temperature 1850m (°C)"), (70, "Salinity 1850m (psu)"), (76, "SSH (m)"))):
            assert mask[channel][patch].all()
            truth = target[channel][patch].astype(np.float64) * std[channel] + mean[channel]
            predictions = [item["prediction"][0, channel][patch].astype(np.float64) * std[channel] + mean[channel] for item in maps]
            errors = [prediction - truth for prediction in predictions]
            state_min = min(values.min() for values in [truth, *predictions])
            state_max = max(values.max() for values in [truth, *predictions])
            error_limit = max(np.abs(values).max() for values in errors)
            for column, values in enumerate([truth, *predictions, *errors]):
                image = axes[row, column].imshow(values, origin="lower", interpolation="nearest",
                    cmap="viridis" if column < 3 else "RdBu_r",
                    vmin=state_min if column < 3 else -error_limit,
                    vmax=state_max if column < 3 else error_limit)
                figure.colorbar(image, ax=axes[row, column], shrink=.72)
                axes[row, column].set_xticks([])
                axes[row, column].set_yticks([])
            axes[row, 0].set_ylabel(label)
        for ax, label in zip(axes[0], ("Target", "MSE-only prediction", "Spatial-loss prediction", "MSE-only error", "Spatial-loss error")):
            ax.set_title(label)
        figure.suptitle(title + " · separate single-example fit; unchanged absolute-output architecture\n"
            "Matched initial checkpoint and L-BFGS protocol. Raw grids; common state/error scales within each row.")
        figure.savefig(output / f"{name}-matched-patch.png", dpi=170)
        plt.close(figure)
    texture_axes[0].set_ylabel("High-pass error RMS / target high-pass RMS")
    texture_axes[-1].legend(fontsize=8)
    texture_figure.suptitle("Original absolute-output architecture: matched L-BFGS objective comparison\n"
        "Global wet interiors; center minus four-neighbor mean. Each context is fitted separately.")
    texture_figure.savefig(output / "all-depth-texture.png", dpi=170)
    plt.close(texture_figure)
    (output / "summary.json").write_text(json.dumps({
        "contexts": summary,
        "interpretation": "Each context is a separate one-example fit, not held-out validation or rollout skill. "
            "Within each context both objectives start from exactly identical spatial-Adam checkpoint predictions. "
            "No output filtering; CPU verification supplies all reported error metrics. "
            "Neither endpoint is claimed to be a converged minimum.",
    }, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in row.items() if key != "arms"} for name, row in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
