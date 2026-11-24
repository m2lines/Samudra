"""Illustrate representability with raw-grid fields, explicitly unequal training budgets."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from verify_maps import highpass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--spatial-pretrain", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline_protocol = json.loads((args.baseline / "protocol.json").read_text())
    spatial_protocol = json.loads((args.spatial_pretrain / "protocol.json").read_text())
    for key in ("seed", "parameters", "backbone_initial_sha256", "base_first_channel", "zero_head"):
        assert baseline_protocol[key] == spatial_protocol[key]
    assert baseline_protocol["base_first_channel"] is None
    assert baseline_protocol["zero_head"] is False
    prepared = np.load(args.data / "prognostic.npy", mmap_mode="r")
    truth, previous = prepared[2].astype(np.float64), prepared[1].astype(np.float64)
    wet = np.load(args.data / "mask.npy").astype(bool)
    fields = {}
    for name, path in (("Ordinary MSE + Adam", args.baseline / "maps-5000.npz"),
                       ("Spatial loss + continued optimization", args.candidate)):
        with np.load(path) as saved:
            assert np.array_equal(saved["truth"][0], truth)
            assert np.array_equal(saved["mask"], wet)
            fields[name] = saved["prediction"][0].astype(np.float64)
    meta = json.loads((args.data / "metadata.json").read_text())
    scale, center = np.array(meta["stds"][:77]), np.array(meta["means"][:77])
    inside = wet.copy()
    for axis, shift in ((-2, -1), (-2, 1), (-1, -1), (-1, 1)):
        inside &= np.roll(wet, shift, axis)
    inside[:, (0, -1), :] = False
    target_hp = highpass(truth)
    increment_hp = highpass(truth - previous)
    target_hp_mse = (target_hp ** 2 * inside).sum((-2, -1)) / inside.sum((-2, -1))
    hp_errors = {name: highpass(prediction - truth) for name, prediction in fields.items()}
    aggregate = {}
    for name, prediction in fields.items():
        state = ((prediction - truth) ** 2 * wet).sum((-2, -1)) / wet.sum((-2, -1))
        texture = (hp_errors[name] ** 2 * inside).sum((-2, -1)) / inside.sum((-2, -1))
        aggregate[name] = {
            "state_mse": float(state.mean()), "highpass_mse": float(texture.mean()),
            "physical_rmse_by_channel": (np.sqrt(state) * scale).tolist(),
            "global_texture_error_over_target_rms_by_channel": np.sqrt(texture / target_hp_mse).tolist(),
        }
    patch = (slice(50, 82), slice(200, 232))
    selected = [(13, "Zonal velocity · 1850 m", "m/s"),
                (32, "Meridional velocity · 1850 m", "m/s"),
                (51, "Temperature · 1850 m", "°C"),
                (70, "Salinity · 1850 m", "psu"), (76, "Sea surface height", "m")]
    fig, axes = plt.subplots(5, 3, figsize=(12, 15), constrained_layout=True)
    metrics = {}
    for row, (channel, label, unit) in enumerate(selected):
        assert wet[channel][patch].all()
        arrays = [a[channel][patch] * scale[channel] + center[channel]
                  for a in (truth, *fields.values())]
        lo, hi = min(a.min() for a in arrays), max(a.max() for a in arrays)
        for ax, array in zip(axes[row], arrays):
            im = ax.imshow(array, origin="lower", interpolation="nearest", cmap="viridis", vmin=lo, vmax=hi)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes[row].tolist(), shrink=.75, label=unit)
        axes[row, 0].set_ylabel(label)
        metrics[label] = {}
        for name, prediction in fields.items():
            error = prediction - truth
            hp_error = hp_errors[name][channel][patch]
            hp_target = target_hp[channel][patch]
            hp_increment = increment_hp[channel][patch]
            metrics[label][name] = {
                "physical_rmse": float(np.sqrt(np.mean(error[channel][patch] ** 2)) * scale[channel]),
                "unit": unit,
                "texture_error_over_target_rms": float(np.sqrt(np.mean(hp_error ** 2) / np.mean(hp_target ** 2))),
                "texture_error_over_increment_rms": float(np.sqrt(np.mean(hp_error ** 2) / np.mean(hp_increment ** 2))),
            }
    for ax, title in zip(axes[0], ("Target", *fields.keys())):
        ax.set_title(title, fontsize=10)
    fig.suptitle("Same absolute-output architecture and October seed16; quiet South Pacific patch\n"
                 "Common physical color scale in each row; raw cells, no smoothing\n"
                 "Different objectives and optimization budgets: representability illustration, not a matched-cost test", fontsize=11)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "before-after.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    for name, metrics_all in aggregate.items():
        ax.semilogy(metrics_all["global_texture_error_over_target_rms_by_channel"], marker=".", label=name)
    ax.axhline(1, color="black", lw=1, ls="--", label="Error texture equals target texture")
    for edge in (18.5, 37.5, 56.5, 75.5):
        ax.axvline(edge, color="gray", lw=.8, alpha=.5)
    ax.set_xticks((9, 28, 47, 66, 76), ("U · 19 depths", "V · 19 depths", "T · 19 depths", "S · 19 depths", "SSH"))
    ax.set_ylabel("Global wet-interior texture error / target RMS")
    ax.set_title("All 77 outputs; same architecture, unequal optimization budgets\nGlobal ratios do not bound error in every quiet patch")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    fig.savefig(args.output / "all-depths.png", dpi=160)
    plt.close(fig)
    (args.output / "before-after.json").write_text(json.dumps({
        "scope": "Raw-grid representability illustration with unequal objectives/optimization budgets. Same seed, architecture, initial backbone hash and prepared truth/mask. This is not a causal one-factor or equal-compute comparison; those are reported separately.",
        "baseline": str(args.baseline), "candidate": str(args.candidate),
        "patch": "rows50:82 cols200:232; fixed from initial diagnostic, not selected for this result",
        "fields": metrics, "all_channels": aggregate,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
