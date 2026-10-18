"""Compare a latest-input residual fit against persistence and actual increments."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--map", type=Path, help="Alternative final map; persistence taken from prepared input")
    parser.add_argument("--label", default="Residual model")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    prepared = np.load(args.data / "prognostic.npy", mmap_mode="r")
    if args.map:
        assert not args.root, "Provide either --root or --map"
        initial, truth = prepared[1], prepared[2]
        wet = np.load(args.data / "mask.npy")
        final_map = args.map
    else:
        assert args.root, "Provide --root or --map"
        with np.load(args.root / "maps-0.npz") as z:
            initial, truth, wet = z["prediction"][0], z["truth"][0], z["mask"]
        assert np.array_equal(initial, prepared[1]), "Reference must be latest-input persistence"
        assert np.array_equal(truth, prepared[2]), "Target differs from prepared first target"
        final_map = args.root / "maps-5000.npz"
    with np.load(final_map) as z:
        assert np.array_equal(z["truth"][0], truth) and np.array_equal(z["mask"], wet)
        prediction = z["prediction"][0].astype(np.float64)
    initial, truth = initial.astype(np.float64), truth.astype(np.float64)
    shifts = [(-2, -1), (-2, 1), (-1, -1), (-1, 1)]
    inside = wet.copy()
    for axis, shift in shifts:
        inside &= np.roll(wet, shift, axis)
    inside[:, 0] = False
    inside[:, -1] = False

    def highpass(x):
        return x - sum(np.roll(x, shift, axis) for axis, shift in shifts) / 4

    change, learned, error = truth - initial, prediction - initial, prediction - truth
    change_hp, error_hp = highpass(change), highpass(error)
    metrics = {}
    for channel, name in [(13, "U1850"), (32, "V1850"), (51, "T1850"),
                          (70, "S1850"), (76, "SSH")]:
        metrics[name] = {}
        patch = (slice(50, 82), slice(200, 232))
        assert inside[channel][patch].all()
        for domain, selection in [("global", inside[channel]), ("patch", patch)]:
            e, d, p = [x[channel][selection].ravel() for x in (error, change, learned)]
            hd, he = [x[channel][selection].ravel() for x in (change_hp, error_hp)]
            metrics[name][domain] = {
                "mse_over_persistence": float(np.mean(e**2) / np.mean(d**2)),
                "increment_correlation": float(np.corrcoef(p, d)[0, 1]),
                "learned_over_true_increment_rms": float(np.sqrt(np.mean(p**2) / np.mean(d**2))),
                "hp_error_over_true_increment_rms": float(np.sqrt(np.mean(he**2) / np.mean(hd**2))),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "label": args.label, "metrics": metrics,
        "notes": "Reference and target verified bitwise against prepared frames1/2. "
                 "Ratios use actual five-day increments, not total-state variance. "
                 "Global excludes mask edges; fixed patch rows50:82,cols200:232. "
                 "Correlations flatten each selected domain before computation.",
    }, indent=2) + "\n")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metadata = json.loads((args.data / "metadata.json").read_text())
        std = np.asarray(metadata["stds"][:77])
        selected = [(13, "U1850m (m/s)"), (32, "V1850m (m/s)"),
                    (51, "T1850m (°C)"), (70, "S1850m (psu)"), (76, "SSH (m)")]
        fig, axes = plt.subplots(5, 3, figsize=(11, 14), layout="constrained")
        for row, (channel, name) in enumerate(selected):
            fields = [x[channel][patch] * std[channel] for x in (change, learned, error)]
            limit = max(float(np.abs(x).max()) for x in fields)
            for col, values in enumerate(fields):
                im = axes[row, col].imshow(values, origin="lower", interpolation="nearest",
                                          cmap="RdBu_r", vmin=-limit, vmax=limit)
                fig.colorbar(im, ax=axes[row, col], shrink=.7)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            axes[row, 0].set_ylabel(name)
        for ax, title in zip(axes[0], ("True five-day change", "Learned correction", "Prediction error")):
            ax.set_title(title)
        fig.suptitle(args.label + ": learned change in the fixed quiet-ocean patch\n"
                     "Same physical scale within each row; no interpolation or output smoothing")
        fig.savefig(args.output.with_suffix(".png"), dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    main()
