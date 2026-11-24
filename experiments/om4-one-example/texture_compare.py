"""Compare residual texture with target structure on identical saved grid cells.

Uses CPU float64 reductions and the original (uninterpolated) arrays. The patch
diagnostic is descriptive, not a new training loss or an independent validation.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SELECTED = [(13, "U 1850 m"), (32, "V 1850 m"), (51, "T 1850 m"),
            (70, "S 1850 m"), (76, "SSH")]
PATCH = (slice(50, 82), slice(200, 232))
SHIFTS = [(-2, -1), (-2, 1), (-1, -1), (-1, 1)]


def highpass(x):
    return x - sum(np.roll(x, shift, axis) for axis, shift in SHIFTS) / 4


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    std = np.asarray(metadata["stds"][:77], dtype=np.float64)
    with np.load(args.reference) as z:
        target = z["truth"][0].astype(np.float64)
        wet = z["mask"].astype(bool)
    interior = wet.copy()
    for axis, shift in SHIFTS:
        interior &= np.roll(wet, shift, axis)
    interior[:, 0] = False
    interior[:, -1] = False
    target_hp = highpass(target)
    files = {"original": args.reference}
    for arm in sorted(args.root.iterdir()):
        if not (arm / "COMPLETE").exists():
            continue
        maps = list(arm.glob("maps-*.npz"))
        if maps:
            files[arm.name] = max(maps, key=lambda p: int(p.stem.split("-")[-1]))

    rows = []
    for name, path in files.items():
        with np.load(path) as z:
            assert np.array_equal(z["truth"][0], target), path
            assert np.array_equal(z["mask"], wet), path
            prediction = z["prediction"][0].astype(np.float64)
        assert np.isfinite(prediction[wet]).all(), path
        error = prediction - target
        error_hp = highpass(error)
        prediction_hp = highpass(prediction)
        for c, field in SELECTED:
            assert interior[c][PATCH].all(), (c, "Patch includes a mask edge")
            for domain, selection in (("global wet interior", interior[c]),
                                      ("South Pacific patch", PATCH)):
                e, h, t = error[c][selection], error_hp[c][selection], target_hp[c][selection]
                p = prediction_hp[c][selection]
                rows.append({
                    "arm": name, "field": field, "domain": domain,
                    "rmse_physical": rms(e) * std[c],
                    "bias_physical": float(e.mean()) * std[c],
                    "centered_rmse_physical": rms(e - e.mean()) * std[c],
                    "highpass_error_rms_physical": rms(h) * std[c],
                    "highpass_target_rms_physical": rms(t) * std[c],
                    "highpass_error_to_target_rms": rms(h) / rms(t),
                    "highpass_prediction_to_target_rms": rms(p) / rms(t),
                })
    with (args.output / "texture.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # One threshold has a literal meaning: residual fine-scale RMS equals the
    # target fine-scale RMS. It is not a criterion for acceptable forecasts.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    for ax, domain in zip(axes, ("global wet interior", "South Pacific patch")):
        for name in files:
            values = [r["highpass_error_to_target_rms"] for r in rows
                      if r["arm"] == name and r["domain"] == domain]
            ax.semilogy(range(len(SELECTED)), values, marker="o", label=name,
                        linewidth=2.5 if name == "original" else 1, alpha=.85)
        ax.axhline(1, color="black", linestyle="--", linewidth=.8)
        ax.set_xticks(range(len(SELECTED)), [field for _, field in SELECTED])
        ax.set_title(domain)
        ax.set_ylabel("High-pass residual RMS / target RMS")
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Texture relative to real target structure\n"
                 "Center minus four neighbors; same cells in every comparison")
    fig.savefig(args.output / "texture-relative-to-target.png", dpi=180)
    plt.close(fig)
    protocol = vars(args) | {
        "files": files, "dtype": "float64 analysis; source arrays unchanged",
        "patch_rows": [50, 82], "patch_columns": [200, 232],
        "boundary": "periodic longitude; exclude latitude endpoints and mask edges",
        "selection": "preexisting fixed South Pacific patch; all completed arms",
        "notes": "High-pass is a local curvature operator, not a spectral cutoff. "
                 "Ratios describe visible error relative to target texture; they "
                 "do not isolate the cause. Absolute RMSE and bias are in CSV.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, default=str, indent=2))
    print(f"Wrote {len(rows)} comparisons from {len(files)} saved maps")


if __name__ == "__main__":
    main()
