"""Verify and summarize matched single-example representation seed repeats."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--repeats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms = ["absolute_random", "same_slot_zero", "latest_zero"]
    rows, hashes = {}, {}
    for seed in (15, 16, 17):
        root = args.first if seed == 15 else args.repeats / f"seed{seed}"
        rows[str(seed)] = {}
        reference = None
        for arm in arms:
            path = root / arm
            assert (path / "COMPLETE").exists()
            protocol = json.loads((path / "protocol.json").read_text())
            assert protocol["seed"] == seed and protocol["steps"] == 5000
            digest = protocol["backbone_initial_sha256"]
            assert hashes.setdefault(seed, digest) == digest
            history = [json.loads(x) for x in (path / "history.jsonl").read_text().splitlines()]
            endpoint = history[-1]
            assert endpoint["step"] == 5000
            with np.load(path / "maps-5000.npz") as z:
                truth, mask = z["truth"].copy(), z["mask"].copy()
                if reference is None:
                    reference = truth, mask
                assert np.array_equal(reference[0], truth) and np.array_equal(reference[1], mask)
                error = z["prediction"].astype(np.float64) - truth
                mse = ((error**2 * mask).sum((-2, -1)) / mask.sum((-2, -1))).mean()
                assert np.isclose(mse, endpoint["eval_mse"], rtol=2e-6)
            rows[str(seed)][arm] = {k: endpoint[k] for k in ("eval_mse", "highpass_mse")}
        for arm in arms:
            for metric in ("eval_mse", "highpass_mse"):
                rows[str(seed)][arm][metric + "_improvement"] = (
                    rows[str(seed)][arms[0]][metric] / rows[str(seed)][arm][metric]
                )
    assert len(set(hashes.values())) == 3
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps({
        "seeds": rows,
        "validation": "Matched within-seed backbone hashes, distinct between seeds; same targets/masks; "
                      "all 5000 updates; raw-map MSE independently reproduced in FP64.",
        "scope": "Three random initializations, one training example; no independent validation data.",
    }, indent=2) + "\n")
    (args.output / "initial-backbones.sha256").write_text(
        "".join(f"{digest}  seed{seed}/initial-backbone\n" for seed, digest in hashes.items()))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, title in zip(axes, ("eval_mse", "highpass_mse"), ("State MSE", "High-pass error MSE")):
        for arm in arms[1:]:
            ax.plot([15, 16, 17], [rows[str(seed)][arm][metric + "_improvement"]
                                  for seed in (15, 16, 17)], marker="o", label=arm)
        ax.set_xticks([15, 16, 17]); ax.set_xlabel("Initialization seed")
        ax.set_ylabel("Error reduction versus absolute output (×)")
        ax.set_title(title); ax.grid(alpha=.2); ax.legend()
    fig.suptitle("Residual parameterization benefit repeats across initializations\n"
                 "Same fixed example, MSE loss, backbone initialization and 5,000 updates within each seed")
    fig.savefig(args.output / "improvement.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
