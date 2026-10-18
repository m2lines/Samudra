# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the complete W&B epoch export without changing metric definitions."""

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    directory = Path(__file__).parent
    with (directory / "epoch-history.csv").open() as source:
        rows = list(csv.DictReader(source))
    epochs = [int(float(row["epoch"])) for row in rows]
    scores = [
        float(row["rollout_val/360d/normalized_rmse/channel_mean"]) for row in rows
    ]
    assert epochs == list(range(1, 71))
    assert all(math.isfinite(value) for value in scores)
    best = min(range(len(scores)), key=scores.__getitem__)
    fig, axis = plt.subplots(figsize=(8.4, 4.2), layout="constrained")
    axis.plot(epochs, scores, color="#17649a", linewidth=2)
    axis.scatter(
        [epochs[best], epochs[-1]], [scores[best], scores[-1]], color="#c04c23"
    )
    axis.annotate(
        f"Best: epoch {epochs[best]}, {scores[best]:.4f}",
        (epochs[best], scores[best]),
        xytext=(-80, 23),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "0.4"},
    )
    axis.annotate(
        f"Epoch 70: {scores[-1]:.4f}",
        (epochs[-1], scores[-1]),
        xytext=(-110, 28),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "0.4"},
    )
    axis.set(
        title="CM4 1° · current Samudra · seed 15",
        xlabel="Training epoch",
        ylabel="Configured 360-day validation rollout\nmean normalized prognostic-channel RMSE",
        xlim=(0, 73),
    )
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    fig.savefig(directory / "learning-curve.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
