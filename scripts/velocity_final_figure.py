# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Plot held-out skill and seed variation from the completed campaign report."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main(root):
    metrics = pd.read_csv(root / "metrics-by-seed.csv")
    metrics = metrics[
        (metrics.split == "test") & (metrics.region == "global_outside_5deg")
    ]
    comparisons = json.loads((root / "summary.json").read_text())["comparisons"]["test"]
    comparison = pd.DataFrame(
        [row for row in comparisons if row["region"] == "global_outside_5deg"]
    ).sort_values("nominal_lead_days")
    fig, (skill, gain) = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for variant, method, label, color in (
        ("D0", "persistence", "Persistence", "#777777"),
        ("D0", "samudra", "D0: DUACS only", "#2864b4"),
        ("D4", "samudra", "D4: multitask, no explicit geometry", "#d06b16"),
    ):
        selected = metrics[(metrics.variant == variant) & (metrics.method == method)]
        pooled = selected.groupby("lead_step")[
            ["weighted_squared_error", "weight"]
        ].sum()
        skill.plot(
            pooled.index * 5,
            np.sqrt(pooled.weighted_squared_error / pooled.weight),
            "o-",
            label=label,
            color=color,
        )
    skill.set(ylabel="Global vector RMSE (m/s)", title="Held-out DUACS forecasts")
    skill.legend(fontsize=8, loc="upper left")
    for seed in (15, 16, 17):
        pair = metrics[(metrics.seed == seed) & (metrics.method == "samudra")].pivot(
            index="lead_step", columns="variant", values="vector_rmse_m_per_s"
        )
        gain.plot(
            pair.index * 5,
            100 * (1 - pair.D4 / pair.D0),
            "o--",
            alpha=0.65,
            label=f"Seed {seed}",
        )
    value = comparison.improvement_percent
    gain.errorbar(
        comparison.nominal_lead_days,
        value,
        yerr=np.stack(
            (
                value - comparison.paired_block_ci_low,
                comparison.paired_block_ci_high - value,
            )
        ),
        fmt="ko-",
        capsize=4,
        label="Pooled + paired 95% interval",
    )
    gain.axhline(0, color="#777777", linewidth=0.8)
    gain.set(
        ylabel="D4 RMSE improvement over D0 (%)",
        title="Seed variation limits the conclusion",
    )
    gain.legend(fontsize=8, loc="upper left")
    for axis in (skill, gain):
        axis.set(xlabel="Forecast lead (days)", xticks=[5, 10, 20, 30])
        axis.grid(alpha=0.2)
    fig.savefig(root / "held-out-skill.png", dpi=140)
    fig.savefig(root / "held-out-skill.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    main(parser.parse_args().report_dir)
