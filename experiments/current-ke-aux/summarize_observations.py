# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare saved observation scores using paired calendar-year resampling."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = pd.read_csv(args.input).fillna({"depth": ""})
    primary = data[data.period_kind == "primary_complete_years"]
    annual = data[data.period_kind == "annual"]
    rows = []
    for (metric, depth), group in primary.groupby(["metric", "depth"]):
        assert len(group) == 6
        for column in (
            "period_start",
            "period_end",
            "n_years",
            "n_time_samples",
            "n_paired_cells",
            "grid_shape",
            "units",
        ):
            assert group[column].nunique() == 1, (metric, depth, column)
        years = annual[(annual.metric == metric) & (annual.depth == depth)]
        table = years.pivot(index="year", columns="model", values="value")
        assert not table.isna().any().any()
        assert len(table) == group.n_years.iloc[0]
        samples = np.random.default_rng(1729).integers(
            0, len(table), size=(10000, len(table))
        )
        scores = dict(zip(group.model, group.value.to_numpy(dtype=float), strict=True))
        for kind in ("ema", "raw"):
            control = table[f"control-{kind}"].to_numpy() ** 2
            control_value = float(scores[f"control-{kind}"])
            np.testing.assert_allclose(np.sqrt(control.mean()), control_value)
            for arm in ("aligned", "seasonal"):
                model = f"{arm}-{kind}"
                treatment = table[model].to_numpy() ** 2
                value = float(scores[model])
                np.testing.assert_allclose(np.sqrt(treatment.mean()), value)
                draws = 100 * (
                    np.sqrt(treatment[samples].mean(1) / control[samples].mean(1)) - 1
                )
                low, high = np.quantile(draws, [0.025, 0.975])
                rows.append(
                    dict(
                        metric=metric,
                        depth=depth,
                        checkpoint=kind,
                        arm=arm,
                        control=control_value,
                        value=value,
                        percent_change=100 * (value / control_value - 1),
                        paired_ci_low=low,
                        paired_ci_high=high,
                        units=group.units.iloc[0],
                        first_year=int(table.index.min()),
                        last_year=int(table.index.max()),
                        n_years=len(table),
                        bootstrap_samples=len(samples),
                        bootstrap_seed=1729,
                    )
                )
    result = pd.DataFrame(rows)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
