#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Audit one-field ensembles, score paired origins, and measure spatial structure."""

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from samudra.experiments.surface_adaptation_analysis import read_bytes


def analyze(root, reference, output, origins):
    output.mkdir(parents=True, exist_ok=True)
    expected = json.loads(origins.read_text())
    specialist = np.load(
        io.BytesIO(
            read_bytes(reference / "field-specialization-eval/heldout-bf16-maps.npz")
        )
    )
    rows, spatial, spectra, ranks = [], [], [], []
    for run in ["diffusion", "deterministic"]:
        directory = root / run
        logged = pd.read_csv(directory / "heldout.csv").sort_values("index")
        assert logged.date.tolist() == expected
        for record in logged.to_dict("records"):
            idx = str(record["index"])
            with np.load(
                io.BytesIO(read_bytes(directory / f"heldout-members-{idx}.npz"))
            ) as data:
                mask = data["mask"].astype(bool)
                lat = data["latitude"]
                std = float(data["std"])
                weight = mask * np.cos(np.deg2rad(lat))[:, None]
                weight /= weight.sum()

                def avg(value):
                    return np.sum(value.astype(np.float64) * weight, axis=(-2, -1))

                truth, climate, baseline, samples = [
                    data[k].astype(np.float64)
                    for k in ["truth", "climatology", "baseline", "samples"]
                ]
                for k in ["truth", "climatology"]:
                    np.testing.assert_array_equal(data[k], specialist[f"{idx}_{k}"])
                np.testing.assert_array_equal(mask, specialist["mask"])
                assert str(data["date"]) == record["date"]
                assert np.isfinite(samples).all() and (samples[:, ~mask] == 0).all()
                mean = samples.mean(0)
                count = len(samples)
                sorted_samples = np.sort(samples, axis=0)
                crps = (
                    np.abs(samples - truth).mean(0)
                    - np.sum(
                        sorted_samples
                        * (2 * np.arange(count) - count + 1)[:, None, None],
                        axis=0,
                    )
                    / count**2
                )
                np.testing.assert_allclose(
                    avg(crps) * std, record["crps"], rtol=2e-5, atol=1e-8
                )
                np.testing.assert_allclose(
                    avg((mean - truth) ** 2) * std**2,
                    record["mean_mse"],
                    rtol=2e-5,
                    atol=1e-9,
                )
                row: dict[str, Any] = {str(k): v for k, v in record.items()}
                row["run"] = run
                row["climatology_mae"] = float(avg(np.abs(climate - truth)) * std)
                correction = np.abs(samples - truth).mean(0) - crps
                fair_crps = crps - correction / (count - 1) if count > 1 else crps
                row["fair_crps"] = float(avg(fair_crps) * std)
                row["coverage_minmax"] = float(
                    avg((truth >= samples.min(0)) & (truth <= samples.max(0)))
                )
                row["calibrated_minmax_reference"] = (count - 1) / (count + 1)
                row["calibrated_spread_rmse_reference"] = np.sqrt(
                    (count - 1) / (count + 1)
                )
                if count > 1:
                    less = (samples < truth).sum(0)
                    ties = (samples == truth).sum(0)
                    rng_ties = np.random.default_rng(1729 + int(idx))
                    rank = less + np.floor(
                        rng_ties.random(truth.shape) * (ties + 1)
                    ).astype(int)
                    for bin_index in range(count + 1):
                        ranks.append(
                            dict(
                                run=run,
                                date=record["date"],
                                rank=bin_index,
                                weight=float(avg(rank == bin_index)),
                                calibrated_reference=1 / (count + 1),
                            )
                        )
                row["specialist_mse"] = float(
                    avg((specialist[idx + "_prediction"] - truth) ** 2) * std**2
                )
                row["specialist_mae"] = float(
                    avg(np.abs(specialist[idx + "_prediction"] - truth)) * std
                )
                rows.append(row)
                fields = {
                    "truth": truth - climate,
                    "D": baseline - climate,
                    "specialist": specialist[idx + "_prediction"] - climate,
                    "mean": mean - climate,
                    "sample0": samples[0] - climate,
                }
                for name, field in fields.items():
                    field = (field - avg(field)) * mask
                    for lag in [1, 3, 9, 18]:
                        valid = mask & np.roll(mask, lag, axis=-1)
                        w = valid * np.cos(np.deg2rad(lat))[:, None]
                        shifted = np.roll(field, lag, axis=-1)
                        covariance = (
                            float((field * shifted * w).sum() / w.sum()) * std**2
                        )
                        structure = (
                            float(((field - shifted) ** 2 * w).sum() / w.sum()) * std**2
                        )
                        spatial.append(
                            dict(
                                run=run,
                                date=record["date"],
                                field=name,
                                lag_cells=lag,
                                covariance=covariance,
                                structure_function=structure,
                            )
                        )
                    # Common-mask grid-space zonal spectrum: not an isotropic ocean spectrum.
                    power = abs(np.fft.rfft(field, axis=-1)) ** 2 / field.shape[-1] ** 2
                    latitude_weight = np.cos(np.deg2rad(lat))
                    latitude_weight /= latitude_weight.sum()
                    power = (power * latitude_weight[:, None]).sum(0) * std**2
                    for wave, value in enumerate(power):
                        spectra.append(
                            dict(
                                run=run,
                                date=record["date"],
                                field=name,
                                zonal_wavenumber=wave,
                                power=value,
                            )
                        )
    table = pd.DataFrame(rows)
    table.to_csv(output / "per-origin.csv", index=False)
    pd.DataFrame(ranks).to_csv(output / "rank-histogram.csv", index=False)
    pd.DataFrame(spatial).to_csv(output / "spatial-covariance.csv", index=False)
    pd.DataFrame(spectra).groupby(
        ["run", "field", "zonal_wavenumber"]
    ).power.mean().reset_index().to_csv(output / "zonal-spectrum.csv", index=False)
    summaries = []
    for group_name, g in table.groupby("run"):
        summaries.append(
            dict(
                run=str(group_name),
                origins=len(g),
                mean_rmse=np.sqrt(g.mean_mse.mean()),
                sample_rmse=np.sqrt(g.sample_mse.mean()),
                d_rmse=np.sqrt(g.d_mse.mean()),
                climatology_rmse=np.sqrt(g.climatology_mse.mean()),
                climatology_mae=g.climatology_mae.mean(),
                specialist_rmse=np.sqrt(g.specialist_mse.mean()),
                crps=g.crps.mean(),
                fair_crps=g.fair_crps.mean(),
                coverage_minmax=g.coverage_minmax.mean(),
                calibrated_minmax_reference=g.calibrated_minmax_reference.mean(),
                calibrated_spread_rmse_reference=g.calibrated_spread_rmse_reference.mean(),
                d_mae=g.d_mae.mean(),
                specialist_mae=g.specialist_mae.mean(),
                coverage90=g.coverage90.mean(),
                spread_rmse_ratio=np.sqrt(g.spread_variance.mean() / g.mean_mse.mean()),
                anomaly_correlation=g.anomaly_correlation.mean(),
                mean_amplitude=np.sqrt(
                    g.mean_anomaly_m2.mean() / g.truth_anomaly_m2.mean()
                ),
                sample_amplitude=np.sqrt(
                    g.sample_anomaly_m2.mean() / g.truth_anomaly_m2.mean()
                ),
            )
        )
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)
    d = table[table.run == "diffusion"].sort_values("date").reset_index(drop=True)
    control = (
        table[table.run == "deterministic"].sort_values("date").reset_index(drop=True)
    )
    assert d.date.tolist() == control.date.tolist()
    years = pd.to_datetime(d.date).dt.year.to_numpy()
    unique = np.unique(years)
    rng = np.random.default_rng(1729)
    boot = [
        np.concatenate(
            [
                np.flatnonzero(years == y)
                for y in rng.choice(unique, len(unique), replace=True)
            ]
        )
        for _ in range(3000)
    ]
    comparisons = []
    for name, baseline in [("D", d), ("specialist", d), ("deterministic", control)]:
        for metric, bkey, ckey, root_metric in [
            (
                "RMSE",
                {
                    "D": "d_mse",
                    "specialist": "specialist_mse",
                    "deterministic": "mean_mse",
                }[name],
                "mean_mse",
                True,
            ),
            (
                "CRPS",
                {"D": "d_mae", "specialist": "specialist_mae", "deterministic": "crps"}[
                    name
                ],
                "crps",
                False,
            ),
        ]:
            b = baseline[bkey].to_numpy()
            c = d[ckey].to_numpy()

            def improvement(ids):
                ratio = c[ids].mean() / b[ids].mean()
                return 1 - (np.sqrt(ratio) if root_metric else ratio)

            low, high = np.quantile([improvement(ids) for ids in boot], [0.025, 0.975])
            comparisons.append(
                dict(
                    baseline=name,
                    metric=metric,
                    relative_improvement=improvement(np.arange(len(d))),
                    year_block_low=low,
                    year_block_high=high,
                )
            )
    pd.DataFrame(comparisons).to_csv(output / "paired.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(
            {
                "origins_per_arm": 99,
                "common_truth_climatology_mask": True,
                "metrics_reconciled": True,
                "spectra": "Common-mask grid-space zonal spectra; land edges affect power. Not an isotropic physical ocean spectrum.",
            },
            indent=2,
        )
        + "\n"
    )
    print(pd.DataFrame(summaries).to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "docs/experiments/surface-initializer-diagnostics-results/artifacts/raw"
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--origins",
        type=Path,
        default=Path("docs/experiments/surface-wave3-origins.json"),
    )
    args = p.parse_args()
    analyze(args.raw, args.reference, args.output, args.origins)


if __name__ == "__main__":
    main()
