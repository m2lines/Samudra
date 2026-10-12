# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Audit and summarize the four primary annual evaluations without model selection."""

import argparse
import gzip
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLES = {
    "A": ("#666666", "^"),
    "B": ("#b44e2f", "s"),
    "C": ("#0072b2", "o"),
    "D": ("#009e73", "D"),
}
LICENSE = (
    "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
)


def save_figure(fig, path):
    # Keep native-grid maps at integer cell-to-pixel scaling; do not use bbox='tight'.
    fig.savefig(path, dpi=100, pil_kwargs={"quality": 82})
    plt.close(fig)
    path.with_suffix(path.suffix + ".license").write_text(LICENSE)


def save_csv(frame, path):
    frame.to_csv(path, index=False, float_format="%.10g")
    path.with_suffix(path.suffix + ".license").write_text(LICENSE)


def bootstrap_pairs(data):
    """Paired origin-year cluster bootstrap; not training-seed uncertainty.

    Preserve the point estimate's equal-origin weights by resampling blocks of
    origins and taking the ratio of all resampled sums, not averaging year ratios.
    Adjacent years can still be dependent because annual windows overlap.
    """
    result = []
    rng = np.random.default_rng(610001)
    d = data.copy()
    d["year"] = d.origin.str[:4].astype(int)
    for lead, group in d.groupby("lead_days"):
        for metric in (
            "rmse_normalized",
            "crps_empirical_normalized",
            "crps_fair_normalized",
        ):
            pivot = group.pivot(
                index=["origin", "year", "variable"], columns="arm", values=metric
            )
            by_year = pivot.groupby("year").sum()
            counts = pivot.groupby("year").size().to_numpy()
            draws = rng.integers(len(counts), size=(4000, len(counts)))
            for arm, reference in (
                ("B", "A"),
                ("C", "A"),
                ("C", "B"),
                ("D", "A"),
                ("D", "B"),
                ("D", "C"),
            ):
                x, y = by_year[arm].to_numpy(), by_year[reference].to_numpy()
                relative = 100 * (x[draws].sum(1) / y[draws].sum(1) - 1)
                low, high = np.quantile(relative, [0.025, 0.975])
                result.append(
                    {
                        "lead_days": lead,
                        "metric": metric,
                        "arm": arm,
                        "reference": reference,
                        "relative_percent": 100 * (x.sum() / y.sum() - 1),
                        "year_ci_low": low,
                        "year_ci_high": high,
                    }
                )
    return pd.DataFrame(result)


def audit(roots):
    frames, manifests = [], {}
    for arm, root in roots.items():
        manifest = json.loads((root / "manifest.json").read_text())
        complete = json.loads((root / "COMPLETE.json").read_text())
        frame = pd.concat(
            [pd.read_csv(p) for p in sorted(root.glob("metrics-rank*.csv"))],
            ignore_index=True,
        )
        assert manifest["arm"] == arm and manifest["checkpoint_epoch"] == 70
        assert complete == {
            "rows": 16170,
            "origins": 30,
            "members": 1 if arm in "AB" else 16,
        }
        keys = list(zip(frame.origin, frame.lead_days, frame.variable, strict=True))
        assert len(keys) == len(set(keys)) == 16170
        assert set(keys) == set(
            itertools.product(
                manifest["origin_times"], manifest["leads_days"], manifest["variables"]
            )
        )
        assert np.isfinite(frame.select_dtypes("number")).all().all()
        assert frame.arm.eq(arm).all() and frame.members.eq(manifest["members"]).all()
        actual = (pd.to_datetime(frame.target) - pd.to_datetime(frame.origin)).dt.days
        assert (actual - frame.lead_days).isin([0, 1]).all()
        scale = frame.variable.map(
            dict(zip(manifest["variables"], manifest["normalization_std"], strict=True))
        )
        for name in (
            "rmse",
            "bias",
            "mae",
            "crps_empirical",
            "crps_fair",
            "spread",
            "member_rmse",
        ):
            np.testing.assert_allclose(
                frame[f"{name}_physical"],
                frame[f"{name}_normalized"] * scale,
                rtol=2e-6,
                atol=1e-8,
            )
        for name in ("coverage_member_range", "coverage_q05_q95"):
            assert frame[name].between(0, 1).all()
        assert (
            frame.crps_fair_normalized <= frame.crps_empirical_normalized + 1e-6
        ).all()
        if arm in "AB":
            assert frame.spread_normalized.eq(0).all()
            np.testing.assert_array_equal(
                frame.crps_empirical_normalized, frame.mae_normalized
            )
        else:
            assert frame.spread_normalized.gt(0).all()
        hist = np.load(root / "rank-histograms.npz")
        assert hist["counts"].shape == (7, 77, manifest["members"] + 1)
        np.testing.assert_allclose(hist["counts"].sum(-1), 30, rtol=1e-6)
        manifests[arm] = manifest
        if arm != "A":
            for name in (
                "source_commit",
                "container",
                "data_root",
                "variables",
                "forcing",
                "normalization_std",
                "origin_indices",
                "origin_times",
                "leads_days",
            ):
                assert manifest[name] == manifests["A"][name]
        for lead in (5, 40, 360):
            fields = np.load(root / f"fields-{lead:03d}d.npz")
            truth = np.load(roots["A"] / f"fields-{lead:03d}d.npz")
            for name in ("variables", "target", "mask"):
                np.testing.assert_array_equal(fields[name], truth[name])
            for name in ("target", "mean", "member0", "member1", "spread"):
                assert np.isfinite(fields[name]).all()
                assert not fields[name][~fields["mask"].astype(bool)].any()
            if arm in "CD":
                assert not np.array_equal(fields["member0"], fields["member1"])
        frames.append(frame)
    return pd.concat(frames, ignore_index=True), manifests


def main():
    parser = argparse.ArgumentParser()
    for arm in "ABCD":
        parser.add_argument(f"--{arm}", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {arm: getattr(args, arm) for arm in "ABCD"}
    data, manifests = audit(roots)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = out / "annual-raw"
    raw_dir.mkdir(exist_ok=True)
    for (arm, lead), frame in data.groupby(["arm", "lead_days"]):
        # Losslessly preserve full numeric rows, split to respect the repository size limit.
        ordered = frame.sort_values(["origin", "variable"])
        for part, chunk in enumerate(np.array_split(np.arange(len(ordered)), 3)):
            path = raw_dir / f"{arm}-{lead:03d}d-{part}.csv.gz"
            path.write_bytes(
                gzip.compress(ordered.iloc[chunk].to_csv(index=False).encode(), mtime=0)
            )
            assert path.stat().st_size < 250 * 1024
            path.with_suffix(path.suffix + ".license").write_text(LICENSE)
    data["family"] = data.variable.str.split("_").str[0]
    metrics = [
        k
        for k in data
        if k.endswith(("_normalized", "_physical")) or k.startswith("coverage")
    ]
    summary = data.groupby(["arm", "lead_days"])[metrics].mean().reset_index()
    summary = summary.drop(columns=[k for k in summary if k.endswith("_physical")])
    # Finite-M-adjusted aggregate spread/error uses RMS, rather than mean of ratios.
    rms = data.groupby(["arm", "lead_days"])[
        ["spread_normalized", "rmse_normalized"]
    ].agg(lambda x: np.sqrt(np.mean(np.square(x))))
    summary["spread_error_adjusted"] = [
        np.sqrt(17 / 16)
        * rms.loc[(r.arm, r.lead_days), "spread_normalized"]
        / rms.loc[(r.arm, r.lead_days), "rmse_normalized"]
        if r.arm in "CD"
        else 0
        for r in summary.itertuples()
    ]
    save_csv(summary, out / "annual-summary.csv")
    save_csv(
        data.groupby(["arm", "lead_days", "family"])[metrics].mean().reset_index(),
        out / "annual-family.csv",
    )
    channel_summary = (
        data.groupby(["arm", "lead_days", "variable"])[metrics].mean().reset_index()
    )
    for arm, frame in channel_summary.groupby("arm"):
        save_csv(frame, out / f"annual-channel-{arm}.csv")
    paired = bootstrap_pairs(data)
    save_csv(paired, out / "annual-paired-year-intervals.csv")
    # Manifest fingerprints remain at the immutable cluster paths; retain the common evaluation contract here.
    contract = {
        k: manifests["A"][k]
        for k in (
            "origin_indices",
            "origin_times",
            "variables",
            "normalization_std",
            "leads_days",
            "forcing",
            "data_root",
        )
    }
    contract["arms"] = {
        arm: {
            k: m[k]
            for k in (
                "slurm_job",
                "wandb_url",
                "checkpoint",
                "checkpoint_epoch",
                "members",
                "source_commit",
            )
        }
        for arm, m in manifests.items()
    }
    path = out / "annual-contract.json"
    path.write_text(json.dumps(contract, indent=2) + "\n")
    path.with_suffix(".json.license").write_text(LICENSE)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for arm, (color, marker) in STYLES.items():
        d = summary[summary.arm == arm]
        for ax, metric, title in zip(
            axes,
            ["rmse_normalized", "crps_empirical_normalized", "spread_error_adjusted"],
            [
                "Mean channel RMSE (normalized)",
                "Mean channel empirical CRPS (normalized)",
                "Finite-member-adjusted RMS spread/error",
            ],
            strict=True,
        ):
            if metric == "spread_error_adjusted" and arm in "AB":
                continue
            ax.plot(d.lead_days, d[metric], color=color, marker=marker, label=arm)
            ax.set(title=title, xlabel="Nominal lead (days)", xscale="log")
            ax.set_xticks(
                [5, 10, 20, 40, 90, 180, 360], labels=[5, 10, 20, 40, 90, 180, 360]
            )
            ax.grid(alpha=0.2)
            ax.legend()
    axes[2].axhline(1, color="black", lw=1, alpha=0.5)
    save_figure(fig, out / "annual-scores.jpg")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    rank_rows: list[dict] = []
    for arm in "CD":
        hist = np.load(roots[arm] / "rank-histograms.npz")
        for ax, lead in zip(axes, [5, 40, 360], strict=True):
            values = hist["counts"][list(hist["leads"]).index(lead)].sum(0)
            values = values / values.sum()
            color, marker = STYLES[arm]
            ax.plot(np.arange(17), values, color=color, marker=marker, label=arm)
            ax.set(
                title=f"{lead} nominal days",
                xlabel="Target rank among 16 members",
                ylabel="Weighted frequency",
            )
            ax.axhline(1 / 17, color="black", lw=1, alpha=0.5)
            ax.legend()
            rank_rows.extend(
                {"arm": arm, "lead_days": lead, "rank": i, "frequency": value}
                for i, value in enumerate(values)
            )
    save_csv(pd.DataFrame(rank_rows), out / "annual-ranks.csv")
    save_figure(fig, out / "annual-ranks.jpg")
    training = json.loads(args.data_manifest.read_text())
    for variable, lead in [(v, 360) for v in ("zos", "thetao_0", "so_9", "uo_9")] + [
        ("so_9", 40)
    ]:
        samples = {
            arm: np.load(root / f"fields-{lead:03d}d.npz")
            for arm, root in roots.items()
        }
        channel = list(samples["A"]["variables"]).index(variable)
        full_channel = training["variables"].index(variable)
        std, mean = (
            training["normalization_std"][full_channel],
            training["normalization_mean"][full_channel],
        )
        wet = samples["A"]["mask"][channel].astype(bool)
        panels = [
            ("Truth", "A", "target"),
            ("A: dynamic MSE", "A", "mean"),
            ("B: plain MSE", "B", "mean"),
            ("C: ensemble mean", "C", "mean"),
            ("D: ensemble mean", "D", "mean"),
            ("C: member 0", "C", "member0"),
            ("D: member 0", "D", "member0"),
            ("D: spread", "D", "spread"),
        ]
        values = [
            np.where(
                wet,
                samples[arm][key][channel] * std + (0 if key == "spread" else mean),
                np.nan,
            )
            for _, arm, key in panels
        ]
        # Shared full union range: do not hide model extremes by clipping to truth quantiles.
        lower, upper = np.nanmin(values[:-1]), np.nanmax(values[:-1])
        fig = plt.figure(figsize=(15.4, 5), dpi=100)
        for i, ((title, _, key), value) in enumerate(zip(panels, values, strict=True)):
            row, col = divmod(i, 4)
            # Exactly 360 x 180 raster pixels per 360 x 180 data grid.
            ax = fig.add_axes(
                (
                    (20 + col * 380) / 1540,
                    (290 - row * 240) / 500,
                    360 / 1540,
                    180 / 500,
                )
            )
            lo, hi = (0, np.nanmax(value)) if key == "spread" else (lower, upper)
            im = ax.imshow(
                value,
                origin="lower",
                interpolation="nearest",
                aspect="auto",
                vmin=lo,
                vmax=hi,
                cmap="viridis",
            )
            ax.set_title(f"{title} | {variable} | {lead} nominal days", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            cax = fig.add_axes(
                ((25 + col * 380) / 1540, (272 - row * 240) / 500, 350 / 1540, 4 / 500)
            )
            fig.colorbar(im, cax=cax, orientation="horizontal").ax.tick_params(
                labelsize=7
            )
        save_figure(fig, out / f"annual-{variable}-{lead:03d}d.jpg")
        if lead == 360:
            error_panels = [
                ("A mean", "A", "mean"),
                ("B mean", "B", "mean"),
                ("C mean", "C", "mean"),
                ("D mean", "D", "mean"),
                ("C member 0", "C", "member0"),
                ("D member 0", "D", "member0"),
            ]
            errors = [
                np.where(
                    wet,
                    (samples[arm][key][channel] - samples["A"]["target"][channel])
                    * std,
                    np.nan,
                )
                for _, arm, key in error_panels
            ]
            limit = np.nanquantile(np.abs(errors), 0.99)
            fig = plt.figure(figsize=(11.6, 5), dpi=100)
            for i, ((title, _, _), value) in enumerate(
                zip(error_panels, errors, strict=True)
            ):
                row, col = divmod(i, 3)
                ax = fig.add_axes(
                    (
                        (20 + col * 380) / 1160,
                        (290 - row * 240) / 500,
                        360 / 1160,
                        180 / 500,
                    )
                )
                im = ax.imshow(
                    value,
                    origin="lower",
                    interpolation="nearest",
                    aspect="auto",
                    vmin=-limit,
                    vmax=limit,
                    cmap="RdBu_r",
                )
                clipped = 100 * np.mean(np.abs(value[wet]) > limit)
                ax.set_title(
                    f"{title} error | {variable} | {clipped:.2f}% beyond scale",
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                cax = fig.add_axes(
                    (
                        (25 + col * 380) / 1160,
                        (272 - row * 240) / 500,
                        350 / 1160,
                        4 / 500,
                    )
                )
                fig.colorbar(
                    im, cax=cax, orientation="horizontal", extend="both"
                ).ax.tick_params(labelsize=7)
            save_figure(fig, out / f"annual-error-{variable}-360d.jpg")
    print(summary[summary.lead_days == 360].to_string(index=False))
    print(
        paired[
            (paired.lead_days == 360) & (paired.metric != "crps_fair_normalized")
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
