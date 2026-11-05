# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare the signed-increment screen at matched update counts and saved weights."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MONTHS = ["2014-01", "2014-04", "2014-05"]
VARIANTS = {
    "B-shared-deep-wide": "Pointwise CRPS",
    "B-shared-deep-wide-variogram": "Variogram ×1",
    "B-shared-deep-wide-signed": "Signed increments ×1",
    "B-shared-deep-wide-weak-variogram": "Variogram ×0.25",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(directories):
    rows, sources = [], []
    common: dict[str, object] = {}
    checkpoints: dict[tuple[str, int], str] = {}
    for variant in VARIANTS:
        root = Path(directories[variant])
        for weights in ("ema", "raw"):
            folder = root / f"initializer-{weights}"
            with np.load(folder / "grid.npz") as grid:
                names = grid["names"].tolist()
            if len(names) != 77:
                raise ValueError("Expected every physical channel")
            for month in MONTHS:
                proof = json.loads((folder / f"input-{month}.json").read_text())
                if proof.pop("weights") != weights or proof["members"] != 8:
                    raise ValueError("Readout does not match")
                if month in common and common[month] != proof:
                    raise ValueError("Forecast inputs/noise/reference cohort differ")
                common[month] = proof
                for step in (500, 1000):
                    path = folder / f"{variant}-{step:05d}-{month}.json"
                    row = json.loads(path.read_text())
                    if (row["step"], row["month"], row["weights"], row["variant"]) != (
                        step,
                        month,
                        weights,
                        variant,
                    ):
                        raise ValueError("Initializer metric identity differs")
                    key = (variant, step)
                    if (
                        key in checkpoints
                        and checkpoints[key] != row["checkpoint_sha256"]
                    ):
                        raise ValueError("Dates/readouts use different checkpoints")
                    checkpoints[key] = row["checkpoint_sha256"]
                    sources.append(dict(path=str(path), sha256=digest(path)))
                    for i, name in enumerate(names):
                        metrics = {k: float(v[i]) for k, v in row["metrics"].items()}
                        rows.append(
                            dict(
                                variant=variant,
                                weights=weights,
                                step=step,
                                month=month,
                                channel=name,
                                variable=name.split("_")[0],
                                **metrics,
                            )
                        )
    frame = pd.DataFrame(rows)
    summary = frame.groupby(
        ["variant", "weights", "step", "channel", "variable"], as_index=False
    ).mean(numeric_only=True)
    summary["mean_rmse"] = np.sqrt(summary.mean_mse)
    summary["member_rmse"] = np.sqrt(summary.member_mse)
    summary["member_power_ratio"] = summary.member_power / summary.gold_power
    consistency = frame.groupby(
        ["variant", "weights", "step", "channel"], as_index=False
    ).agg(
        negative_dates=("correlation", lambda x: int((x < 0).sum())),
        strong_negative_dates=("correlation", lambda x: int((x < -0.5).sum())),
        minimum_correlation=("correlation", "min"),
        maximum_correlation=("correlation", "max"),
    )
    summary = summary.merge(
        consistency, on=["variant", "weights", "step", "channel"], validate="one_to_one"
    )
    return (
        frame,
        summary,
        dict(
            inputs=common,
            sources=sources,
            checkpoints={f"{a}/{s}": v for (a, s), v in checkpoints.items()},
        ),
    )


def plot_depths(summary, depths, output):
    for weights in ("ema", "raw"):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), constrained_layout=True)
        for r, variable in enumerate(("uo", "vo")):
            for variant, label in VARIANTS.items():
                selected = summary[
                    (summary.variant == variant)
                    & (summary.weights == weights)
                    & (summary.step == 1000)
                    & (summary.variable == variable)
                ].copy()
                selected["level"] = selected.channel.str.split("_").str[-1].astype(int)
                selected = selected.sort_values("level")
                for c, metric in enumerate(
                    ("correlation", "mean_rmse", "fair_crps", "member_power_ratio")
                ):
                    axes[r, c].plot(
                        selected[metric], depths, marker="o", markersize=3, label=label
                    )
            for c, title in enumerate(
                (
                    "Spatial correlation",
                    "Mean RMSE (m/s)",
                    "Fair CRPS (m/s)",
                    "Member / reference power",
                )
            ):
                ax = axes[r, c]
                ax.set_yscale("symlog", linthresh=10)
                ax.invert_yaxis()
                ax.set_title(f"{variable}: {title}")
                ax.grid(alpha=0.2)
                ax.set_ylabel("Depth (m)")
                if c == 0:
                    ax.axvline(0, color="black", lw=0.7)
                if c == 3:
                    ax.axvline(1, color="black", lw=0.7)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"1,000 mixed updates · {weights.upper()} · mean of three held-out cases · eight members"
        )
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"depths-{weights}.{suffix}", dpi=150)
        plt.close(fig)


def plot_maps(directories, output):
    for variant in ("B-shared-deep-wide-signed", "B-shared-deep-wide-weak-variogram"):
        root = Path(directories[variant]) / "initializer-ema"
        grid = dict(np.load(root / "grid.npz"))
        selected = grid["names"][grid["selection"]].tolist()
        for month in MONTHS:
            fields = np.load(root / f"{variant}-01000-{month}-maps.npz")
            channels = ["uo_0", "vo_0", "vo_3", "vo_8", "vo_10"]
            fig, axes = plt.subplots(
                len(channels), 4, figsize=(14.4, 9), constrained_layout=True
            )
            for r, name in enumerate(channels):
                i = selected.index(name)
                mask = grid["mask"][list(grid["names"]).index(name)]
                gold, members = fields["gold"][i], fields["members"][:, i]
                mean = members.mean(0)
                limit = np.nanquantile(np.abs(gold[mask]), 0.995)
                arrays = [gold, mean, members[0], mean - gold]
                for c, (array, title) in enumerate(
                    zip(
                        arrays,
                        (
                            "OM4 reference",
                            "Mean of eight",
                            "Member 1",
                            "Mean − reference",
                        ),
                        strict=True,
                    )
                ):
                    image = axes[r, c].imshow(
                        np.where(mask, array, np.nan),
                        origin="lower",
                        interpolation="nearest",
                        cmap="RdBu_r",
                        vmin=-limit,
                        vmax=limit,
                        aspect="equal",
                    )
                    axes[r, c].set_title(f"{name} · {title}", fontsize=9)
                    axes[r, c].set_xticks([])
                    axes[r, c].set_yticks([])
                    fig.colorbar(
                        image, ax=axes[r, c], fraction=0.026, pad=0.01, label="m/s"
                    )
            fig.suptitle(
                f"{VARIANTS[variant]} · EMA · 1,000 mixed updates · {month} · same-time initialization"
            )
            fig.savefig(output / f"{variant}-{month}-maps.png", dpi=150)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directories", type=Path, required=True)
    parser.add_argument("--normalization-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directories = json.loads(args.directories.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    cases, summary, provenance = collect(directories)
    cases.to_csv(args.output / "initializer-cases.csv", index=False)
    summary.to_csv(args.output / "initializer-summary.csv", index=False)
    variables = []
    final = summary[(summary.weights == "ema") & (summary.step == 1000)]
    for (variant, variable), group in final.groupby(["variant", "variable"]):
        variables.append(
            dict(
                variant=variant,
                variable=variable,
                mean_rmse=float(np.sqrt(group.mean_mse.mean())),
                fair_crps=float(group.fair_crps.mean()),
                member_power_ratio=float(
                    group.member_power.sum() / group.gold_power.sum()
                ),
                mean_correlation=float(group.correlation.mean()),
                negative_channels=int((group.correlation < 0).sum()),
            )
        )
    pd.DataFrame(variables).to_csv(args.output / "variable-summary.csv", index=False)
    forecasts = []
    for variant in ("B-shared-deep-wide-signed", "B-shared-deep-wide-weak-variogram"):
        for weights in ("ema", "raw"):
            result = json.loads(
                (
                    Path(directories[variant])
                    / "evaluation"
                    / f"step-01000-{weights}"
                    / "COMPLETE.json"
                ).read_text()
            )
            if (
                result["step"] != 1000
                or result["weights"] != weights
                or len(result["metrics"]["origins"]) != 9
            ):
                raise ValueError("Forecast checkpoint or cohort differs")
            forecasts.append(
                dict(
                    variant=variant,
                    weights=weights,
                    step=result["step"],
                    score=result["score"],
                    **result["ratios"],
                )
            )
    pd.DataFrame(forecasts).to_csv(args.output / "forecast-summary.csv", index=False)
    with np.load(args.normalization_grid) as grid:
        depths = grid["depth"]
    plot_depths(summary, depths, args.output)
    plot_maps(directories, args.output)
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print(
        summary[
            (summary.weights == "ema")
            & (summary.step == 1000)
            & summary.channel.isin(["uo_0", "vo_0", "vo_3", "vo_8", "vo_10"])
        ][
            [
                "variant",
                "channel",
                "mean_rmse",
                "fair_crps",
                "correlation",
                "strong_negative_dates",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
