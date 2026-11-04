# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched fresh variogram runs: pointwise, forecast, and saved-member diagnostics."""

import argparse
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import diagnose_stochastic_shared_spatial as spatial  # type: ignore[import-not-found]
import matplotlib
import numpy as np
import pandas as pd
import report_stochastic_shared_noise as shared  # type: ignore[import-not-found]
import torch
from diagnose_stochastic_velocity import (  # type: ignore[import-not-found]
    metrics as velocity_metrics,
)

from samudra.experiments.stochastic_variogram import variogram_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANTS = [
    "B-learned-control",
    "B-learned-variogram",
    "B-shared-deep-wide",
    "B-shared-deep-wide-variogram",
]
LABELS = [
    "Learned-noise control",
    "Learned noise + variogram",
    "Shared deep + wide control",
    "Shared deep + wide + variogram",
]
PAIRS = {
    "B-learned-variogram": "B-learned-control",
    "B-shared-deep-wide-variogram": "B-shared-deep-wide",
}
CHANNELS = ["uo_0", "uo_9", "uo_18", "vo_0", "vo_9", "vo_18"]
LAGS = (1, 2, 4, 8)
LICENSE = (
    "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
)


def paired_rows(frame, keys, metrics):
    """Pair each intervention only with its own architecture's control."""
    rows = []
    for variant, control in PAIRS.items():
        left = frame[frame.variant == variant]
        right = frame[frame.variant == control]
        merged = left.merge(
            right, on=keys, suffixes=("", "_control"), validate="one_to_one"
        )
        if len(merged) != len(left) or len(merged) != len(right) or not len(merged):
            raise ValueError(f"Incomplete matched comparison: {variant}, {control}")
        for _, row in merged.iterrows():
            for metric in metrics:
                value, reference = row[metric], row[metric + "_control"]
                rows.append(
                    dict(
                        variant=variant,
                        control_variant=control,
                        **{key: row[key] for key in keys},
                        metric=metric,
                        value=value,
                        control=reference,
                        difference=value - reference,
                        relative_change_percent=100 * (value / reference - 1)
                        if reference > 0
                        else np.nan,
                    )
                )
    return pd.DataFrame(rows)


def initializer_pairs(frame):
    return paired_rows(frame, ["step", "weights", "month", "variable"], shared.METRICS)


@contextmanager
def configured_reports():
    changes = [
        (shared, "VARIANTS", VARIANTS),
        (shared, "LABELS", LABELS),
        (shared, "paired_differences", initializer_pairs),
        (spatial, "ARMS", VARIANTS),
        (spatial, "LABELS", LABELS),
    ]
    originals = [(module, name, getattr(module, name)) for module, name, _ in changes]
    try:
        for module, name, value in changes:
            setattr(module, name, value)
        yield
    finally:
        for module, name, value in originals:
            setattr(module, name, value)


def normalization(path, grid, input_proof):
    """Verify the actual training normalization file rather than infer scales."""
    digest = shared.fingerprint(path)
    if digest != input_proof["grid_sha256"]:
        raise ValueError(
            "Normalization grid fingerprint differs from initializer input"
        )
    with np.load(path) as source:
        for key in ("lat", "lon", "names", "mask"):
            np.testing.assert_array_equal(source[key], grid[key], err_msg=key)
        mean, std = source["mean"].copy(), source["std"].copy()
    if mean.shape != grid["names"].shape or std.shape != mean.shape:
        raise ValueError("Expected one normalization mean/std per physical channel")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or not (std > 0).all():
        raise ValueError("Invalid fixed normalization statistics")
    return mean, std, digest


def score_case(members, gold, mask, lat, mean, std, names):
    """Exact training score on the six exported velocity channels; CPU only."""
    if members.shape[0] != 8 or members.shape[1:] != gold.shape:
        raise ValueError("Expected eight saved members and a matching reference")
    if len(names) != 6 or set(names) != set(CHANNELS):
        raise ValueError("Expected u/v at saved levels 0, 9, and 18")
    valid = mask & np.isfinite(gold)
    if not np.isfinite(members[:, valid]).all():
        raise ValueError("Nonfinite prediction on scored wet support")
    x = torch.from_numpy(
        (members - mean[None, :, None, None]) / std[None, :, None, None]
    )
    y = torch.from_numpy((gold - mean[:, None, None]) / std[:, None, None])
    wet = torch.from_numpy(mask)
    area = torch.from_numpy(np.cos(np.deg2rad(lat))[:, None])
    rows = []
    selections = [(name, [i]) for i, name in enumerate(names)] + [
        ("all-six", list(range(len(names))))
    ]
    with torch.inference_mode():
        for channel, ids in selections:
            terms = variogram_score(x[:, None, ids], y[None, ids], wet[ids], area, LAGS)
            for index, lag in enumerate(LAGS):
                rows.append(
                    dict(
                        channel=channel,
                        lag=lag,
                        **{
                            key: float(terms[key + "_by_lag"][index])
                            for key in ("fair", "plugin", "correction")
                        },
                    )
                )
            rows.append(
                dict(
                    channel=channel,
                    lag=0,
                    **{
                        key: float(terms[key])
                        for key in ("fair", "plugin", "correction")
                    },
                )
            )
    return rows


def score_saved_variograms(production, normalization_grid):
    rows: list[dict[str, Any]] = []
    sources = []
    for variant in VARIANTS:
        for weights in ("raw", "ema"):
            folder = production / variant / f"initializer-{weights}"
            with np.load(folder / "grid.npz") as source:
                grid = {key: source[key].copy() for key in source.files}
            selection = grid["selection"]
            names = grid["names"][selection].tolist()
            for month in shared.MONTHS:
                proof = json.loads((folder / f"input-{month}.json").read_text())
                mean, std, digest = normalization(normalization_grid, grid, proof)
                path = folder / f"{variant}-02000-{month}-maps.npz"
                with np.load(path) as maps:
                    members, gold = maps["members"].copy(), maps["gold"].copy()
                scores = score_case(
                    members,
                    gold,
                    grid["mask"][selection].astype(bool),
                    grid["lat"],
                    mean[selection],
                    std[selection],
                    names,
                )
                rows.extend(
                    dict(
                        variant=variant, weights=weights, month=month, step=2000, **row
                    )
                    for row in scores
                )
                sources.append(dict(path=str(path), sha256=shared.fingerprint(path)))
    sources.append(dict(path=str(normalization_grid), sha256=digest))
    return pd.DataFrame(rows), sources


def draw_variograms(frame, output):
    for weights in ("raw", "ema"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, channel in zip(axes.ravel(), CHANNELS, strict=True):
            for variant, label, marker in zip(
                VARIANTS, LABELS, ["o", "s", "^", "D"], strict=True
            ):
                subset = frame[
                    (frame.variant == variant)
                    & (frame.weights == weights)
                    & (frame.channel == channel)
                    & (frame.lag > 0)
                ].sort_values("lag")
                ax.plot(subset.lag, subset.fair, marker=marker, label=label)
            ax.set(
                title=channel,
                xlabel="Separation (grid cells)",
                ylabel="Fair variogram score (normalized)",
                xticks=LAGS,
            )
            ax.grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"Final 2k initializer: {weights.upper()}; three dates, eight members; lower is better"
        )
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"variogram-by-lag-{weights}.{suffix}", dpi=150)
        plt.close(fig)


def training_rows(production, *, expected_updates=2000):
    """Separate the changed auxiliary from components comparable to controls."""
    rows, sources = [], []
    keys = [
        "fair",
        "plugin",
        "finite_ensemble_correction",
        "weighted_auxiliary",
        "weighted_base_initializer_crps",
    ]
    for variant in VARIANTS:
        path = production / variant / "events.jsonl"
        events = [json.loads(line) for line in path.read_text().splitlines()]
        if [row["step"] for row in events] != list(range(1, expected_updates + 1)):
            raise ValueError(f"Incomplete or duplicated training updates: {variant}")
        for event in events:
            step = event["step"]
            expected_source = "observation" if step % 2 else "om4"
            if event["source"] != expected_source or event["counts"] != dict(
                om4=step // 2,
                observation=(step + 1) // 2,
            ):
                raise ValueError(f"Training exposure differs: {variant}, {step}")
            components = np.asarray(event["components"], dtype=float)
            if (
                components.shape != (3,)
                or not np.isfinite(components).all()
                or not np.isfinite(
                    [event["loss"], event["grad_norm"], event["lr"]]
                ).all()
            ):
                raise ValueError(
                    f"Nonfinite or malformed training event: {variant}, {step}"
                )
            terms = {key: 0.0 for key in keys}
            if variant in PAIRS:
                terms = {key: event[f"velocity_init_variogram/{key}"] for key in keys}
                if not np.isfinite(list(terms.values())).all():
                    raise ValueError(
                        f"Nonfinite variogram diagnostic: {variant}, {step}"
                    )
                if expected_source == "observation" and any(terms.values()):
                    raise ValueError(
                        "Observation update contains variogram contribution"
                    )
            native = expected_source == "om4"
            base = (
                terms["weighted_base_initializer_crps"]
                if variant in PAIRS
                else 0.1 * components[2]
            )
            if native:
                np.testing.assert_allclose(
                    0.1 * components[2],
                    base + terms["weighted_auxiliary"],
                    rtol=1e-5,
                    atol=1e-7,
                    err_msg="Initializer CRPS/auxiliary decomposition",
                )
                np.testing.assert_allclose(
                    event["loss"],
                    0.5 * components[0] + 0.5 * components[1] + 0.1 * components[2],
                    rtol=1e-5,
                    atol=1e-7,
                    err_msg="Native training objective decomposition",
                )
                if variant in PAIRS:
                    np.testing.assert_allclose(
                        terms["weighted_auxiliary"],
                        0.1 * terms["fair"],
                        rtol=1e-5,
                        atol=1e-7,
                        err_msg="Expected variogram weight one inside reconstruction",
                    )
                    np.testing.assert_allclose(
                        terms["fair"],
                        terms["plugin"] - terms["finite_ensemble_correction"],
                        rtol=1e-5,
                        atol=1e-7,
                    )
            rows.append(
                dict(
                    variant=variant,
                    step=step,
                    source=expected_source,
                    source_update=event["counts"][expected_source],
                    om4_updates=event["counts"]["om4"],
                    observation_updates=event["counts"]["observation"],
                    lr=event["lr"],
                    grad_norm=event["grad_norm"],
                    total_loss=event["loss"],
                    om4_weighted_initializer_crps=base if native else np.nan,
                    om4_weighted_monthly_loss=0.5 * components[0] if native else np.nan,
                    om4_weighted_native_forecast=0.5 * components[1]
                    if native
                    else np.nan,
                    observation_loss=event["loss"] if not native else np.nan,
                    om4_weighted_variogram=terms["weighted_auxiliary"]
                    if native
                    else np.nan,
                    om4_variogram_fair=terms["fair"] if native else np.nan,
                    om4_variogram_plugin=terms["plugin"] if native else np.nan,
                    om4_variogram_correction=terms["finite_ensemble_correction"]
                    if native
                    else np.nan,
                )
            )
        sources.append(dict(path=str(path), sha256=shared.fingerprint(path)))
    return pd.DataFrame(rows), sources


def draw_training(frame, output):
    panels = [
        (
            "om4",
            "om4_weighted_initializer_crps",
            "OM4 initializer: base fair CRPS × 0.1",
            False,
        ),
        (
            "om4",
            "om4_weighted_native_forecast",
            "OM4 native forecast loss × 0.5",
            False,
        ),
        ("observation", "observation_loss", "Observation total loss", False),
        ("om4", "om4_weighted_variogram", "Added variogram contribution × 0.1", True),
        ("om4", "om4_variogram_fair", "Variogram fair score, unweighted", True),
        (
            "om4",
            "om4_variogram_correction",
            "Finite-ensemble correction, unweighted",
            True,
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    for ax, (source, metric, title, treatments_only) in zip(
        axes.ravel(), panels, strict=True
    ):
        for index, (variant, label, marker) in enumerate(
            zip(VARIANTS, LABELS, ["o", "s", "^", "D"], strict=True)
        ):
            if treatments_only and variant not in PAIRS:
                continue
            group = frame[
                (frame.variant == variant) & (frame.source == source)
            ].sort_values("step")
            smoothed = group[metric].rolling(50, min_periods=50).mean()
            ax.plot(
                group.step,
                smoothed,
                label=label,
                color=f"C{index}",
                marker=marker,
                markevery=100,
                markersize=4,
            )
        ax.set(
            title=title,
            xlabel="Total mixed optimizer updates",
            ylabel="Normalized training loss",
            xlim=(0, 2000),
        )
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2)
    fig.suptitle(
        "Training components: trailing mean of 50 updates from the same source (100 mixed updates)",
        y=1.10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(
            output / f"training-components.{suffix}", dpi=150, bbox_inches="tight"
        )
    plt.close(fig)


def depth_diagnostics(channels, channel_summary, normalization_grid, output):
    """Expose channel-wise sign errors hidden by depth-averaged scores."""
    with np.load(normalization_grid) as grid:
        depths = grid["depth"].copy()
    if depths.shape != (19,) or not np.all(np.diff(depths) > 0):
        raise ValueError("Expected the 19 increasing physical depth coordinates")
    selected = channel_summary[
        channel_summary.step.eq(2000)
        & channel_summary.channel.str.startswith(("uo_", "vo_"))
    ].copy()
    cases = channels[
        channels.step.eq(2000) & channels.channel.str.startswith(("uo_", "vo_"))
    ].copy()
    for frame in (selected, cases):
        frame["variable"] = frame.channel.str.split("_").str[0]
        frame["level"] = frame.channel.str.split("_").str[1].astype(int)
        frame["depth_m"] = depths[frame.level]
        # Diagnostic only: field-wide physical sign reversal from saved moments.
        # It is selected after seeing the reference and is not a forecast score.
        flipped = 2 * (frame.mean_power + frame.gold_power) - frame.mean_mse
        if (flipped < -1e-7).any():
            raise ValueError("Negative sign-reversed diagnostic MSE beyond roundoff")
        frame["counterfactual_negated_mean_rmse"] = np.sqrt(flipped.clip(lower=0))
    keys = ["variant", "weights", "channel"]
    for key, group in cases.groupby(keys):
        if sorted(group.month.tolist()) != shared.MONTHS:
            raise ValueError(f"Incomplete depth diagnostic dates: {key}")
    consistency = cases.groupby(keys, as_index=False).agg(
        negative_correlation_dates=("correlation", lambda x: int((x < 0).sum())),
        correlation_min=("correlation", "min"),
        correlation_max=("correlation", "max"),
    )
    selected = selected.merge(consistency, on=keys, validate="one_to_one")
    columns = [
        "variant",
        "weights",
        "channel",
        "variable",
        "level",
        "depth_m",
        "mean_rmse",
        "member_rmse",
        "fair_crps",
        "correlation",
        "mean_amplitude_slope",
        "bias",
        "counterfactual_negated_mean_rmse",
    ]
    selected[
        columns + ["negative_correlation_dates", "correlation_min", "correlation_max"]
    ].to_csv(
        output / "initializer-depth-diagnostics.csv",
        index=False,
    )
    cases[columns + ["month"]].to_csv(
        output / "initializer-depth-by-date.csv", index=False
    )
    for weights in ("raw", "ema"):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
        for row, variable in enumerate(("uo", "vo")):
            for col, (metric, title, units) in enumerate(
                [
                    ("mean_rmse", "Ensemble-mean RMSE", "m/s"),
                    ("correlation", "Spatial correlation with OM4", "Correlation"),
                    (
                        "mean_amplitude_slope",
                        "Spatial amplitude slope",
                        "Cov(prediction, gold) / Var(gold)",
                    ),
                ]
            ):
                ax = axes[row, col]
                for index, (variant, label, marker) in enumerate(
                    zip(VARIANTS, LABELS, ["o", "s", "^", "D"], strict=True)
                ):
                    group = selected[
                        (selected.variant == variant)
                        & (selected.weights == weights)
                        & (selected.variable == variable)
                    ].sort_values("level")
                    ax.plot(
                        group.depth_m,
                        group[metric],
                        label=label,
                        marker=marker,
                        markersize=4,
                        color=f"C{index}",
                    )
                if metric != "mean_rmse":
                    ax.axhline(0, color="black", linewidth=0.8)
                    ax.set_ylim(-1.05, 1.05)
                if variable == "vo":
                    for depth in (40, 375, 775):
                        ax.axvline(depth, color="grey", linestyle=":", linewidth=0.7)
                ax.set(
                    title=f"{variable}: {title}",
                    xlabel="Depth (m; log scale)",
                    ylabel=units,
                    xscale="log",
                    xticks=[2.5, 10, 40, 100, 375, 775, 2000, 6000],
                    xticklabels=[
                        "2.5",
                        "10",
                        "40",
                        "100",
                        "375",
                        "775",
                        "2000",
                        "6000",
                    ],
                )
                ax.tick_params(axis="x", labelrotation=45)
                ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2
        )
        fig.suptitle(
            f"Final 2k initializer, {weights.upper()}: all 19 depths; three dates, eight members\n"
            "Negative correlation/slope indicates reversed spatial structure; vertical guides mark 40, 375, 775 m",
            y=1.12,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                output / f"initializer-depth-diagnostics-{weights}.{suffix}",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)
    for path in output.glob("initializer-depth-*"):
        if path.is_file() and not path.name.endswith(".license"):
            Path(str(path) + ".license").write_text(LICENSE)


def render_depth_audit(audit, production, output):
    """Verify an additional channel export against the original six cases."""
    output.mkdir(parents=True, exist_ok=True)
    variants = ["B-shared-deep-wide", "B-shared-deep-wide-variogram"]
    expected = [
        f"{variant}-02000-{month}" for variant in variants for month in shared.MONTHS
    ]
    complete_path = audit / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    if complete["complete"] is not True or sorted(complete["results"]) != sorted(
        expected
    ):
        raise ValueError("Unexpected depth-audit completion cohort")
    sources = [dict(path=str(complete_path), sha256=shared.fingerprint(complete_path))]
    with np.load(audit / "grid.npz") as saved:
        grid = {key: saved[key].copy() for key in saved.files}
    names = grid["names"][grid["selection"]].tolist()
    if names != ["vo_3", "vo_8", "vo_10"]:
        raise ValueError("Unexpected depth-audit channel selection")
    sources.append(
        dict(
            path=str(audit / "grid.npz"), sha256=shared.fingerprint(audit / "grid.npz")
        )
    )
    mask = grid["mask"][grid["selection"]].astype(bool)
    limits, checks = [], []
    for month in shared.MONTHS:
        input_path = audit / f"input-{month}.json"
        proof = json.loads(input_path.read_text())
        if proof["weights"] != "ema" or proof["members"] != 8:
            raise ValueError("Unexpected depth-audit weights or member count")
        sources.append(
            dict(path=str(input_path), sha256=shared.fingerprint(input_path))
        )
        cases: list[tuple[np.ndarray, np.ndarray]] = []
        for variant in variants:
            folder = production / variant / "initializer-ema"
            with np.load(folder / "grid.npz") as original_grid:
                shared.matching_arrays(
                    grid, original_grid, ["lat", "lon", "names", "mask"]
                )
            if json.loads((folder / f"input-{month}.json").read_text()) != proof:
                raise ValueError(
                    "Depth-audit input proof differs from original evaluation"
                )
            stem = f"{variant}-02000-{month}"
            path = audit / f"{stem}.json"
            record = json.loads(path.read_text())
            original_path = folder / path.name
            original = json.loads(original_path.read_text())
            for key, value in dict(
                variant=variant, step=2000, month=month, weights="ema"
            ).items():
                if record[key] != value:
                    raise ValueError(f"Depth-audit {key} mismatch")
            if record["checkpoint_sha256"] != original["checkpoint_sha256"]:
                raise ValueError(
                    "Depth-audit checkpoint differs from original evaluation"
                )
            if record["metrics"].keys() != original["metrics"].keys():
                raise ValueError("Depth-audit metric keys differ")
            maximum_difference = 0.0
            for key, values in record["metrics"].items():
                difference = np.max(
                    np.abs(np.asarray(values) - np.asarray(original["metrics"][key]))
                )
                maximum_difference = max(maximum_difference, float(difference))
                np.testing.assert_allclose(
                    values,
                    original["metrics"][key],
                    atol=1e-6,
                    rtol=0,
                    err_msg=f"{stem}: {key}",
                )
            maps_path = audit / f"{stem}-maps.npz"
            with np.load(maps_path) as saved:
                members, gold = saved["members"].copy(), saved["gold"].copy()
            if (
                members.shape != (8, 3, len(grid["lat"]), len(grid["lon"]))
                or gold.shape != members.shape[1:]
            ):
                raise ValueError("Unexpected depth-audit map shape")
            if (
                not np.isfinite(members[:, mask]).all()
                or not np.isfinite(gold[mask]).all()
            ):
                raise ValueError("Nonfinite depth-audit fields on wet support")
            if cases:
                np.testing.assert_array_equal(gold, cases[0][1])
            # Recompute every exported metric from the actual saved channels.
            with torch.inference_mode():
                measured = velocity_metrics(
                    torch.from_numpy(members),
                    torch.from_numpy(gold),
                    torch.from_numpy(mask),
                    torch.from_numpy(grid["lat"]),
                )
            for key, values in measured.items():
                np.testing.assert_allclose(
                    values,
                    np.asarray(record["metrics"][key])[grid["selection"]],
                    atol=1e-6,
                    rtol=0,
                    err_msg=f"Saved maps: {stem}: {key}",
                )
            cases.append((members, gold))
            checks.append(
                dict(
                    variant=variant,
                    month=month,
                    checkpoint_sha256=record["checkpoint_sha256"],
                    original_metrics_maximum_absolute_difference=maximum_difference,
                )
            )
            for source in (path, maps_path, original_path):
                sources.append(
                    dict(path=str(source), sha256=shared.fingerprint(source))
                )
        fig, axes = plt.subplots(
            3, 5, figsize=(24, 9), constrained_layout=True, sharex=True, sharey=True
        )
        for row, (name, depth) in enumerate(zip(names, [40, 375, 775], strict=True)):
            control, gold = cases[0][0][:, row], cases[0][1][row]
            new = cases[1][0][:, row]
            fields = [gold, control.mean(0), new.mean(0), new[0], new.mean(0) - gold]
            limit = max(
                float(
                    np.quantile(
                        np.abs(np.concatenate([field[mask[row]] for field in fields])),
                        0.99,
                    )
                ),
                1e-6,
            )
            limits.append(
                dict(
                    month=month,
                    channel=name,
                    depth_m=depth,
                    color_limit_m_per_s=limit,
                    definition="Symmetric 99th percentile of absolute wet values pooled across the five row panels",
                )
            )
            for col, field in enumerate(fields):
                ax = axes[row, col]
                image = ax.pcolormesh(
                    grid["lon"],
                    grid["lat"],
                    np.where(mask[row], field, np.nan),
                    shading="nearest",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                    rasterized=True,
                )
                ax.set(ylim=(-90, 90), xlim=(0, 360), aspect="equal")
                if row == 0:
                    ax.set_title(
                        [
                            "OM4 reference",
                            "Control mean of 8",
                            "+ Variogram mean of 8",
                            "+ Variogram member 1",
                            "+ Variogram mean − OM4",
                        ][col]
                    )
                if col == 0:
                    ax.set_ylabel(f"v at {depth} m\nLatitude (°N)")
                if row == 2:
                    ax.set_xlabel("Longitude (°E)")
            fig.colorbar(
                image,
                ax=axes[row].tolist(),
                location="right",
                shrink=0.75,
                label=f"m/s; all row panels share ±{limit:.3f}",
            )
        fig.suptitle(
            f"{month} initialization: northward velocity sign reversals at selected depths\n"
            f"Shared deep + wide, EMA at 2,000 updates; reference state {proof['history_midpoints'][-1]}\n"
            "Native grid, full latitude; each row clips at its joint wet-value 99th percentile",
            fontsize=14,
        )
        for suffix in ("png", "pdf"):
            fig.savefig(
                output / f"maps-ema-{month}-vo-sign-reversal.{suffix}",
                dpi=150,
                bbox_inches="tight",
            )
        plt.close(fig)
    pd.DataFrame(limits).to_csv(output / "color-limits.csv", index=False)
    receipt = dict(
        complete=True,
        producer_script_sha256=complete["script_sha256"],
        producer_manifest_sha256=complete["manifest_sha256"],
        report_script_sha256=shared.fingerprint(Path(__file__)),
        sources=sources,
        cases=checks,
        checks="Six final EMA cases: original input proofs and checkpoint fingerprints match; all 77-channel metrics agree within absolute 1e-6; selected-channel metrics recomputed from saved maps agree within absolute 1e-6; controls and intervention have identical gold fields and grids.",
        interpretation="These three depths were selected after the all-depth metric audit found negative correlation; this is a diagnostic selection, not an unbiased sample of model output.",
    )
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    for path in output.iterdir():
        if path.is_file() and not path.name.endswith(".license"):
            Path(str(path) + ".license").write_text(LICENSE)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production", type=Path, required=True)
    parser.add_argument("--normalization-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--depth-audit",
        type=Path,
        help="Additional matched EMA exports for v at 40, 375, and 775 m",
    )
    parser.add_argument(
        "--skip-spatial",
        action="store_true",
        help="Skip existing spectra/derivative diagnostics",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with configured_reports():
        shared.main(
            [
                "--production",
                str(args.production),
                "--output",
                str(args.output),
                "--maps-output",
                str(args.output / "maps"),
            ]
        )
        if not args.skip_spatial:
            spatial.main(
                [
                    "--production",
                    str(args.production),
                    "--output",
                    str(args.output / "spatial"),
                ]
            )
        cases, sources = score_saved_variograms(
            args.production, args.normalization_grid
        )
    summary = cases.groupby(
        ["variant", "weights", "step", "channel", "lag"], as_index=False
    )[["fair", "plugin", "correction"]].mean()
    differences = paired_rows(
        cases,
        ["weights", "step", "month", "channel", "lag"],
        ["fair", "plugin", "correction"],
    )
    cases.to_csv(args.output / "variogram-by-date.csv", index=False)
    summary.to_csv(args.output / "variogram-summary.csv", index=False)
    differences.to_csv(args.output / "variogram-paired-differences.csv", index=False)
    forecast = pd.read_csv(args.output / "forecast-summary.csv")
    paired_rows(forecast, ["weights", "step"], ["blend"]).to_csv(
        args.output / "forecast-paired-differences.csv", index=False
    )
    draw_variograms(summary, args.output)
    training, training_sources = training_rows(args.production)
    training.to_csv(args.output / "training-components-by-update.csv", index=False)
    draw_training(training, args.output)
    depth_diagnostics(
        pd.read_csv(args.output / "initializer-channel-by-date.csv"),
        pd.read_csv(args.output / "initializer-channel-summary.csv"),
        args.normalization_grid,
        args.output,
    )
    audit = (
        render_depth_audit(
            args.depth_audit, args.production, args.output / "depth-audit"
        )
        if args.depth_audit
        else None
    )
    complete_path = args.output / "COMPLETE.json"
    complete = json.loads(complete_path.read_text())
    complete.update(
        wrapper_script_sha256=shared.fingerprint(Path(__file__)),
        variogram_loss_script_sha256=shared.fingerprint(
            Path(inspect.getfile(variogram_score))
        ),
        matched_controls=PAIRS,
        variogram_sources=sources,
        training_sources=training_sources,
        training_diagnostics="All 2000 events per arm have exact alternating exposure counts and finite loss/gradient. Compare base initializer CRPS and native forecast contributions separately from the added variogram. Curves use trailing 50 same-source updates; CSV contains unsmoothed values. No raw total OM4 objective comparison is used to claim improvement across changed losses.",
        depth_diagnostics="All 19 u/v depths are retained separately. Spatial correlation and slope use cosine-weighted wet covariance after spatial demeaning; slope divides by gold spatial variance. Summary correlations/slopes average three per-date statistics; RMSE is sqrt(mean of three spatial MSEs). Counterfactual_negated_mean_rmse derives MSE(-prediction,gold)=2*(mean_power+gold_power)-mean_mse from saved moments. This is a reference-selected sign diagnostic, not a corrected forecast or selection score.",
        depth_audit=audit,
        variogram_scope="Final initializer only: u/v at levels 0, 9, 18, three dates, eight members; six saved channels, not all 19 depths.",
        variogram_definition="Exact training fair p=1 score with wet straight paths, periodic longitude, bounded latitude, endpoint area weights, lags 1/2/4/8. Lag 0 means equal average of supported lags. Normalize using fingerprint-verified original training grid. Negative fair estimates are retained. Lower is better; relative changes only for positive reference scores.",
    )
    complete_path.write_text(json.dumps(complete, indent=2) + "\n")
    for path in args.output.rglob("*"):
        if path.is_file() and not path.name.endswith(".license"):
            Path(str(path) + ".license").write_text(LICENSE)


if __name__ == "__main__":
    main()
