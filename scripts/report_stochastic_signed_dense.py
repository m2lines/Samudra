# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate and report the fresh signed-increment, direct-ERA5 learning curve.

Default mode requires all five milestones, final conditioning diagnostics, and
terminal GPU accounting. --preview renders only verified available evidence.
This program reads artifacts and renders figures; it never submits compute.
"""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = "b4f8089226e75d0a8261c5c07a92890aa612ac7e952be896f6bfb66bd51da01c"
VARIANT = "B-signed-direct8-dense"
MONTHS = ["2014-01", "2014-04", "2014-05"]
STEPS = [500, 1000, 2000, 4000, 8000]
NAMES = [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + ["zos"]
COMPONENTS = [
    "sst_rmse",
    "velocity_rmse",
    "eke_rmse",
    "ohc_0_700_rmse",
    "ohc_700_2000_rmse",
]
LABELS = ["SST", "SSH-derived velocity", "EKE", "OHC 0–700 m", "OHC 700–2000 m"]
MODES = ["recipient", "input-swap", "interior-state-swap", "full-state-swap"]
MODE_LABELS = [
    "Recipient inputs",
    "Surface-history swap",
    "Interior-state swap",
    "Full-state swap",
]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path, sources=None):
    if sources is not None:
        sources[str(path)] = digest(path)
    return json.loads(Path(path).read_text())


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def initializer_summary(frame):
    """Equal case/depth weighting; take roots and power ratios after averaging."""
    result = frame.groupby(
        ["weights", "step", "channel", "variable"], as_index=False
    ).mean(numeric_only=True)
    result["rmse"] = np.sqrt(result.mean_mse)
    result["member_rmse"] = np.sqrt(result.member_mse)
    result["member_power_ratio"] = result.member_power / result.gold_power
    return result


def family_summary(frame):
    result = frame.groupby(["weights", "step", "variable"], as_index=False).mean(
        numeric_only=True
    )
    result["rmse"] = np.sqrt(result.mean_mse)
    result["member_power_ratio"] = result.member_power / result.gold_power
    return result


def collect_evaluations(directories, *, preview=False):
    rows: list[dict[str, object]] = []
    forecasts: list[dict[str, object]] = []
    sources: dict[str, str] = {}
    inputs: dict[str, dict[str, object]] = {}
    checkpoints: dict[int, str] = {}
    bundles: list[tuple[Path, list[int]]] = []
    seen_steps: set[int] = set()
    forecast_origins, climate_metrics, initializer_script = None, None, None
    for directory in directories:
        directory = Path(directory)
        if not directory.exists():
            if preview:
                continue
            raise ValueError(f"Missing evaluation: {directory}")
        top = read(directory / "COMPLETE.json", sources)
        steps = top["initializer_steps"]
        if (
            top.get("complete") is not True
            or top["source"] != SOURCE
            or top["members"] != 8
            or set(top["weights"]) != {"raw", "ema"}
            or not steps
            or len(set(steps)) != len(steps)
            or any(s not in STEPS for s in steps)
            or seen_steps.intersection(steps)
        ):
            raise ValueError("Evaluation source, readout or milestone identity differs")
        seen_steps.update(steps)
        manifest = read(directory / "manifest.json", sources)
        if [(e["variant"], e["step"]) for e in manifest] != [
            (VARIANT, s) for s in steps
        ]:
            raise ValueError("Evaluation manifest differs from declared milestones")
        for step in steps:
            checkpoints[step] = top["checkpoints"][str(step)]
        bundles.append((directory, steps))
        for weights in ("raw", "ema"):
            folder = directory / f"initializer-{weights}"
            complete = read(folder / "COMPLETE.json", sources)
            expected = [f"{VARIANT}-{s:05d}-{m}" for s in steps for m in MONTHS]
            if (
                complete.get("complete") is not True
                or complete["results"] != expected
                or complete["manifest_sha256"] != digest(directory / "manifest.json")
            ):
                raise ValueError("Initializer completion or manifest identity differs")
            if (
                initializer_script is not None
                and complete["script_sha256"] != initializer_script
            ):
                raise ValueError("Initializer scoring implementation changed")
            initializer_script = complete["script_sha256"]
            for month in MONTHS:
                proof = read(folder / f"input-{month}.json", sources)
                if (
                    proof.pop("weights") != weights
                    or proof["members"] != 8
                    or proof["month"] != month
                    or proof["seed"] != 904003
                ):
                    raise ValueError("Initializer input readout differs")
                if month in inputs and proof != inputs[month]:
                    raise ValueError(
                        "Initializer dates, forcing inputs, noise or reference support changed"
                    )
                inputs[month] = proof
                for step in steps:
                    value = read(folder / f"{VARIANT}-{step:05d}-{month}.json", sources)
                    if (
                        value["variant"],
                        value["weights"],
                        value["step"],
                        value["month"],
                        value["checkpoint_sha256"],
                    ) != (VARIANT, weights, step, month, checkpoints[step]):
                        raise ValueError(
                            "Initializer result checkpoint identity differs"
                        )
                    metrics = {
                        k: np.asarray(v, dtype=float)
                        for k, v in value["metrics"].items()
                    }
                    if any(
                        v.shape != (77,) or not np.isfinite(v).all()
                        for v in metrics.values()
                    ):
                        raise ValueError(
                            "Initializer requires 77 finite channel metrics"
                        )
                    for key in ("mean_mse", "member_mse", "gold_power", "member_power"):
                        if (metrics[key] < 0).any():
                            raise ValueError("Negative squared metric")
                    if (np.abs(metrics["correlation"]) > 1.00001).any():
                        raise ValueError("Invalid spatial correlation")
                    for i, channel in enumerate(NAMES):
                        rows.append(
                            dict(
                                weights=weights,
                                step=step,
                                month=month,
                                channel=channel,
                                variable=channel.split("_")[0],
                                **{k: float(v[i]) for k, v in metrics.items()},
                            )
                        )
            forecast = read(directory / f"forecast-{weights}/COMPLETE.json", sources)
            step = top["forecast_step"]
            if (
                forecast["step"] != step
                or forecast["weights"] != weights
                or forecast["checkpoint_sha256"] != checkpoints[step]
                or forecast["qualification"]
                or forecast["metrics"]["members"] != 8
                or set(forecast["ratios"]) != set(COMPONENTS)
            ):
                raise ValueError("Forecast checkpoint or protocol differs")
            origins = forecast["metrics"]["origins"]
            if (
                len(origins) != 9
                or len(set(origins)) != 9
                or (forecast_origins is not None and forecast_origins != origins)
            ):
                raise ValueError("Forecast origin cohort changed")
            forecast_origins = origins
            controls = [forecast["climatology"]["metrics"][k] for k in COMPONENTS]
            if climate_metrics is not None:
                np.testing.assert_allclose(controls, climate_metrics, rtol=1e-8, atol=0)
            climate_metrics = controls
            ratios = []
            for key, baseline in zip(COMPONENTS, controls, strict=True):
                actual = forecast["metrics"]["metrics"][key]
                if (
                    not np.isfinite([actual, baseline]).all()
                    or baseline <= 0
                    or actual < 0
                ):
                    raise ValueError("Invalid forecast RMSE or normalization")
                ratio = actual / baseline
                np.testing.assert_allclose(
                    ratio, forecast["ratios"][key], rtol=1e-10, atol=1e-12
                )
                ratios.append(ratio)
                forecasts.append(
                    dict(
                        weights=weights,
                        step=step,
                        component=key,
                        rmse=actual,
                        climatology_rmse=baseline,
                        ratio=ratio,
                    )
                )
            np.testing.assert_allclose(
                np.mean(ratios), forecast["score"], rtol=1e-10, atol=1e-12
            )
    if not rows or (not preview and seen_steps != set(STEPS)):
        raise ValueError("Missing required learning-curve milestones")
    return (
        pd.DataFrame(rows),
        pd.DataFrame(forecasts),
        dict(
            sources=sources,
            inputs=inputs,
            checkpoints={str(k): v for k, v in checkpoints.items()},
            forecast_origins=forecast_origins,
            initializer_script=initializer_script,
        ),
        bundles,
    )


def collect_conditioning(directory, prepared_path, checkpoints, grid_sha):
    directory, prepared_path = Path(directory), Path(prepared_path)
    sources: dict[str, str] = {}
    result = read(directory / "COMPLETE.json", sources)
    release = read(directory / "RELEASE_COMPLETE.json", sources)
    prepared = read(prepared_path, sources)
    if (
        result.get("complete") is not True
        or release.get("complete") is not True
        or release["diagnostic_complete_sha256"] != digest(directory / "COMPLETE.json")
        or result["checkpoint_sha256"] != checkpoints["8000"]
        or release["checkpoint_sha256"] != checkpoints["8000"]
        or result["prepared_sha256"] != digest(prepared_path)
        or result["contract"]["producer"] != SOURCE
        or prepared["grid_sha256"] != grid_sha
        or result["members"] != 8
        or result["weights"] != "ema"
        or result["seed"] != 904003
        or result["modes"] != MODES
        or prepared["schema"] != "stochastic-conditioning-v2"
        or result["script_sha256"] != prepared["script_sha256"]
    ):
        raise ValueError("Conditioning diagnostic identity differs")
    cases = {c["name"]: c["source"] for c in prepared["cases"]}
    if (
        len(cases) != 12
        or list(cases.values()).count("om4") != 3
        or list(cases.values()).count("observation") != 9
    ):
        raise ValueError("Conditioning cohort differs")
    if {r["case"] for r in result["receipts"]} != set(cases) or len(
        result["receipts"]
    ) != 12:
        raise ValueError("Conditioning cases incomplete")
    frame = pd.DataFrame(result["rows"])
    frame["lead"] = frame.lead.astype(str)
    if set(zip(frame.case, frame.intervention)) != {
        (c, m) for c in cases for m in MODES
    }:
        raise ValueError("Missing conditioning intervention")
    if frame.duplicated(["case", "intervention", "lead", "channel"]).any():
        raise ValueError("Duplicate conditioning metric")
    if any(cases[row.case] != row.source for row in frame.itertuples()):
        raise ValueError("Conditioning source label differs")
    expected_rows: set[tuple[str, str, str, str]] = set()
    for case, source in cases.items():
        keys = [
            (str(lead), channel)
            for lead in (0, 5, 15, 30)
            for channel in (NAMES if source == "om4" else ["thetao_0", "zos"])
        ]
        if source == "observation":
            keys += [
                ("calendar-month", NAMES[i]) for i in [*range(38, 52), *range(57, 71)]
            ]
            keys += [
                ("calendar-month-OHC", channel)
                for channel in ("ohc_0_700", "ohc_700_2000")
            ]
        expected_rows.update(
            (case, mode, lead, channel) for mode in MODES for lead, channel in keys
        )
    if (
        set(zip(frame.case, frame.intervention, frame.lead, frame.channel))
        != expected_rows
    ):
        raise ValueError("Conditioning channel/lead inventory differs")
    validate_conditioning_support(frame)
    perturb = pd.DataFrame(result["initial_perturbations"])
    expected = {
        (c, m, slot, channel)
        for c in cases
        for m in MODES
        for slot in range(2)
        for channel in NAMES
    }
    actual = set(
        zip(perturb.case, perturb.intervention, perturb.history_slot, perturb.channel)
    )
    if len(perturb) != len(expected) or actual != expected:
        raise ValueError("Initial perturbation inventory differs")
    fixed = perturb[
        (perturb.intervention == "recipient")
        | (
            (perturb.intervention == "interior-state-swap")
            & perturb.channel.isin(["thetao_0", "zos"])
        )
    ]
    if not np.allclose(
        fixed[["paired_member_change_rmse", "paired_mean_change_rmse"]],
        0,
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("State swap changed the protected recipient surfaces")
    frame["variable"] = frame.channel.map(
        lambda s: s if s.startswith("ohc_") else s.split("_")[0]
    )
    return frame, perturb, sources


def validate_conditioning_support(frame):
    """Every counterfactual must be scored on the same recipient support."""
    if not np.isfinite(frame.support_weight).all() or (frame.support_weight < 0).any():
        raise ValueError("Invalid conditioning support")
    for _, group in frame.groupby(["case", "lead", "channel"]):
        if set(group.intervention) != set(MODES):
            raise ValueError("Conditioning support omits a mode")
        for key in ("climatology_mse", "climatology_mae", "support_weight"):
            values = group[key].to_numpy(float)
            if not np.allclose(
                values, values[0], rtol=1e-10, atol=1e-12, equal_nan=True
            ):
                raise ValueError(
                    "Reference/climatology support changed across interventions"
                )


def conditioning_summary(frame):
    keys = ["source", "lead", "intervention", "variable"]
    frame = frame.copy()
    changes = ["paired_member_change_rmse", "paired_mean_change_rmse"]
    frame[changes] = frame[changes] ** 2
    summary = frame.groupby(keys, as_index=False).mean(numeric_only=True)
    summary[changes] = np.sqrt(summary[changes])
    summary["rmse"] = np.sqrt(summary.mean_mse)
    summary["climatology_rmse"] = np.sqrt(summary.climatology_mse)
    summary["mse_skill"] = 1 - summary.mean_mse / summary.climatology_mse
    summary["crps_skill"] = 1 - summary.fair_crps / summary.climatology_mae
    return summary


def plot_learning(frame, output):
    from samudra.constants import build_om4_layout

    summary = initializer_summary(frame)
    families = family_summary(frame)
    depths = np.asarray(build_om4_layout().depth_levels)
    for weights in ("ema", "raw"):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
        steps = sorted(frame.step.unique())
        for row, variable in enumerate(("uo", "vo")):
            for index, step in enumerate(steps):
                group = summary[
                    (summary.weights == weights)
                    & (summary.variable == variable)
                    & (summary.step == step)
                ].copy()
                group["depth_index"] = group.channel.str.split("_").str[-1].astype(int)
                group = group.sort_values("depth_index")
                for col, metric in enumerate(
                    ("correlation", "rmse", "fair_crps", "member_power_ratio")
                ):
                    axes[row, col].plot(
                        group[metric],
                        depths,
                        marker=["o", "s", "^", "D", "v"][index],
                        ms=3,
                        label=f"{step:,} mixed updates",
                    )
            for col, title in enumerate(
                (
                    "Spatial correlation",
                    "Mean RMSE (m/s)",
                    "Fair CRPS (m/s)",
                    "Member/reference power",
                )
            ):
                ax = axes[row, col]
                ax.set_title(f"{variable}: {title}")
                ax.set_yscale("symlog", linthresh=10)
                ax.invert_yaxis()
                if col == 0:
                    ax.set_ylabel("Depth (m)")
                ax.grid(alpha=0.2)
                if col in (0, 3):
                    ax.axvline(0 if col == 0 else 1, color="black", lw=0.6)
        axes[0, 0].legend(fontsize=7, handlelength=3)
        fig.suptitle(
            f"Fresh direct-ERA5 model · {weights.upper()} · three OM4 cases · eight members"
        )
        save_figure(fig, output / f"depths-{weights}")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, variable in enumerate(("uo", "vo")):
        for weights, marker in (("raw", "o"), ("ema", "s")):
            g = families[
                (families.variable == variable) & (families.weights == weights)
            ].sort_values("step")
            for col, metric in enumerate(("rmse", "fair_crps", "member_power_ratio")):
                axes[row, col].plot(
                    g.step, g[metric], marker=marker, label=weights.upper()
                )
        for col, title in enumerate(
            ("Mean RMSE (m/s)", "Fair CRPS (m/s)", "Member/reference power")
        ):
            axes[row, col].set(
                title=f"{variable}: {title}", xlabel="Mixed optimizer updates"
            )
            axes[row, col].grid(alpha=0.2)
        axes[row, 2].axhline(1, color="black", lw=0.6)
    axes[0, 0].legend()
    fig.suptitle(
        "Equal weighting across three cases and 19 depths; ratios after aggregation"
    )
    save_figure(fig, output / "velocity-learning")
    return summary, families


def save_figure(figure, path):
    for suffix in ("png", "pdf"):
        figure.savefig(path.with_suffix("." + suffix), dpi=150)
    plt.close(figure)


def plot_forecast(frame, output):
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    groups = sorted(set(zip(frame.step, frame.weights)))
    width = 0.8 / len(groups)
    for index, (step, weights) in enumerate(groups):
        g = (
            frame[(frame.step == step) & (frame.weights == weights)]
            .set_index("component")
            .loc[COMPONENTS]
        )
        ax.bar(
            np.arange(5) - 0.4 + width / 2 + index * width,
            g.ratio,
            width,
            label=f"{step:,} mixed · {weights.upper()}",
        )
    ax.axhline(1, color="black", lw=0.8)
    ax.set(
        xticks=np.arange(5),
        xticklabels=LABELS,
        ylabel="RMSE / existing climatology RMSE",
        title="Nine OBS origins; existing five-component protocol; lower is better",
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output / "forecast-components")


def map_figure(path, grid, output, title, *, climate=None):
    with np.load(path) as data:
        members, gold = (
            data["members"],
            data["gold"] if "gold" in data else data["reference"],
        )
        names = (
            data["names"].tolist()
            if "names" in data
            else grid["names"][grid["selection"]].tolist()
        )
        if climate is True:
            baseline = data["climatology"]
            members, gold = members - baseline, gold - baseline
    if members.shape != (8, *gold.shape) or gold.ndim != 3 or len(names) != len(gold):
        raise ValueError("Map member/reference shapes differ")
    channels = [c for c in ("uo_0", "vo_0", "vo_8") if c in names]
    fig, axes = plt.subplots(
        len(channels),
        6,
        figsize=(18, 1.8 * len(channels)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, name in enumerate(channels):
        index = names.index(name)
        support = np.isfinite(gold[index])
        if grid is not None:
            support &= grid["mask"][grid["names"].tolist().index(name)].astype(bool)
        draws = members[:, index]
        fields = [
            gold[index],
            draws.mean(0),
            draws[0],
            draws[1],
            draws[2],
            draws.mean(0) - gold[index],
        ]
        limit = max(float(np.quantile(np.abs(gold[index][support]), 0.995)), 1e-8)
        error_limit = max(float(np.quantile(np.abs(fields[-1][support]), 0.995)), 1e-8)
        for col, (field, label) in enumerate(
            zip(
                fields,
                (
                    "OM4 reference",
                    "Mean of 8",
                    "Member 1",
                    "Member 2",
                    "Member 3",
                    "Mean error",
                ),
                strict=True,
            )
        ):
            bound = error_limit if col == 5 else limit
            im = axes[row, col].imshow(
                np.where(support, field, np.nan),
                origin="lower",
                interpolation="nearest",
                aspect="equal",
                cmap="RdBu_r",
                vmin=-bound,
                vmax=bound,
            )
            axes[row, col].set_title(f"{name} · {label}", fontsize=9)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            fig.colorbar(im, ax=axes[row, col], fraction=0.026, pad=0.01, label="m/s")
    fig.suptitle(
        title
        + (" · monthly training-climatology anomalies" if climate else "")
        + " · error has its own scale"
    )
    save_figure(fig, output)


def plot_maps(bundles, output, sources, *, preview=False):
    latest = max(s for _, steps in bundles for s in steps)
    directory = next(d for d, steps in bundles if latest in steps)
    folder = directory / "initializer-ema"
    missing = []
    grid_path = folder / "grid.npz"
    if not grid_path.exists():
        if not preview:
            raise ValueError("Final initializer map grid is missing")
        return [str(grid_path)]
    sources[str(grid_path)] = digest(grid_path)
    with np.load(grid_path) as data:
        grid = dict(data)
    if grid["names"].tolist() != NAMES:
        raise ValueError("Map grid channel order differs")
    for month in MONTHS:
        path = folder / f"{VARIANT}-{latest:05d}-{month}-maps.npz"
        if not path.exists():
            missing.append(str(path))
            continue
        sources[str(path)] = digest(path)
        map_figure(
            path,
            grid,
            output / f"members-{latest}-{month}",
            f"Fresh direct-ERA5 · EMA · {latest:,} mixed updates · {month}",
        )
    if missing and not preview:
        raise ValueError("Final member maps are incomplete")
    return missing


def plot_conditioning(summary, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for row, variable in enumerate(("uo", "vo")):
        g = (
            summary[
                (summary.source == "om4")
                & (summary.lead == "0")
                & (summary.variable == variable)
            ]
            .set_index("intervention")
            .loc[MODES]
        )
        for col, metric in enumerate(
            ("mse_skill", "crps_skill", "paired_member_change_rmse")
        ):
            axes[row, col].bar(np.arange(4), g[metric])
            axes[row, col].set(
                xticks=np.arange(4),
                xticklabels=MODE_LABELS,
                title=f"{variable}: {metric.replace('_', ' ')}",
            )
            axes[row, col].tick_params(axis="x", labelrotation=25, labelsize=7)
            axes[row, col].grid(axis="y", alpha=0.2)
    fig.suptitle(
        "OM4 initialization: training-only monthly climatology; paired member noise"
    )
    save_figure(fig, output / "conditioning-initializer")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, variable, unit in zip(axes, ("thetao", "zos"), ("°C", "m"), strict=True):
        g = summary[
            (summary.source == "observation")
            & (summary.variable == variable)
            & summary.lead.isin(["0", "5", "15", "30"])
        ]
        for mode, label, marker in zip(
            MODES[1:], MODE_LABELS[1:], ("o", "s", "^"), strict=True
        ):
            values = g[g.intervention == mode].copy()
            values["days"] = values.lead.astype(int)
            values = values.sort_values("days")
            ax.plot(
                values.days,
                values.paired_member_change_rmse,
                marker=marker,
                label=label,
            )
        ax.set(
            title=variable,
            xlabel="Forecast lead (days)",
            ylabel=f"Paired member change ({unit})",
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle(
        "Forecast sensitivity with recipient future forcing and calendar fixed"
    )
    save_figure(fig, output / "conditioning-forecast")


def training_curve(paths, output, *, preview=False):
    rows: dict[int, dict[str, object]] = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            step = row["step"]
            if step in rows and rows[step] != row:
                raise ValueError("Conflicting training event at duplicate update")
            rows[step] = row
    if not rows:
        return None
    if sorted(rows) != list(range(1, max(rows) + 1)) or (
        not preview and max(rows) != 8000
    ):
        raise ValueError(
            "Training history is not continuous through the declared endpoint"
        )
    frame = pd.DataFrame(rows.values()).sort_values("step")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for ax, source in zip(axes, ("om4", "observation"), strict=True):
        g = frame[frame.source == source]
        ax.plot(
            g.step,
            g.loss.rolling(100, min_periods=10).mean(),
            label="Mean of previous 100 source updates",
        )
        ax.set(
            xlabel="Mixed optimizer updates", ylabel="Training objective", title=source
        )
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7)
    save_figure(fig, output / "training-objectives")
    frame.to_csv(output / "training-events.csv", index=False)
    return dict(last_step=int(frame.step.max()), events=len(frame))


def cost_summary(path):
    value = read(path)
    fixed = float(value.get("fixed_completed_gpu_hours", 0))
    if not np.isfinite(fixed) or fixed < 0:
        raise ValueError("Invalid fixed completed GPU cost")
    jobs = set()
    for row in value["allocations"]:
        if (
            row["job_id"] in jobs
            or not np.isfinite([row["gpus"], row["elapsed_seconds"]]).all()
            or row["gpus"] <= 0
            or int(row["gpus"]) != row["gpus"]
            or row["elapsed_seconds"] < 0
        ):
            raise ValueError("Invalid or duplicate actual GPU allocation")
        jobs.add(row["job_id"])
        if row.get("state") not in {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
        }:
            raise ValueError("GPU cost must come from terminal allocations")
        fixed += row["gpus"] * row["elapsed_seconds"] / 3600
    return dict(
        actual_gpu_hours=fixed,
        allocations=value["allocations"],
        fixed_completed_gpu_hours=value.get("fixed_completed_gpu_hours", 0),
    )


def markdown_table(frame, columns, formats=None):
    formats = formats or {}
    rows = [
        [formats.get(c, lambda x: str(x))(v) for c, v in zip(columns, row, strict=True)]
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    return "\n".join(
        [
            "| " + " | ".join(columns) + " |",
            "| " + " | ".join(["---"] * len(columns)) + " |",
            *["| " + " | ".join(r) + " |" for r in rows],
        ]
    )


def report_text(frame, forecast, conditioning, cost, missing, preview, training=None):
    latest = int(frame.step.max())
    families = family_summary(frame)
    selected = families[
        (families.step == latest) & families.variable.isin(["uo", "vo"])
    ]
    selected = selected[
        [
            "weights",
            "variable",
            "rmse",
            "fair_crps",
            "correlation",
            "member_power_ratio",
        ]
    ]
    text = [
        f"# Signed-increment model with direct ERA5 and dense OM4 sampling{' (preview)' if preview else ''}",
        f"This report covers saved checkpoints through {latest:,} mixed optimizer updates. The run uses a stochastic initializer and a deterministic processor. Both are Samudra ConvNeXt U-Nets. Eight-member evaluation uses raw weights and an exponential moving average (EMA).",
        "The initializer uses widths 96/192/384 with three blocks per stage; the processor uses 64/128/256 with one block per stage. Both retain InstanceNorm. All eight ERA5 channels enter each model directly. Training uses effective batch eight, four ensemble members, AdamW at 1e-4, and no cooldown or replay.",
        "The auxiliary loss is fair CRPS on signed spatial velocity increments at separations 1, 2, 4 and 8 cells, with weight one inside the initializer reconstruction term (outer weight 0.1). Differences are start minus offset endpoint in E/N/NE/NW directions, using fixed OM4 channel standardization, wet traversed paths, endpoint-mean area weights, and equal supported channel/direction/lag weights. It applies to all 38 OM4 initializer velocity channels; the pointwise fair-CRPS objectives remain active. Training contains 2,548 OM4 windows at five-day origin spacing and 243 observational windows (May 1993–July 2013). The previous short screen used 409 monthly OM4 origins and an ERA5 channel bottleneck; comparisons with that screen change both data sampling and forcing representation.",
        "Initializer scores use the common wet-cell mask and cosine-latitude weights, with no extreme-latitude exclusion. Channel metrics are in physical units: velocity m/s, temperature °C, salinity psu, and SSH m. Initializer metrics average three held-out OM4 cases (January, April and May 2014). The displayed CRPS is pointwise fair CRPS of the ensemble, and RMSE scores the ensemble mean. RMSE is the square root of mean squared error after equal case/depth aggregation; member power is divided by reference power after aggregation. Correlations are arithmetic means of the case/channel spatial correlations. Lower RMSE and CRPS are better; power ratio one matches reference second moment, which does not establish realistic members.",
        markdown_table(
            selected,
            list(selected.columns),
            {
                c: lambda x: f"{x:.5f}"
                for c in selected.columns
                if c not in ("weights", "variable")
            },
        ),
        "![Velocity learning](velocity-learning.png)",
        "![All-depth EMA metrics](depths-ema.png)",
        "[All-depth raw-weight metrics](depths-raw.png)",
    ]
    first = int(frame.step.min())
    start = families[families.step == first].set_index(["weights", "variable"])
    finish = families[families.step == latest].set_index(["weights", "variable"])
    changes = pd.DataFrame(
        {
            "RMSE change (%)": 100 * (finish.rmse / start.rmse - 1),
            "CRPS change (%)": 100 * (finish.fair_crps / start.fair_crps - 1),
        }
    ).reset_index()
    text += [
        f"Initializer changes from {first:,} to {latest:,} mixed updates are within this fresh run; negative percentages mean lower error.",
        markdown_table(
            changes,
            list(changes.columns),
            {
                "RMSE change (%)": lambda x: f"{x:+.2f}",
                "CRPS change (%)": lambda x: f"{x:+.2f}",
            },
        ),
    ]
    velocity = frame[(frame.step == latest) & frame.variable.isin(["uo", "vo"])]
    text.append(
        f"Across {len(velocity)} velocity case/channel/readout combinations at the latest checkpoint, "
        f"{int((velocity.correlation < 0).sum())} have negative spatial correlation; "
        f"the minimum is {velocity.correlation.min():.4f}. These are global spatial correlations of ensemble means. Positive correlation does not establish locally correct signs, individual-member alignment, or signed-increment skill. This run alone does not isolate the auxiliary loss effect."
    )
    missing_names = {Path(p).name for p in missing}
    for month in MONTHS:
        if (
            f"{VARIANT}-{latest:05d}-{month}-maps.npz" not in missing_names
            and "grid.npz" not in missing_names
        ):
            text.append(
                f"![OM4 reference, ensemble mean and individual members for {month}](members-{latest}-{month}.png)"
            )
    text.append(
        "The forecast table uses nine observational origins, November 2013–July 2014. Each component is divided by its fixed climatology RMSE; the blend is the unweighted mean of five ratios. Surface metrics average separately computed RMSEs at days 5, 15 and 30; OHC uses the calendar-weighted monthly ensemble-mean trajectory. The baseline uses the 243 prepared training windows: monthly surface climatologies fill never-observed cells with a monthly ocean mean, while monthly interior climatologies fill missing cells with channel training means. Its OHC is integrated from that climatological temperature state rather than averaged from original IAP OHC targets. Scores use common observed wet support and cosine-latitude weights, with geostrophic velocity excluding 5°S–5°N. Table values are dimensionless ratios: one matches this baseline and lower is better. The blend contains no spectral DEX term.",
    )
    table = forecast.pivot(
        index=["step", "weights"], columns="component", values="ratio"
    ).reset_index()
    table["blend"] = table[COMPONENTS].mean(axis=1)
    ratio_columns = {
        key: label + " ratio" for key, label in zip(COMPONENTS, LABELS, strict=True)
    }
    table = table.rename(columns=ratio_columns)
    text += [
        markdown_table(
            table,
            ["step", "weights", *ratio_columns.values(), "blend"],
            {c: lambda x: f"{x:.4f}" for c in [*ratio_columns.values(), "blend"]},
        ),
        "![Forecast components](forecast-components.png)",
    ]
    if conditioning is not None:
        text += [
            "The separate conditioning diagnostics use monthly climatologies computed only from training records. Their observational OHC climatology averages the saved original IAP OHC targets directly. These skill scores therefore have a different baseline from the forecast blend above.",
            "Surface-history swaps replace SST/SSH histories while preserving recipient ERA5 and calendar. Interior-state swaps initialize a donor-year state from its own complete inputs, then replace both recipient interior history slots while preserving recipient SST/SSH slots. Full-state swaps also replace those surface slots. Every forecast retains recipient future forcing/calendar, and all initializations use paired member noise. These interventions test model sensitivity; mixing states can create inputs outside the training distribution.",
            "Skill is 1 − model error / climatology error: positive is better than climatology. MSE skill uses squared errors and CRPS skill uses fair CRPS against climatology MAE. Paired changes are root mean squared differences after equal case/channel aggregation. All modes retain common reference/climatology support.",
            "![Climatology and initialization sensitivity](conditioning-initializer.png)",
            "![Forecast sensitivity](conditioning-forecast.png)",
        ]
        text.extend(
            f"![Reference and member anomalies for {month}](anomalies-{month}.png)"
            for month in MONTHS
        )
    else:
        text += [
            "Final training-only climatology and independent state-swap diagnostics are pending."
        ]
    exposures = pd.DataFrame(
        [
            dict(
                mixed_updates=s,
                updates_per_source=s // 2,
                examples_per_source=4 * s,
                om4_passes=4 * s / 2548,
                observation_passes=4 * s / 243,
            )
            for s in sorted(frame.step.unique())
        ]
    )
    text += [
        markdown_table(
            exposures,
            list(exposures.columns),
            {
                "om4_passes": lambda x: f"{x:.2f}",
                "observation_passes": lambda x: f"{x:.2f}",
            },
        ),
        f"Actual allocated GPU time in the supplied terminal ledger: {cost['actual_gpu_hours']:.3f} GPU-hours."
        if cost
        else "Final allocated GPU-hours are pending terminal scheduler accounting.",
        "This is one training seed with three OM4 initializer cases and nine observational forecast origins. Thirty-day results do not establish annual rollout stability or physically coherent uncertainty. The next architecture comparison may consider the full Samudra-v2 processor and GroupNorm discussed in deterministic thread `01a1a372-5a3e-75c0-aab6-a2a7a535d653`; those settings have not been audited or used in this run.",
    ]
    if training:
        text.append(
            f"Training events are continuous from update 1 through {training['last_step']:,}; the reference allocation's first 22 updates are joined to the resumed production events. This was a fresh run, with optimizer and EMA retained at that engineering handoff."
        )
        text.append(
            "![Training objectives, shown separately by data source](training-objectives.png)"
        )
    text.append(
        "[Input/checkpoint fingerprints](REPORT_INPUTS.json), [all initializer case metrics](initializer-cases.csv), [forecast components](forecast-components.csv)."
    )
    if missing:
        text += [
            "Some preview map assets are not available locally: "
            + ", ".join(Path(x).name for x in missing)
            + "."
        ]
    return "\n\n".join(text) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--conditioning", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--cost-ledger", type=Path)
    parser.add_argument("--training-events", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    if not args.preview and (
        not args.conditioning or not args.prepared or not args.cost_ledger
    ):
        parser.error(
            "Full report requires conditioning, prepared manifest and terminal cost ledger"
        )
    frame, forecasts, provenance, bundles = collect_evaluations(
        args.evaluations, preview=args.preview
    )
    args.output.mkdir(parents=True, exist_ok=False)
    summary, families = plot_learning(frame, args.output)
    plot_forecast(forecasts, args.output)
    missing = plot_maps(
        bundles, args.output, provenance["sources"], preview=args.preview
    )
    frame.to_csv(args.output / "initializer-cases.csv", index=False)
    summary.to_csv(args.output / "initializer-depths.csv", index=False)
    families.to_csv(args.output / "initializer-families.csv", index=False)
    forecasts.to_csv(args.output / "forecast-components.csv", index=False)
    conditioning = None
    if args.conditioning:
        if not args.prepared:
            parser.error("Conditioning results require their prepared manifest")
        conditioning, perturb, sources = collect_conditioning(
            args.conditioning,
            args.prepared,
            provenance["checkpoints"],
            provenance["inputs"][MONTHS[0]]["grid_sha256"],
        )
        provenance["sources"].update(sources)
        conditioning.to_csv(args.output / "conditioning-cases.csv", index=False)
        perturb.to_csv(args.output / "initial-perturbations.csv", index=False)
        aggregates = conditioning_summary(conditioning)
        aggregates.to_csv(args.output / "conditioning-summary.csv", index=False)
        plot_conditioning(aggregates, args.output)
        for month in MONTHS:
            path = args.conditioning / f"om4-{month}-recipient.npz"
            if not path.exists():
                raise ValueError("Final conditioning anomaly map is missing")
            provenance["sources"][str(path)] = digest(path)
            map_figure(
                path,
                None,
                args.output / f"anomalies-{month}",
                f"Final EMA · {month}",
                climate=True,
            )
    training = training_curve(args.training_events, args.output, preview=args.preview)
    if args.training_events:
        provenance["sources"].update({str(p): digest(p) for p in args.training_events})
    cost = cost_summary(args.cost_ledger) if args.cost_ledger else None
    if args.cost_ledger:
        provenance["sources"][str(args.cost_ledger)] = digest(args.cost_ledger)
    (args.output / "REPORT.md").write_text(
        report_text(
            frame, forecasts, conditioning, cost, missing, args.preview, training
        )
    )
    save_json(
        args.output / "REPORT_INPUTS.json",
        dict(
            provenance,
            preview=args.preview,
            missing_map_assets=missing,
            training=training,
            cost=cost,
            report_script_sha256=digest(__file__),
        ),
    )
    print(
        json.dumps(
            dict(
                output=str(args.output),
                milestones=sorted(frame.step.unique().tolist()),
                preview=args.preview,
                missing_map_assets=missing,
            )
        )
    )


if __name__ == "__main__":
    main()
