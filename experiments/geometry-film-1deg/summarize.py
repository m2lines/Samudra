# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Summarize checksum-verified outputs downloaded under .LOCAL/geometry-results."""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(".LOCAL/geometry-results")
OUT = Path("experiments/geometry-film-1deg/results")
OUT.mkdir(exist_ok=True)
frames = []
runtime_rows = []
configs = []
for arm in ["control", "concat", "film"]:
    run = ROOT / f"output/geometry-{arm}-seed15"
    config = yaml.load((run / "config.yaml").read_text(), Loader=yaml.BaseLoader)
    config.pop("experiment")
    config.pop("post_train_eval")
    config["model"]["unet"].pop("geometry_mode")
    configs.append(config)
    wandb = json.loads((ROOT / arm / "wandb-summary.json").read_text())
    compute = json.loads((ROOT / arm / "compute.json").read_text())
    log = next((ROOT / "logs").glob(f"{arm}-*.out")).read_text()
    epoch_durations = [
        float(x)
        for x in re.findall(
            r"Training Epoch: \[\d+\] Total time: .*?\(([\d.]+) s / it\)", log
        )
    ]
    assert len(epoch_durations) == 70
    waits = [
        float(x)
        for x in re.findall(r"Training Epoch:.*data_wait_time: [\d.]+\(([\d.]+)\)", log)
    ]
    runtime_rows.append(
        dict(
            arm=arm,
            parameters=compute["parameters"],
            compute_seconds=compute["median_seconds"],
            compute_peak_bytes=compute["peak_allocated_bytes"],
            median_training_epoch_seconds=np.median(epoch_durations) * 176,
            mean_logged_data_wait_seconds=np.mean(waits),
            validation_360d_rmse=wandb["rollout_val/360d/normalized_rmse/channel_mean"],
        )
    )
    for ckpt in ["epoch_0070", "ema_latest"]:
        path = run / "evals" / ckpt
        status = json.loads((path / "stability.json").read_text())
        assert status["forecast_times"] == 598 and status["nonfinite_wet_cells"] == 0
        df = pd.read_csv(path / "latitude-errors.csv")
        assert len(df) == 319930 and df.nonfinite_wet_cells.sum() == 0
        assert (
            np.isfinite(df[["rmse", "bias", "prediction_rms", "prediction_absmax"]])
            .all()
            .all()
        )
        assert df.groupby(["channel", "latitude_band"]).size().eq(598).all()
        for window, selection in [
            ("full", df),
            ("first360d", df[df.lead_days <= 360]),
            ("last365d", df[df.lead_days > df.lead_days.max() - 365]),
        ]:
            selection = selection.assign(
                mse=selection.rmse**2, prediction_ms=selection.prediction_rms**2
            )
            table = (
                selection.groupby(["channel", "latitude_band"])
                .agg(
                    mse=("mse", "mean"),
                    bias=("bias", "mean"),
                    prediction_ms=("prediction_ms", "mean"),
                    prediction_absmax=("prediction_absmax", "max"),
                    times=("time", "count"),
                )
                .reset_index()
            )
            table["rmse"] = np.sqrt(table.pop("mse"))
            table["prediction_rms"] = np.sqrt(table.pop("prediction_ms"))
            table["arm"] = arm
            table["checkpoint"] = ckpt
            table["window"] = window
            frames.append(table)
assert configs[0] == configs[1] == configs[2], (
    "Resolved training configs differ beyond arm metadata and geometry"
)
summary = pd.concat(frames, ignore_index=True)
base = summary[summary.arm == "control"][
    ["checkpoint", "window", "channel", "latitude_band", "rmse"]
].rename(columns={"rmse": "control_rmse"})
summary = summary.merge(
    base,
    on=["checkpoint", "window", "channel", "latitude_band"],
    validate="many_to_one",
)
summary["rmse_change_percent"] = 100 * (summary.rmse / summary.control_rmse - 1)
for window in ["full", "first360d", "last365d"]:
    summary[summary.window == window].to_csv(
        OUT / f"latitude-{window}.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
runtime = pd.DataFrame(runtime_rows)
for metric in ["compute_seconds", "compute_peak_bytes", "validation_360d_rmse"]:
    runtime[metric + "_change_percent"] = 100 * (
        runtime[metric] / runtime.iloc[0][metric] - 1
    )
runtime.to_csv(OUT / "runtime.csv", index=False)
surface = summary[
    (summary.window == "full")
    & (summary.latitude_band == "global")
    & (summary.channel.isin(["zos", "thetao_0", "so_0", "uo_0", "vo_0"]))
]
surface.to_csv(OUT / "surface-global.csv", index=False)
(OUT / "SHA256SUMS").write_text((ROOT / "SHA256SUMS").read_text())
# A fixed final-EMA latitude view; report raw results alongside in the CSV.
bands = ["-90:-60", "-60:-30", "-30:0", "0:30", "30:60", "60:90"]
channels = ["zos", "thetao_0", "so_0", "uo_0", "vo_0"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
for ax, arm in zip(axes, ["concat", "film"]):
    data = summary[
        (summary.arm == arm)
        & (summary.window == "full")
        & (summary.checkpoint == "ema_latest")
    ]
    pivot = data.pivot(
        index="channel", columns="latitude_band", values="rmse_change_percent"
    ).reindex(index=channels, columns=bands)
    im = ax.imshow(pivot, cmap="RdBu_r", vmin=-5, vmax=5, aspect="auto")
    for i in range(5):
        for j in range(6):
            ax.text(
                j, i, f"{pivot.iloc[i, j]:+.1f}", ha="center", va="center", fontsize=8
            )
    ax.set_title("Input concatenation" if arm == "concat" else "Geometry FiLM")
    ax.set_xticks(
        range(6),
        ["90–60°S", "60–30°S", "30–0°S", "0–30°N", "30–60°N", "60–90°N"],
        rotation=45,
        ha="right",
    )
    ax.set_yticks(
        range(5),
        [
            "SSH",
            "Temperature, 2.5 m",
            "Salinity, 2.5 m",
            "Zonal velocity, 2.5 m",
            "Meridional velocity, 2.5 m",
        ],
    )
fig.suptitle(
    "Final EMA: rollout RMSE change vs control (%)\nSingle seed; negative is better; pooled over all 598 forecast times",
    fontsize=11,
)

fig.subplots_adjust(left=0.2, bottom=0.25, top=0.78, right=0.82, wspace=0.15)
cax = fig.add_axes((0.85, 0.25, 0.018, 0.53))
fig.colorbar(im, cax=cax, label="RMSE change (%)", extend="both")
fig.savefig(OUT / "latitude-ema.png", dpi=160)
plt.close(fig)
print(runtime.to_string(index=False))
print(
    surface[
        ["arm", "checkpoint", "channel", "rmse", "rmse_change_percent", "bias"]
    ].to_string(index=False)
)
print("FiLM latitude temperature and SSH:")
print(
    summary[
        (summary.arm == "film")
        & (summary.window == "full")
        & (summary.checkpoint == "ema_latest")
        & (summary.channel.isin(["zos", "thetao_0"]))
    ][["channel", "latitude_band", "rmse_change_percent"]].to_string(index=False)
)
