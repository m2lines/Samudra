# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from samudra.utils.band_loss import masked_highpass

p = argparse.ArgumentParser()
p.add_argument("--diagnostics-dir", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
a = p.parse_args()
x = json.loads((a.diagnostics_dir / "comparison.json").read_text())
names = x["channels"]
out = a.output_dir
out.mkdir(parents=True, exist_ok=True)

rows = []
for model, d in x["models"].items():
    raw = d["rows"]
    energy = np.mean([r["hf_prediction_energy"] for r in raw], 0) / np.mean(
        [r["hf_truth_energy"] for r in raw], 0
    )
    corr = np.mean([r["hf_correlation"] for r in raw], 0)
    for lead_index, lead in enumerate(x["leads_days"]):
        for i, name in enumerate(names):
            rows.append(
                dict(
                    model=model,
                    lead_days=lead,
                    channel=name,
                    rmse_physical=d["rmse_physical"][lead_index][i],
                    hf_rmse_physical=d["hf_rmse_physical"][lead_index][i],
                    hf_energy_ratio=energy[lead_index, i],
                    hf_correlation=corr[lead_index, i],
                )
            )
with (out / "one-call-metrics.csv").open("w") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
for kind in ("raw", "ema"):
    baseline = x["models"]["mse_" + kind]
    b = x["models"]["band_" + kind]
    for name in ["thetao_0", "so_0", "uo_0", "vo_0", "zos"]:
        i = names.index(name)
        percent = 100 * (
            np.array(b["rmse_physical"])[:, i]
            / np.array(baseline["rmse_physical"])[:, i]
            - 1
        )
        wins = sum(
            r2["hf_mse_normalized"][0][i] < r1["hf_mse_normalized"][0][i]
            for r1, r2 in zip(baseline["rows"], b["rows"])
        )
        print(kind, name, "RMSE percent", percent, "HF wins5day", wins)
torch.set_num_threads(2)
data = {
    k: np.load(a.diagnostics_dir / (k + "-maps.npz"))
    for k in ["mse_raw", "band_raw", "control_raw"]
}
ref = data["mse_raw"]
lat = ref["lat"]
lon = ((ref["lon"] + 180) % 360) - 180
ix = np.argsort(lon)
lon = lon[ix]
ys = np.where((lat >= 15) & (lat <= 65))[0]
xs = np.where((lon >= -85) & (lon <= 10))[0]
for c, name, unit in [(1, "temperature", "°C"), (0, "sea-surface-height", "m")]:
    for j in range(2):
        truth = ref["truth_" + str(j)][c]
        mask = ref["mask"][c].astype(bool)

        def crop(a):
            return np.where(mask, a, np.nan)[:, ix][np.ix_(ys, xs)]

        def hp(a):
            return masked_highpass(
                torch.from_numpy(a.copy())[None, None], torch.from_numpy(mask)[None]
            ).numpy()[0, 0]

        fields = [truth] + [data[k]["prediction_" + str(j)][c] for k in data]
        vmin, vmax = np.nanpercentile(crop(truth), [1, 99])
        errmax = 1.0 if c == 1 else 0.08
        hmax = 0.35 if c == 1 else 0.025
        fig, axs = plt.subplots(3, 4, figsize=(12, 6.8), layout="constrained")
        for k, (title, f) in enumerate(
            zip(["Truth", "Plain MSE", "MSE + band", "Historical dynamic"], fields)
        ):
            for row in range(3):
                ax = axs[row, k]
                if k == 0 and row > 0:
                    ax.axis("off")
                    continue
                val = f if row == 0 else f - truth if row == 1 else hp(f - truth)
                im = ax.pcolormesh(
                    lon[xs],
                    lat[ys],
                    crop(val),
                    shading="auto",
                    cmap="viridis" if row == 0 else "RdBu_r",
                    vmin=vmin if row == 0 else -(errmax if row == 1 else hmax),
                    vmax=vmax if row == 0 else (errmax if row == 1 else hmax),
                    rasterized=True,
                )
                ax.set_facecolor("#dedede")
                ax.set_title(
                    title
                    if row == 0
                    else ("Prediction − truth" if row == 1 else "High-pass error"),
                    fontsize=9,
                )
                ax.tick_params(labelsize=7)
                if k == 3:
                    fig.colorbar(im, ax=axs[row, :], label=unit, shrink=0.8)
        date = x["models"]["mse_raw"]["rows"][j]["target_dates"][0]
        fig.suptitle(
            f"{name.replace('-', ' ').title()} | 5-day forecast, {date[:10]} | final raw checkpoints",
            fontsize=12,
        )
        fig.savefig(
            out / f"{name}-example-{j + 1}.jpg", dpi=115, pil_kwargs={"quality": 85}
        )
        plt.close(fig)
print("WROTE", [(p.name, p.stat().st_size) for p in out.iterdir()])
