# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compute Samudra-2 paper baseline comparison metrics for a rollout.

Loads predictions.zarr (split-per-level) and OM4.zarr ground truth (lev-dim),
aligns time, and computes the three diagnostics emphasized in Yuan et al. 2026
(Samudra 2 paper):

1. Niño 3.4 R² and RMSE on deseasoned SST anomalies.
2. Detrended global-mean temperature R² for the three depth bands
   (0-700 m, 700-2000 m, 2000-7000 m).
3. Deseasoned temperature snapshot near 2022-09-30 at 2.5 m, 700 m, 2000 m
   (correlation + RMSE vs OM4).

Prints a markdown table comparing to the paper's published Samudra-2 (1°) numbers.
"""

import argparse

import numpy as np
import xarray as xr

PAPER_BASELINE = {
    "nino34_r2": 0.93,
    "nino34_rmse": 0.222,
    "globalT_r2_upper": 0.87,
    "globalT_r2_mid": -1.60,
    "globalT_r2_deep": -16.14,
    "snap_2.5m_corr": 0.6741,
    "snap_2.5m_rmse": 0.4580,
    "snap_700m_corr": 0.2871,
    "snap_700m_rmse": 0.2455,
    "snap_2000m_corr": 0.3565,
    "snap_2000m_rmse": 0.0421,
}


def stack_pred_levels(pred_ds: xr.Dataset, var: str) -> xr.DataArray:
    arrays = [pred_ds[f"{var}_{i}"] for i in range(19)]
    stacked = xr.concat(arrays, dim="lev")
    stacked = stacked.assign_coords(lev=pred_ds["lev"].values)
    return stacked.transpose("time", "lev", "lat", "lon")


def deseason(da: xr.DataArray) -> xr.DataArray:
    clim = da.groupby("time.dayofyear").mean("time")
    return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear", errors="ignore")


def detrend_linear(da: xr.DataArray) -> xr.DataArray:
    coeffs = da.polyfit(dim="time", deg=1)
    fitted = xr.polyval(da["time"], coeffs.polyfit_coefficients)
    return da - fitted


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt, yp = y_true[mask], y_pred[mask]
    ss_res = float(((yt - yp) ** 2).sum())
    ss_tot = float(((yt - yt.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])


def main(pred_path: str, truth_path: str) -> None:
    print(f"Loading predictions: {pred_path}", flush=True)
    pred = xr.open_zarr(pred_path)
    print(
        f"  time={pred.sizes['time']}, lat={pred.sizes['lat']}, lon={pred.sizes['lon']}, lev={pred.sizes['lev']}",
        flush=True,
    )
    print(
        f"  pred time range: {pred.time.values[0]} to {pred.time.values[-1]}",
        flush=True,
    )

    print(f"Loading ground truth: {truth_path}", flush=True)
    truth = xr.open_zarr(truth_path).rename({"y": "lat", "x": "lon"})
    print(
        f"  truth time range: {truth.time.values[0]} to {truth.time.values[-1]}",
        flush=True,
    )

    # Align time. Pred uses julian days since 1958-01-01 (float64); truth uses cftime.
    # Reindex truth to pred's timestamps (nearest-neighbor).
    truth = truth.reindex(time=pred.time, method="nearest", tolerance=None)
    print(f"  aligned truth -> {truth.sizes['time']} timesteps", flush=True)

    pred_T = stack_pred_levels(pred, "thetao")
    truth_T = truth["thetao"].transpose("time", "lev", "lat", "lon")
    levs = pred.lev.values
    print(f"  levs (m): {[round(float(v), 1) for v in levs]}", flush=True)

    cos_lat = np.cos(np.deg2rad(pred.lat))

    # ------------------------------------------------------------------
    # 1. Niño 3.4: 5N-5S, 170W-120W (lon 190-240) on SST (lev 0).
    # ------------------------------------------------------------------
    print("\n--- Niño 3.4 ---", flush=True)
    region_T = truth_T.isel(lev=0).sel(lat=slice(-5, 5), lon=slice(190, 240))
    region_P = pred_T.isel(lev=0).sel(lat=slice(-5, 5), lon=slice(190, 240))
    w = cos_lat.sel(lat=slice(-5, 5))
    nino_truth = region_T.weighted(w).mean(["lat", "lon"]).compute()
    nino_pred = region_P.weighted(w).mean(["lat", "lon"]).compute()

    # Deseason both, then compare anomalies.
    nino_truth_ano = deseason(nino_truth).values
    nino_pred_ano = deseason(nino_pred).values
    nino_r2 = r2(nino_truth_ano, nino_pred_ano)
    nino_rmse = rmse(nino_truth_ano, nino_pred_ano)
    print(f"  R²   = {nino_r2:.4f}", flush=True)
    print(f"  RMSE = {nino_rmse:.4f} °C", flush=True)

    # ------------------------------------------------------------------
    # 2. Detrended global-mean T R² across three depth bands.
    # ------------------------------------------------------------------
    print("\n--- Detrended global-mean T R² ---", flush=True)
    # Layer thickness from lev coordinate (cell-edge approx).
    edges = np.concatenate(
        [[0.0], (levs[:-1] + levs[1:]) / 2.0, [levs[-1] + (levs[-1] - levs[-2]) / 2.0]]
    )
    dz = np.diff(edges)
    dz_da = xr.DataArray(dz, dims="lev", coords={"lev": levs})
    weight_3d = cos_lat * dz_da  # broadcasts over lon

    bands = {
        "upper": (0, 700),
        "mid": (700, 2000),
        "deep": (2000, 7000),
    }
    band_results = {}
    for name, (lo, hi) in bands.items():
        sub_levs = levs[(levs >= lo) & (levs < hi)]
        if len(sub_levs) == 0:
            print(f"  {name} ({lo}-{hi} m): no levels in range", flush=True)
            continue
        truth_sub = truth_T.sel(lev=sub_levs)
        pred_sub = pred_T.sel(lev=sub_levs)
        ws = weight_3d.sel(lev=sub_levs)
        truth_mean = truth_sub.weighted(ws).mean(["lev", "lat", "lon"]).compute()
        pred_mean = pred_sub.weighted(ws).mean(["lev", "lat", "lon"]).compute()
        gt_dt = detrend_linear(truth_mean).values
        pr_dt = detrend_linear(pred_mean).values
        band_r2 = r2(gt_dt, pr_dt)
        band_results[name] = band_r2
        print(
            f"  {name} ({lo}-{hi} m, {len(sub_levs)} lev): R² = {band_r2:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # 3. Deseasoned snapshot near 2022-09-30 at 2.5 m, 700 m, 2000 m.
    # ------------------------------------------------------------------
    print("\n--- Deseasoned T snapshot @ 2022-09-30 ---", flush=True)
    snap_results = {}
    import cftime

    target = cftime.DatetimeJulian(2022, 9, 30, 12)
    times = pred.time.values
    deltas = np.array([abs((t - target).total_seconds()) for t in times])
    target_idx = int(np.argmin(deltas))
    print(f"  snapshot t = {pred.time.values[target_idx]}", flush=True)

    for depth in [2.5, 700, 2000]:
        # Find nearest level
        i = int(np.argmin(np.abs(levs - depth)))
        lev_actual = float(levs[i])
        truth_lev = truth_T.isel(lev=i)
        pred_lev = pred_T.isel(lev=i)
        truth_des = deseason(truth_lev)
        pred_des = deseason(pred_lev)
        ts = pred.time.values[target_idx]
        gt_snap = truth_des.sel(time=ts, method="nearest").compute().values
        pr_snap = pred_des.sel(time=ts, method="nearest").compute().values
        c = corr(gt_snap, pr_snap)
        e = rmse(gt_snap, pr_snap)
        snap_results[depth] = (c, e, lev_actual)
        print(
            f"  depth ≈ {depth:>4} m (actual lev={lev_actual:.1f}): corr={c:.4f}, RMSE={e:.4f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # Markdown table
    # ------------------------------------------------------------------
    print("\n# Comparison: large_kernel_v4 vs Samudra-2 (paper, 1°)\n", flush=True)
    print("| Metric | Paper (Samudra-2 1°) | Ours (large_kernel_v4) |", flush=True)
    print("|---|---|---|", flush=True)
    print(
        f"| Niño 3.4 R² | {PAPER_BASELINE['nino34_r2']:.2f} | {nino_r2:.4f} |",
        flush=True,
    )
    print(
        f"| Niño 3.4 RMSE (°C) | {PAPER_BASELINE['nino34_rmse']:.3f} | {nino_rmse:.4f} |",
        flush=True,
    )
    if "upper" in band_results:
        print(
            f"| Global-mean T R² (0-700 m) | {PAPER_BASELINE['globalT_r2_upper']:.2f} | {band_results['upper']:.4f} |",
            flush=True,
        )
    if "mid" in band_results:
        print(
            f"| Global-mean T R² (700-2000 m) | {PAPER_BASELINE['globalT_r2_mid']:.2f} | {band_results['mid']:.4f} |",
            flush=True,
        )
    if "deep" in band_results:
        print(
            f"| Global-mean T R² (2000-7000 m) | {PAPER_BASELINE['globalT_r2_deep']:.2f} | {band_results['deep']:.4f} |",
            flush=True,
        )
    if 2.5 in snap_results:
        c, e, _ = snap_results[2.5]
        print(
            f"| Snapshot 2.5 m (corr / RMSE) | 0.67 / 0.46 | {c:.4f} / {e:.4f} |",
            flush=True,
        )
    if 700 in snap_results:
        c, e, _ = snap_results[700]
        print(
            f"| Snapshot 700 m (corr / RMSE) | 0.29 / 0.25 | {c:.4f} / {e:.4f} |",
            flush=True,
        )
    if 2000 in snap_results:
        c, e, _ = snap_results[2000]
        print(
            f"| Snapshot 2000 m (corr / RMSE) | 0.36 / 0.04 | {c:.4f} / {e:.4f} |",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        default="/scratch/am16581/runs/2026-05-06-om4_samudra_v2_large_kernel_v4_eval/predictions.zarr",
    )
    parser.add_argument(
        "--truth",
        default="/scratch/am16581/data/om4_onedeg_v3/OM4.zarr",
    )
    args = parser.parse_args()
    main(args.pred, args.truth)
