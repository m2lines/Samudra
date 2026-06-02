# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Paper-canonical metrics driver for Samudra-2 paper comparison.

Loads an Ocean-Emulator predictions.zarr and an OM4.zarr ground truth,
then computes the headline diagnostics from Yuan et al. 2026 (Samudra-2):

1. Niño 3.4 R² / corr / MAE / RMSE on smoothed, deseasoned anomalies
   (5-month rolling mean over per-pixel anomalies, area-weighted).
2. Detrended global-mean temperature R² for depth bands
   (0-700 m, 700-2000 m, 2000-7000 m), volume-weighted with wet_mask.
3. Deseasoned T snapshot RMSE / corr at depth indices 0, 10, 14
   (2.5 m, 775 m, 2400 m — matching the paper's DEPTH_LAYERS).

Metric implementations are vendored verbatim from `functions.py` (the
paper's plotting/analysis code, sibling file in this directory). They
are inlined here rather than imported because `functions.py` has heavy
matplotlib/cartopy/cmocean dependencies not needed for the metrics.
"""

import argparse

import numpy as np
import xarray as xr


def stack_pred_levels(pred: xr.Dataset, var: str, n_levels: int = 19) -> xr.DataArray:
    """Stack pred's per-level vars (var_0..var_{n-1}) into (time, lev, y, x)."""
    arrays = [pred[f"{var}_{i}"] for i in range(n_levels)]
    stacked = xr.concat(arrays, dim="lev")
    return stacked.transpose("time", "lev", "y", "x")


def build_areacello(y: xr.DataArray, x: xr.DataArray) -> xr.DataArray:
    """Construct areacello(y, x) for a regular lat-lon grid.

    On a regular grid, cell area = R^2 * cos(lat) * dlat * dlon. The constant
    R^2 * dlat * dlon factors out of every weighted mean, so we only need the
    relative shape: cos(lat) broadcast over lon.
    """
    coslat = np.cos(np.deg2rad(y))
    area = coslat.broadcast_like(
        xr.DataArray(np.ones(x.size), dims=["x"], coords={"x": x})
    )
    area = area.transpose("y", "x")
    area.name = "areacello"
    return area


def build_dz(lev: xr.DataArray) -> xr.DataArray:
    """Construct layer thicknesses dz(lev) from level midpoints.

    Approximates OM4's actual layer thicknesses by taking cell edges at
    midpoints between adjacent lev values. The top edge is at 0 and the
    bottom edge is extrapolated symmetrically from the last interior gap.
    """
    levs = lev.values.astype(float)
    edges = np.concatenate(
        [[0.0], (levs[:-1] + levs[1:]) / 2.0, [levs[-1] + (levs[-1] - levs[-2]) / 2.0]]
    )
    dz = np.diff(edges)
    return xr.DataArray(dz, dims="lev", coords={"lev": lev}, name="dz")


OM4_LEV_M = np.array(
    [
        2.5,
        10.0,
        22.5,
        40.0,
        65.0,
        105.0,
        165.0,
        250.0,
        375.0,
        550.0,
        775.0,
        1050.0,
        1400.0,
        1850.0,
        2400.0,
        3100.0,
        4000.0,
        5000.0,
        6000.0,
    ]
)


def load_pred(pred_path: str) -> xr.Dataset:
    """Load predictions, stack thetao_0..18, rename to paper's (y, x) convention.

    Older eval pipelines wrote per-level vars without a `lev` coord; we inject
    OM4_LEV_M in that case so downstream band slicing works.
    """
    pred = xr.open_zarr(pred_path)
    rename_map = {}
    if "lat" in pred.dims:
        rename_map["lat"] = "y"
    if "lon" in pred.dims:
        rename_map["lon"] = "x"
    pred = pred.rename(rename_map)
    if "lev" not in pred.coords:
        pred = pred.assign_coords(lev=("lev", OM4_LEV_M))
    thetao = stack_pred_levels(pred, "thetao")
    out = xr.Dataset(
        coords={"time": pred.time, "lev": pred.lev, "y": pred.y, "x": pred.x}
    )
    out["thetao"] = thetao
    return out


def load_truth(truth_path: str) -> xr.Dataset:
    """Load truth and ensure thetao has (time, lev, y, x) dim order."""
    truth = xr.open_zarr(truth_path)
    if "thetao" not in truth:
        raise ValueError(f"truth at {truth_path} has no stacked 'thetao'")
    thetao = truth["thetao"].transpose("time", "lev", "y", "x")
    out = xr.Dataset(
        coords={"time": truth.time, "lev": truth.lev, "y": truth.y, "x": truth.x}
    )
    out["thetao"] = thetao
    return out


def align_truth_to_pred(truth: xr.Dataset, pred: xr.Dataset) -> xr.Dataset:
    """Reindex truth to pred's time axis via nearest-neighbor."""
    return truth.reindex(time=pred.time, method="nearest")


def attach_grid_metrics(
    ds: xr.Dataset, areacello: xr.DataArray, dz: xr.DataArray
) -> xr.Dataset:
    """Add areacello and dz as coords on the dataset (paper functions expect ds['areacello'], ds['dz'])."""
    ds = ds.assign_coords(areacello=areacello, dz=dz)
    return ds


# ---------------------------------------------------------------------------
# Metric helpers (vendored verbatim from paper functions.py for fidelity).
# ---------------------------------------------------------------------------


def compute_nino34_index(sst, area, dt=5, window=150):
    """Verbatim copy of functions.py:5148 — compute_nino34_index.

    Steps:
        1. Select Niño 3.4 region (190-240°E, 5°S-5°N)
        2. Remove climatology (dayofyear mean), per-pixel
        3. Apply rolling mean smoothing (window/dt timesteps)
        4. Area-weighted spatial mean
    """
    sst = sst.load()
    sst_nino = sst.sel(x=slice(190, 240), y=slice(-5, 5))
    area_nino = area.sel(x=slice(190, 240), y=slice(-5, 5)).load()

    clim = sst_nino.groupby("time.dayofyear").mean("time").compute()
    window_steps = int(window / dt)

    anom = sst_nino.copy()
    for i, t in enumerate(sst_nino.time.values):
        day = int(t.dayofyr)
        anom[i] = (sst_nino[i] - clim.sel(dayofyear=day)).data

    anom = anom.rolling(time=window_steps).mean()
    nino34 = anom.weighted(area_nino).mean(["x", "y"])

    return nino34[window_steps:]


def _detrend_1d(ts: np.ndarray) -> np.ndarray:
    """Paper's _detrend_1d (functions.py:749). Removes a linear trend but
    preserves the time-mean — R² is invariant to this, kept for fidelity."""
    n = len(ts)
    t = np.arange(n, dtype=float)
    t_mean = t.mean()
    ts_mean = ts.mean()
    slope = np.sum((t - t_mean) * (ts - ts_mean)) / np.sum((t - t_mean) ** 2)
    return ts - (slope * (t - t_mean) + ts_mean)


def _compute_band_ts(
    ds: xr.Dataset, depth_slice: slice, area: xr.DataArray, wet_mask: xr.DataArray
) -> xr.DataArray:
    """Paper's nested `_compute_ts` from
    plot_mean_timeseries_detrended_comparison_together (functions.py:766).

    Volume-weighted global mean of thetao over a depth band, with wet_mask
    zeroing land cells in the denominator."""
    field = ds["thetao"].sel(lev=depth_slice)
    dz = ds["dz"].sel(lev=depth_slice)
    vol_weight = dz * area
    num = (field * vol_weight).sum(["x", "y", "lev"]).compute()
    den = (vol_weight * wet_mask.sel(lev=depth_slice)).sum(["x", "y", "lev"]).compute()
    return num / den


def _deseason(da: xr.DataArray) -> xr.DataArray:
    """Per-pixel deseason: subtract dayofyear climatology."""
    clim = da.groupby("time.dayofyear").mean("time")
    return (da.groupby("time.dayofyear") - clim).drop_vars("dayofyear", errors="ignore")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ---------------------------------------------------------------------------
# Top-level metric blocks.
# ---------------------------------------------------------------------------


def nino34_metrics(
    pred: xr.Dataset, truth: xr.Dataset, dt: int = 5, window: int = 150
) -> dict:
    """Niño 3.4 R² / corr / MAE / RMSE via paper's compute_nino34_index pipeline."""
    sst_truth = truth["thetao"].isel(lev=0)
    sst_pred = pred["thetao"].isel(lev=0)
    area = truth["areacello"]

    nino_truth = compute_nino34_index(sst_truth, area, dt=dt, window=window)
    nino_pred = compute_nino34_index(sst_pred, area, dt=dt, window=window)

    n = min(len(nino_truth), len(nino_pred))
    yt = nino_truth.values[:n]
    yp = nino_pred.values[:n]
    mask = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp = yt[mask], yp[mask]

    return {
        "n_points": int(yt.size),
        "r2": _r2(yt, yp),
        "corr": float(np.corrcoef(yt, yp)[0, 1]),
        "mae": float(np.mean(np.abs(yt - yp))),
        "rmse": float(np.sqrt(np.mean((yt - yp) ** 2))),
    }


def depth_band_metrics(pred: xr.Dataset, truth: xr.Dataset) -> dict:
    """Detrended global-mean T R² per depth band (paper-canonical wet/vol weights)."""
    area = truth["areacello"]
    wet_mask = xr.where(np.isnan(truth["thetao"].isel(time=0)), 0.0, 1.0)

    bands = [
        ("upper", slice(0, 700)),
        ("mid", slice(700, 2000)),
        ("deep", slice(2000, 7000)),
    ]
    out = {}
    for name, dslice in bands:
        ts_truth = _compute_band_ts(truth, dslice, area, wet_mask).values
        ts_pred = _compute_band_ts(pred, dslice, area, wet_mask).values
        n = min(len(ts_truth), len(ts_pred))
        ts_truth_dt = _detrend_1d(ts_truth[:n])
        ts_pred_dt = _detrend_1d(ts_pred[:n])
        out[name] = {"r2": _r2(ts_truth_dt, ts_pred_dt), "n_steps": int(n)}
    return out


def snapshot_metrics(pred: xr.Dataset, truth: xr.Dataset, time_idx: int = -1) -> dict:
    """Deseasoned T snapshot RMSE/corr at paper's DEPTH_LAYERS indices: 0, 10, 14.

    The snapshot is the deseasoned anomaly at `time_idx` (default: last
    timestep — end of rollout). Paper's `plot_snapshot_comparison_multidepth_together`
    accepts a list of time_indices; the published table values pick one
    specific date. The user can override via --snapshot_time_idx.
    """
    truth_des = _deseason(truth["thetao"])
    pred_des = _deseason(pred["thetao"])
    wet = truth["thetao"].isel(time=0).notnull()

    layer_indices = [(0, "2.5m"), (10, "775m_paper-700m"), (14, "2400m_paper-2000m")]
    out = {}
    for idx, label in layer_indices:
        t_snap = truth_des.isel(time=time_idx, lev=idx).compute().values
        p_snap = pred_des.isel(time=time_idx, lev=idx).compute().values
        valid = wet.isel(lev=idx).values & ~np.isnan(t_snap) & ~np.isnan(p_snap)
        tv = t_snap[valid]
        pv = p_snap[valid]
        out[label] = {
            "lev_idx": idx,
            "corr": float(np.corrcoef(tv, pv)[0, 1]),
            "rmse": float(np.sqrt(np.mean((tv - pv) ** 2))),
            "n_pixels": int(valid.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(
    pred_path: str,
    truth_path: str,
    tag: str,
    snapshot_time_idx: int = -1,
    json_out: str | None = None,
) -> dict:
    print(f"\n# Paper-canonical comparison: {tag}", flush=True)
    print(f"  pred:  {pred_path}", flush=True)
    print(f"  truth: {truth_path}", flush=True)

    pred = load_pred(pred_path)
    truth = load_truth(truth_path)
    print(
        f"  pred  dims: {dict(pred.sizes)} | time={pred.time.values[0]} .. {pred.time.values[-1]}",
        flush=True,
    )
    print(
        f"  truth dims: {dict(truth.sizes)} | time={truth.time.values[0]} .. {truth.time.values[-1]}",
        flush=True,
    )

    truth = align_truth_to_pred(truth, pred)
    print(f"  aligned truth -> {truth.sizes['time']} timesteps", flush=True)

    area = build_areacello(pred["y"], pred["x"])
    dz = build_dz(pred["lev"])
    pred = attach_grid_metrics(pred, area, dz)
    truth = attach_grid_metrics(truth, area, dz)

    print(
        "\n--- Niño 3.4 (5-mo rolling, per-pixel deseason, area-weighted) ---",
        flush=True,
    )
    nino = nino34_metrics(pred, truth)
    print(
        f"  R²={nino['r2']:.4f}  corr={nino['corr']:.4f}  "
        f"MAE={nino['mae']:.4f}  RMSE={nino['rmse']:.4f}  (n={nino['n_points']})",
        flush=True,
    )

    print("\n--- Detrended global-mean T R² (vol-weighted, wet-masked) ---", flush=True)
    bands = depth_band_metrics(pred, truth)
    for name in ["upper", "mid", "deep"]:
        print(f"  {name:5s} R²={bands[name]['r2']:+.4f}", flush=True)

    snap_t = pred.time.isel(time=snapshot_time_idx).values
    print(
        f"\n--- Deseasoned T snapshot @ {snap_t} (time_idx={snapshot_time_idx}) ---",
        flush=True,
    )
    snaps = snapshot_metrics(pred, truth, time_idx=snapshot_time_idx)
    for label, m in snaps.items():
        print(
            f"  lev_idx={m['lev_idx']:2d} ({label:<22}): corr={m['corr']:.4f}  RMSE={m['rmse']:.4f}",
            flush=True,
        )

    summary = {
        "tag": tag,
        "pred_path": pred_path,
        "truth_path": truth_path,
        "snapshot_time": str(snap_t),
        "nino34": nino,
        "bands": bands,
        "snapshots": snaps,
    }
    if json_out:
        import json

        with open(json_out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nWrote {json_out}", flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="predictions.zarr path")
    parser.add_argument("--truth", required=True, help="OM4 truth zarr path")
    parser.add_argument(
        "--tag", default="run", help="label for the run in printed output"
    )
    parser.add_argument(
        "--snapshot_time_idx",
        type=int,
        default=-1,
        help="time index for snapshot metric (-1 = last)",
    )
    parser.add_argument("--json_out", default=None, help="optional JSON summary path")
    args = parser.parse_args()
    main(args.pred, args.truth, args.tag, args.snapshot_time_idx, args.json_out)
