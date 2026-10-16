# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Residual-anomaly forecast errors from saved continuous rollouts and OM4."""

import argparse
import hashlib
import json
from pathlib import Path

import dask
import numpy as np
import xarray as xr

from samudra.metrics.kernels import coerce_datetime_values, detrended_deseasonalized

ARMS = ("control", "aligned", "seasonal")


def residual(field):
    # Use the production metric's ordering, in float64 to avoid climatology
    # subtraction losing precision for large background temperatures/salinities.
    return detrended_deseasonalized(field.astype("float64"))


def error_moments(gold_residual, model_residual):
    gold_residual, model_residual = xr.align(
        gold_residual, model_residual, join="exact"
    )
    return {
        "mse": ((model_residual - gold_residual) ** 2).mean("time").values,
        "model_variance": model_residual.var("time", ddof=0).values,
    }


def compute(root, data_root, output):
    output.mkdir(exist_ok=False, parents=True)
    dask.config.set(scheduler="threads", num_workers=8)
    grid = dict(np.load(root / "control-ema/grid.npz"))
    channels = grid["channels"].tolist()
    datasets = {}
    times = None
    receipts = {}
    for arm in ARMS:
        directory = root / f"{arm}-ema"
        receipt = directory / "COMPLETE.json"
        receipts[arm] = json.loads(receipt.read_text())
        assert not receipts[arm]["smoke"]
        for key, values in grid.items():
            np.testing.assert_array_equal(np.load(directory / "grid.npz")[key], values)
        ds = xr.open_zarr(directory / "continuous/predictions.zarr", chunks={})
        stamps = coerce_datetime_values(ds.time.values)
        if times is None:
            times = stamps
        assert times.equals(stamps)
        datasets[arm] = ds.assign_coords(time=stamps)
    assert times is not None and len(times) == 598
    assert times.is_unique and times.is_monotonic_increasing
    gold = xr.open_zarr(data_root / "OM4.zarr", chunks={})
    gold_times = coerce_datetime_values(gold.time.values)
    positions = gold_times.get_indexer(times)
    assert (positions >= 0).all()
    gold = gold.isel(time=positions).assign_coords(time=times)
    for ds in [gold, *datasets.values()]:
        for dim in ("x", "y"):
            np.testing.assert_array_equal(ds[dim], datasets["control"][dim])
        for coord in ("lat", "lon"):
            actual = ds[coord].values
            expected = grid[coord]
            if actual.ndim == 2:
                expected = np.broadcast_to(
                    expected[:, None] if coord == "lat" else expected[None, :],
                    actual.shape,
                )
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    shape = grid["wet"].shape
    result = {
        arm: {
            key: np.full(shape, np.nan, dtype="float32")
            for key in ("mse", "model_variance", "raw_mse")
        }
        for arm in ARMS
    }
    gold_variance = np.full(shape, np.nan, dtype="float32")
    checks = {arm: 0.0 for arm in ARMS}
    saved_raw = {
        arm: np.load(root / f"{arm}-ema/continuous/scores.npz")["mean_squared_error"]
        for arm in ARMS
    }
    for variable in ("thetao", "so", "uo", "vo", "zos"):
        # A saved Zarr chunk contains every depth. Read each variable once, then
        # residualize one depth at a time instead of rereading it 19 times.
        models = {arm: datasets[arm][variable].load() for arm in ARMS}
        for channel, name in enumerate(channels):
            if name.split("_")[0] != variable:
                continue
            wet = xr.DataArray(
                grid["wet"][channel],
                dims=("y", "x"),
                coords={"y": gold.y, "x": gold.x},
            )
            truth = gold[name].load().where(wet).astype("float64")
            assert np.isfinite(truth.values[:, grid["wet"][channel]]).all()
            gold_residual = residual(truth)
            gold_variance[channel] = gold_residual.var("time", ddof=0).values
            for arm in ARMS:
                model = models[arm]
                if variable != "zos":
                    model = model.isel(lev=int(name.rsplit("_", 1)[1]))
                model = model.where(wet).astype("float64")
                assert np.isfinite(model.values[:, grid["wet"][channel]]).all()
                raw = ((model - truth) ** 2).mean("time").values
                scaled_raw = raw / grid["std"][channel] ** 2
                support = grid["wet"][channel]
                reference = saved_raw[arm][channel]
                np.testing.assert_allclose(
                    scaled_raw[support],
                    reference[support],
                    rtol=4e-4,
                    atol=5e-6,
                    err_msg=f"Saved physical rollout does not reproduce original errors: {arm}/{name}",
                )
                checks[arm] = max(
                    checks[arm],
                    float(np.max(np.abs(scaled_raw[support] - reference[support]))),
                )
                moments = error_moments(gold_residual, residual(model))
                for key, values in moments.items():
                    result[arm][key][channel] = values
                result[arm]["raw_mse"][channel] = raw
                del model, moments
            del truth, gold_residual
            print(f"Completed {name}", flush=True)
        del models
    for arm in ARMS:
        np.savez_compressed(output / f"{arm}-ema-residual.npz", **result[arm])
    np.savez_compressed(output / "gold-residual.npz", variance=gold_variance)
    np.savez_compressed(output / "grid.npz", **grid)
    manifest = dict(
        period=[str(times[0]), str(times[-1])],
        n_times=len(times),
        gold=str(data_root / "OM4.zarr"),
        evaluation_root=str(root),
        checkpoints=receipts,
        preprocessing="Independently per field/cell: own pentad climatology, then own OLS linear trend with intercept; float64; production detrended_deseasonalized kernel",
        climatology_fit="Entire shared saved-rollout period; retrospective diagnostic",
        error="sqrt(mean_time((residual(model) - residual(gold))**2)); physical units",
        support="Original per-channel wet mask, exact common timestamps, original grid; no interpolation",
        raw_normalized_mse_max_abs_discrepancy=checks,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    (output / "COMPLETE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


def summarize_and_render(output):
    from analyze import (
        draw_fields,
        normalized_weights,
        regions,
        selected_depth_channels,
        write_csv,
    )

    grid = dict(np.load(output / "grid.npz"))
    channels = grid["channels"].tolist()
    maps = {arm: dict(np.load(output / f"{arm}-ema-residual.npz")) for arm in ARMS}
    masks = regions(grid["lat"], grid["lon"])
    weights = normalized_weights(grid["wet"], grid["area"], masks)
    groups = {"all": np.arange(len(channels))}
    groups.update(
        {
            var: np.array([i for i, c in enumerate(channels) if c.split("_")[0] == var])
            for var in ("thetao", "so", "uo", "vo", "zos")
        }
    )
    summaries, detailed = [], []
    gold_var = np.load(output / "gold-residual.npz")["variance"]
    std = grid["std"]
    for arm in ARMS[1:]:
        for r, region in enumerate(masks):
            active = weights[r].sum((-2, -1)) > 0
            for metric in ("mse", "raw_mse"):
                b = np.nansum(maps["control"][metric] * weights[r], axis=(-2, -1))
                t = np.nansum(maps[arm][metric] * weights[r], axis=(-2, -1))
                for group, indices in groups.items():
                    which = indices[active[indices]]
                    if not len(which):
                        continue
                    base = float(np.sqrt(np.mean(b[which] / std[which] ** 2)))
                    treatment = float(np.sqrt(np.mean(t[which] / std[which] ** 2)))
                    summaries.append(
                        dict(
                            arm=arm,
                            region=region,
                            group=group,
                            metric="residual" if metric == "mse" else "raw",
                            control_rmse=base,
                            treatment_rmse=treatment,
                            change_percent=100 * (treatment / base - 1),
                        )
                    )
                if metric == "mse":
                    gv = np.nansum(gold_var * weights[r], axis=(-2, -1))
                    mv = np.nansum(
                        maps[arm]["model_variance"] * weights[r], axis=(-2, -1)
                    )
                    cv = np.nansum(
                        maps["control"]["model_variance"] * weights[r], axis=(-2, -1)
                    )
                    for i, name in enumerate(channels):
                        if active[i]:
                            detailed.append(
                                dict(
                                    arm=arm,
                                    region=region,
                                    channel=name,
                                    control_rmse=np.sqrt(b[i]),
                                    treatment_rmse=np.sqrt(t[i]),
                                    change_percent=100 * (np.sqrt(t[i] / b[i]) - 1),
                                    gold_residual_std=np.sqrt(gv[i]),
                                    control_residual_std=np.sqrt(cv[i]),
                                    model_residual_std=np.sqrt(mv[i]),
                                )
                            )
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "channels.csv", detailed)
    draw_fields(
        output,
        grid,
        channels,
        selected_depth_channels(grid, channels),
        {arm: np.sqrt(values["mse"]) for arm, values in maps.items()},
        signed=False,
        title="Residual-anomaly RMSE vs OM4; full continuous rollout; final EMA",
        filename_tag="residual-rmse-rollout",
    )
    print(json.dumps([r for r in summaries if r["group"] == "all"], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    if args.render:
        summarize_and_render(args.output)
    else:
        assert args.root is not None and args.data_root is not None
        compute(args.root, args.data_root, args.output)


if __name__ == "__main__":
    main()
