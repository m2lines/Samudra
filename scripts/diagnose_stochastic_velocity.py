# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched held-out multi-date initializer diagnostics; no processor or training."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch


def metrics(x, y, mask, lat):
    """Physical-unit per-channel metrics with fair CRPS and amplitude diagnostics."""
    w = mask * torch.cos(torch.deg2rad(lat))[None, :, None]
    den = w.sum((-2, -1))

    def avg(v):
        return (v * w).sum((-2, -1)) / den

    mean = x.mean(0)
    absolute = (x - y).abs().mean(0)
    spread = sum((x[i] - x[j]).abs() for i in range(len(x)) for j in range(i)) / (
        len(x) * (len(x) - 1)
    )
    ym, pm = avg(y), avg(mean)
    yc, pc = y - ym[:, None, None], mean - pm[:, None, None]
    cov = avg(yc * pc)
    result = dict(
        mean_mse=avg((mean - y).square()),
        member_mse=avg((x - y).square()).mean(0),
        absolute=avg(absolute),
        spread=avg(spread),
        fair_crps=avg(absolute - spread),
        bias=avg(mean - y),
        gold_power=avg(y.square()),
        mean_power=avg(mean.square()),
        member_power=avg(x.square()).mean(0),
        covariance=cov,
        gold_variance=avg(yc.square()),
        prediction_variance=avg(pc.square()),
        zero_mse=avg(y.square()),
        coverage=avg(((y >= x.min(0).values) & (y <= x.max(0).values)).float()),
        mean_amplitude_slope=cov / avg(yc.square()).clamp_min(1e-20),
    )
    result["correlation"] = cov / (
        result["gold_variance"] * result["prediction_variance"]
    ).sqrt().clamp_min(1e-20)
    return {k: v.cpu().numpy() for k, v in result.items()}


def detailed(x, y, mask, lat):
    out = metrics(x, y, mask, lat)
    w = mask * torch.cos(torch.deg2rad(lat))[None, :, None]
    speed = (y[:19].square() + y[19:38].square()).sqrt()
    edges = [0, 0.01, 0.03, 0.1, 0.3, 1, float("inf")]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        support = (speed >= lo) & (speed < hi) & mask[:19].bool()
        both = torch.cat([support, support])
        weight = w[:38] * both
        den = weight.sum((-2, -1))
        mean = x[:, :38].mean(0)
        err = (mean - y[:38]).square()
        bins.append(
            dict(
                low=lo,
                high=hi if np.isfinite(hi) else None,
                area=den.cpu().tolist(),
                squared_error_sum=(err * weight).sum((-2, -1)).cpu().tolist(),
                predicted_power_sum=(mean.square() * weight)
                .sum((-2, -1))
                .cpu()
                .tolist(),
                gold_power_sum=(y[:38].square() * weight).sum((-2, -1)).cpu().tolist(),
            )
        )
    return out, bins


@torch.no_grad()
def main():
    import cftime
    import zarr  # type: ignore[import-untyped]

    from samudra.experiments.observation_pilot import atomic_json, digest
    from samudra.experiments.observation_training import Samples
    from samudra.experiments.stochastic_transition_train import (  # type: ignore[import-not-found]
        build_model,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--data", type=Path, default=Path("/scratch/jr7309/data/obs-d-pilot")
    )
    p.add_argument(
        "--om4", type=Path, default=Path("/scratch/jr7309/data/om4_onedeg_v3/OM4.zarr")
    )
    p.add_argument("--months", nargs="+", default=["2014-01", "2014-04", "2014-05"])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    data = Samples(a.data, "cuda", surface_fill="zero", global_observations=True)
    store = zarr.open_consolidated(str(a.om4), mode="r")
    ta = store["time"]
    labels = [
        d.strftime("%Y-%m-%dT%H:%M:%S")
        for d in cftime.num2date(
            ta[:], ta.attrs["units"], ta.attrs.get("calendar", "standard")
        )
    ]
    names = data.grid["names"].tolist()
    lat = data.tensor(data.grid["lat"])
    mask = np.stack(
        [store[f"mask_{i}"][:] for i in range(19)] * 4 + [store["mask_0"][:]]
    ).astype(bool)
    np.testing.assert_array_equal(mask, data.grid["mask"])
    np.testing.assert_allclose(store["x"][:], data.grid["lon"], atol=1e-6, rtol=0)
    np.testing.assert_allclose(store["y"][:], data.grid["lat"], atol=1e-6, rtol=0)
    cases = []
    for month in a.months:
        path = a.data / "validation" / f"{month}.npz"
        assert path in data.paths("validation")
        sample = data.load(path)
        wanted = (
            np.asarray(sample["raw"]["midpoints"][:19], dtype="datetime64[s]")
            .astype(str)
            .tolist()
        )
        if any(labels.count(t) != 1 for t in wanted):
            raise ValueError(f"{month}: history has no exact unique OM4 matches")
        ids = [labels.index(t) for t in wanted]
        assert np.all(np.diff(ids) == 1)
        assert wanted[-1] > "2013-08-04"
        surface = np.stack(
            [np.stack([store[n][i] for n in ["thetao_0", "zos"]]) for i in ids]
        ).astype("float32")
        gold = np.stack([store[n][ids[-1]] for n in names]).astype("float32")
        assert (
            np.isfinite(surface[:, mask[[38, 76]]]).all()
            and np.isfinite(gold[mask]).all()
        )
        normalized = np.where(
            mask[[38, 76]],
            (surface - data.grid["mean"][[38, 76], None, None])
            / data.grid["std"][[38, 76], None, None],
            0,
        )
        valid = data.tensor(np.broadcast_to(mask[[38, 76]], surface.shape).copy())[None]
        cases.append(
            (
                month,
                data.tensor(normalized)[None],
                valid,
                sample,
                data.tensor(np.where(mask, gold, 0)),
            )
        )
        atomic_json(
            dict(
                month=month,
                history_midpoints=wanted,
                history_indices=ids,
                sample_sha256=digest(path),
                seed=904003,
                members=8,
                grid_sha256=digest(a.data / "grid.npz"),
                om4_metadata_sha256=digest(a.om4 / ".zmetadata"),
                training_latest_target_bin_end="2013-08-04",
                input_scope="OM4 surface history plus matched ERA5; no interior or future input",
            ),
            a.output / f"input-{month}.json",
        )
    selection = [0, 9, 18, 19, 28, 37]
    np.savez_compressed(
        a.output / "grid.npz",
        lat=data.grid["lat"],
        lon=data.grid["lon"],
        mask=mask,
        names=np.array(names),
        selection=selection,
    )
    entries = json.loads(a.manifest.read_text())
    receipts = []
    for entry in entries:
        path = Path(entry["path"])
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["step"] == entry["step"]
        model = build_model(saved, data, use_ema=True).eval()
        del saved
        for month, surface, valid, sample, gold in cases:
            generator = torch.Generator(device="cuda").manual_seed(904003)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cond, _, _ = model.encode_native(
                    surface,
                    sample["atmosphere"][:, :19],
                    sample["contexts"][:, 18],
                    data.mask,
                    valid,
                    return_anchors=False,
                )
                cond = cond.unsqueeze(0).expand(8, *cond.shape).flatten(0, 1)
                initial = model.initialize(cond, None, None, data.mask, generator)[
                    :, -1
                ].float()
            assert torch.isfinite(initial).all()
            draws = initial * data.std[0, 0] + data.mean[0, 0]
            values, bins = detailed(draws, gold, data.mask, lat)
            name = f"{entry['variant']}-{entry['step']:05d}-{month}"
            row = dict(
                **entry,
                month=month,
                checkpoint_sha256=digest(path),
                metrics={k: v.tolist() for k, v in values.items()},
                speed_bins=bins,
            )
            atomic_json(row, a.output / f"{name}.json")
            receipts.append(name)
            if entry["step"] == 2000:
                np.savez_compressed(
                    a.output / f"{name}-maps.npz",
                    members=draws[:, selection].cpu().numpy(),
                    gold=gold[selection].cpu().numpy(),
                )
            print(
                json.dumps(
                    dict(
                        case=name,
                        u_rmse=float(np.sqrt(values["mean_mse"][:19].mean())),
                        v_rmse=float(np.sqrt(values["mean_mse"][19:38].mean())),
                    )
                ),
                flush=True,
            )
        del model, initial, draws, cond
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(
        dict(
            complete=True,
            results=receipts,
            script_sha256=digest(Path(__file__)),
            manifest_sha256=digest(a.manifest),
        ),
        a.output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
