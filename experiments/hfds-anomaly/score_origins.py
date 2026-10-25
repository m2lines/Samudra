"""Score saved quarterly rollouts with cosine-latitude ocean-area weights.

Raw RMSE/bias and normalization by the original training standard deviations.
Lead positions retain actual elapsed days in the output (including leap gaps).
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import xarray as xr

ROOT = Path("/scratch/jr7309/om4-hfds-ablation-20261021")
POSITIONS = [0, 1, 5, 7, 17, 35, 71]


def main():
    label = ["selected", "final-ema"][int(sys.argv[1])]
    truth = xr.open_zarr("/scratch/jr7309/data/om4_onedeg_v3/OM4.zarr", chunks=None)
    stds = xr.open_zarr(ROOT / "normalization/OM4_stds.zarr", chunks=None)
    out = ROOT / "scores" / label
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for origin_dir in sorted(
        (ROOT / "multi-origin" / f"with-{label}").glob("????-??-??")
    ):
        origin = origin_dir.name
        records = [
            json.loads(
                (
                    ROOT / "multi-origin" / f"{arm}-{label}" / origin / "COMPLETE.json"
                ).read_text()
            )
            for arm in ("with", "without")
        ]
        assert records[0]["target_times"] == records[1]["target_times"]
        for arm, record in zip(("with", "without"), records):
            pred = xr.open_zarr(
                ROOT / "multi-origin" / f"{arm}-{label}" / origin / "predictions.zarr",
                chunks=None,
            )
            assert [str(t) for t in pred.time.values] == record["target_times"]
            times = pred.time.values[POSITIONS]
            # Source and saved predictions use the same declared Julian calendar.
            ref = truth.sel(time=times)
            for variable in ("thetao", "so", "uo", "vo", "zos"):
                values = pred[variable].isel(time=POSITIONS).values
                if variable == "zos":
                    values = values[:, None, :, :]
                for depth in range(values.shape[1]):
                    channel = variable if variable == "zos" else f"{variable}_{depth}"
                    mask = truth[f"mask_{depth}"].values.astype(bool)
                    weights = np.cos(np.deg2rad(truth.y.values))[:, None] * mask
                    weights = weights / weights.sum()
                    target = ref[channel].values.astype(np.float64)
                    generated = values[:, depth].astype(np.float64)
                    assert np.isfinite(target[:, mask]).all(), channel
                    assert np.isfinite(generated[:, mask]).all(), channel
                    diff = np.where(mask, generated - target, 0)
                    rmse = np.sqrt(np.sum(diff**2 * weights, axis=(-2, -1)))
                    bias = np.sum(diff * weights, axis=(-2, -1))
                    scale = float(stds[channel].values)
                    assert scale > 0 and np.isfinite(scale)
                    for j, pos in enumerate(POSITIONS):
                        rows.append(
                            dict(
                                selection=label,
                                arm=arm,
                                origin=origin,
                                channel=channel,
                                nominal_days=(pos + 1) * 5,
                                actual_days=record["lead_days"][pos],
                                rmse=rmse[j],
                                bias=bias[j],
                                normalized_rmse=rmse[j] / scale,
                                checkpoint_sha256=record["checkpoint_sha256"],
                            )
                        )
            pred.close()
        print("SCORED", label, origin, flush=True)
    with (out / "lead_metrics.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (out / "COMPLETE.json").write_text(
        json.dumps(
            dict(selection=label, rows=len(rows), origins=28, positions=POSITIONS)
        )
    )


if __name__ == "__main__":
    main()
