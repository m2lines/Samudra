# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-date artifact and temporal diagnostics from already generated physical maps."""

import argparse
import json
from pathlib import Path

import numpy as np

CHANNELS = ("so_9", "thetao_9", "uo_9")
ORIGINS = ("2015-01", "2018-01", "2021-01")


def moments(value, weight):
    """Area-weighted second moment, preserving all leading dimensions."""
    return (np.where(weight > 0, value, 0) * weight).sum((-2, -1)) / weight.sum(
        (-2, -1)
    )


def temporal(fields, weight):
    """Member,batch-free time,channel,y,x; samples remain in stored member order."""
    anomaly = fields - fields.mean(0, keepdims=True)
    delta = fields[:, 1:] - fields[:, :-1]
    mean_delta = fields.mean(0)[1:] - fields.mean(0)[:-1]
    # Pool member cross-products before normalization: no near-zero member divisions.
    a, b = anomaly[:, :-1], anomaly[:, 1:]
    cross = moments(a * b, weight).mean(0)
    denominator = np.sqrt(moments(a**2, weight).mean(0) * moments(b**2, weight).mean(0))
    return dict(
        member_increment_mse=moments(delta**2, weight).mean(0).tolist(),
        mean_increment_mse=moments(mean_delta**2, weight).tolist(),
        member_anomaly_correlation=np.divide(
            cross, denominator, out=np.full_like(cross, np.nan), where=denominator > 0
        ).tolist(),
    )


def map_record(path):
    with np.load(path) as z:
        indices = [list(z["channel_names"]).index(c) for c in CHANNELS]
        initial = z["initial"][:, 0, -1, indices].astype(np.float64)
        future = z["states_at_leads"][:, 0][:, :, indices].astype(np.float64)
        lat = z["lat"]
        mask = z["mask"][indices].astype(bool)
        np.testing.assert_array_equal(z["leads_days"], [5, 15, 30])
    weight = (
        mask
        * np.cos(np.deg2rad(lat))[None, :, None]
        * (np.abs(lat)[None, :, None] <= 60)
    )
    if (
        not np.isfinite(initial[:, weight > 0]).all()
        or not np.isfinite(future[:, :, weight > 0]).all()
    ):
        raise ValueError("Nonfinite field on fixed scored support")
    anomaly = initial - initial.mean(0, keepdims=True)
    # The same index crop used in the published diagonal-artifact investigation.
    patch = anomaly[:, 0, 70:100, 180:240]
    patch_mask = mask[0, 70:100, 180:240]
    diag = []
    for sign in (1, -1):
        left = patch[:, :-1, :-1] if sign == 1 else patch[:, :-1, 1:]
        right = patch[:, 1:, 1:] if sign == 1 else patch[:, 1:, :-1]
        support = (
            patch_mask[:-1, :-1] & patch_mask[1:, 1:]
            if sign == 1
            else patch_mask[:-1, 1:] & patch_mask[1:, :-1]
        )
        # All members contribute equally; covariance is pooled before normalization.
        a = left[:, support].ravel()
        b = right[:, support].ravel()
        diag.append(float(np.corrcoef(a, b)[0, 1]))
    return dict(
        origin=path.stem,
        channels=list(CHANNELS),
        members=len(initial),
        initialization_member_deviation_rms=np.sqrt(
            moments(anomaly**2, weight).mean(0)
        ).tolist(),
        salinity_patch_member_deviation_rms=float(
            np.sqrt((patch[:, patch_mask] ** 2).mean())
        ),
        salinity_patch_diagonal_correlations=diag,
        future=temporal(future, weight),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.maps / "protocol.json").read_text())
    marker = (
        "PRETRAINING_MAPS_COMPLETE.json"
        if protocol.get("scope") == "pretraining-maps"
        else "MONTHLY_REPORT_COMPLETE.json"
    )
    complete = json.loads((args.maps / marker).read_text())
    if any(complete.get(k) != v for k, v in protocol.items()):
        raise ValueError("Map completion contract differs")
    records = [
        map_record(args.maps / "members" / (origin + ".npz")) for origin in ORIGINS
    ]
    result = dict(
        checkpoint_sha256=protocol["checkpoint_sha256"],
        scope="Three fixed examples, not a population calibration estimate",
        temporal_intervals_days=[[5, 15], [15, 30]],
        caution="Member identity is propagated in physical diffusion but arbitrary across independently decoded latent readouts; correlations describe that difference, not validated uncertainty",
        support="Existing per-channel wet mask within 60S-60N; cosine latitude weights. Diagonal diagnostic uses published 70:100,180:240 tropical crop.",
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
