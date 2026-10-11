# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Single-origin native OM4 field distributions; not observation rollout scores.

Members at different leads are independent decoder draws, not trajectories.
Quantiles use interpolation against cumulative cosine-latitude wet-cell weights.
Compare each forecast with the same-lead truth to account for seasonal changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = [
    "parent",
    "control",
    "replay",
    "replay-bound",
    "pushforward",
    "curvature",
    "block",
    "curvature-replay",
    "block-replay",
]
CHANNELS = ["thetao_0", "zos", "thetao_9", "so_9"]
LEADS = [0, 30, 90, 180, 365]
SUMMARY_COLUMNS = [
    "mean",
    "spatial_std",
    "minimum",
    "maximum",
    "q001",
    "q999",
    "mean_bias",
    "rmse",
    "std_over_reference",
    "area_outside_reference_minmax",
]


def stats(a: np.ndarray, ref: np.ndarray, w: np.ndarray) -> dict[str, float]:
    """Cosine-area statistics and same-lead reference comparisons."""
    mean = np.dot(a, w)
    refmean = np.dot(ref, w)
    std = np.sqrt(np.dot((a - mean) ** 2, w))
    refstd = np.sqrt(np.dot((ref - refmean) ** 2, w))
    if not np.isfinite(refstd) or refstd <= 0:
        raise ValueError("Reference must have positive spatial variance")
    order = np.argsort(a)
    cum = np.cumsum(w[order])
    q = np.interp([0.001, 0.01, 0.5, 0.99, 0.999], cum, a[order])
    return dict(
        mean=float(mean),
        spatial_std=float(std),
        minimum=float(a.min()),
        maximum=float(a.max()),
        q001=float(q[0]),
        q01=float(q[1]),
        median=float(q[2]),
        q99=float(q[3]),
        q999=float(q[4]),
        mean_bias=float(mean - refmean),
        rmse=float(np.sqrt(np.dot((a - ref) ** 2, w))),
        std_over_reference=float(std / refstd),
        area_outside_reference_minmax=float(w[(a < ref.min()) | (a > ref.max())].sum()),
    )


def collect(root: Path, arms: list[str], year: int) -> tuple[list[dict], dict]:
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("Require distinct, nonempty arms")
    rows: list[dict] = []
    provenance: dict = {}
    anchor: dict = {}
    for arm_index, arm in enumerate(arms):
        path = root / arm / "native" / f"fields-{year}.npz"
        complete_path = path.parent / "COMPLETE.json"
        complete = json.loads(complete_path.read_text())
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if complete["files"][path.name] != digest:
            raise ValueError(f"NPZ checksum mismatch: {path}")
        if complete["members"] != 8 or complete["sampling_steps"] != 32:
            raise ValueError(f"Expected eight members and 32 sampling steps: {arm}")
        if complete["leads_days"] != LEADS:
            raise ValueError(f"Unexpected completion leads: {arm}")
        provenance[arm] = dict(
            path=str(path),
            sha256=digest,
            bytes=len(payload),
            complete_sha256=hashlib.sha256(complete_path.read_bytes()).hexdigest(),
            checkpoint_sha256=complete["checkpoint_sha256"],
        )
        with np.load(path, allow_pickle=False) as data:
            shared = {
                key: data[key].copy()
                for key in (
                    "lat",
                    "lon",
                    "mask",
                    "reference",
                    "leads_days",
                    "channels",
                )
            }
            members = data["members"].astype(np.float64)
        expected_shapes = {
            "lat": (180,),
            "lon": (360,),
            "mask": (4, 180, 360),
            "reference": (5, 4, 180, 360),
            "leads_days": (5,),
            "channels": (4,),
        }
        for key, shape in expected_shapes.items():
            if shared[key].shape != shape:
                raise ValueError(f"Unexpected {key} shape for {arm}")
        if (
            shared["channels"].tolist() != CHANNELS
            or shared["leads_days"].tolist() != LEADS
        ):
            raise ValueError(f"Unexpected channels or leads for {arm}")
        if shared["mask"].dtype != np.bool_:
            raise ValueError(f"Expected Boolean wet masks: {arm}")
        if members.shape != (8, 5, 4, 180, 360) or not np.isfinite(members).all():
            raise ValueError(f"Invalid members: {arm}")
        for key in ("lat", "lon", "reference"):
            if not np.isfinite(shared[key]).all():
                raise ValueError(f"Nonfinite {key}: {arm}")
        if not anchor:
            anchor = shared
        for key in shared:
            if not np.array_equal(shared[key], anchor[key]):
                raise ValueError(f"Arms differ in {key}: {arm}")
        area = np.broadcast_to(np.cos(np.deg2rad(shared["lat"]))[:, None], (180, 360))
        if not (np.abs(shared["lat"]) < 90).all() or not (area > 0).all():
            raise ValueError(f"Latitude weights must be positive: {arm}")
        for li, lead in enumerate(LEADS):
            for ci, channel in enumerate(CHANNELS):
                mask = shared["mask"][ci]
                w = area[mask]
                if w.size == 0 or not np.isfinite(w.sum()) or w.sum() <= 0:
                    raise ValueError(f"Invalid wet-cell weights: {arm}/{channel}")
                w = w / w.sum()
                ref = shared["reference"][li, ci][mask].astype(np.float64)
                if arm_index == 0:
                    rows.append(
                        dict(
                            arm="reference",
                            lead_days=lead,
                            channel=channel,
                            kind="reference",
                            member="",
                            **stats(ref, ref, w),
                        )
                    )
                mean = members[:, li, ci].mean(axis=0)[mask]
                rows.append(
                    dict(
                        arm=arm,
                        lead_days=lead,
                        channel=channel,
                        kind="ensemble_mean",
                        member="",
                        **stats(mean, ref, w),
                    )
                )
                for mi in range(8):
                    rows.append(
                        dict(
                            arm=arm,
                            lead_days=lead,
                            channel=channel,
                            kind="member",
                            member=mi,
                            **stats(members[mi, li, ci][mask], ref, w),
                        )
                    )
    metadata = dict(
        origin=f"{year}-01-03 12:00:00",
        initialization="OM4 surface initialized; native OM4 truth",
        leads_days=LEADS,
        channels=CHANNELS,
        ensemble_members=8,
        weights="cos(latitude) normalized over each depth-specific wet mask; approximate gridcell area",
        units={
            "thetao_0": "degC",
            "zos": "m",
            "thetao_9": "degC",
            "so_9": "practical salinity",
        },
        verification=(
            "NPZ hashes match native COMPLETE.json. All arms have identical coordinates, "
            "masks, references, channels and leads; finite fields and positive wet-cell "
            "weights; members shape 8x5x4x180x360."
        ),
        limitations=[
            "One OM4-native origin only; these are not observation-initialized annual metrics.",
            "Member indices at different leads are separate decoder draws, not coherent stochastic trajectories.",
            "Cos(latitude) is approximate area weighting on this latitude/longitude grid.",
            "Compare against same-lead reference; seasonal changes should not be treated as instability.",
            "No deterministic native field artifact included; cannot directly compare distributional growth with deterministic.",
        ],
        artifacts=provenance,
    )
    return rows, metadata


def summarize(rows: list[dict], arms: list[str]) -> list[dict]:
    summary = []
    for arm in arms:
        for lead in LEADS:
            for channel in CHANNELS:
                selected = [
                    r
                    for r in rows
                    if (
                        r["arm"] == arm
                        and r["lead_days"] == lead
                        and r["channel"] == channel
                    )
                ]
                ensemble = next(r for r in selected if r["kind"] == "ensemble_mean")
                members = [r for r in selected if r["kind"] == "member"]
                summary.append(
                    dict(
                        arm=arm,
                        lead_days=lead,
                        channel=channel,
                        ensemble=ensemble,
                        member_average={
                            key: float(np.mean([r[key] for r in members]))
                            for key in SUMMARY_COLUMNS
                        },
                        member_minimum=float(min(r["minimum"] for r in members)),
                        member_maximum=float(max(r["maximum"] for r in members)),
                        member_mean_range=[
                            float(min(r["mean"] for r in members)),
                            float(max(r["mean"] for r in members)),
                        ],
                    )
                )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Evaluation root containing ARM/native"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=ARMS)
    parser.add_argument("--year", type=int, default=2015)
    args = parser.parse_args()
    rows, provenance = collect(args.root, args.arms, args.year)
    summary = summarize(rows, args.arms)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "field-statistics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name, value in (("provenance", provenance), ("summary", summary)):
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n"
        )
    print(
        f"Wrote {len(rows)} field-statistics rows and {len(summary)} summaries to {args.output}"
    )


if __name__ == "__main__":
    main()
