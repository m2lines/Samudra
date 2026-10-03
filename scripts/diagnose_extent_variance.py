#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-only variance decomposition of the accessory target; no model fit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def diagnose(root):
    ready = json.loads((root / "READY.json").read_text())
    for name in ("grid", "normalization", "manifest"):
        path = root / (name + (".npz" if name == "grid" else ".json"))
        if digest(path) != ready[name + "_sha256"]:
            raise ValueError("Target provenance changed")
    grid = np.load(root / "grid.npz")
    norm = json.loads((root / "normalization.json").read_text())
    store = zarr.open_consolidated(str(root / "targets.zarr"))
    if not store["verified"][:].all():
        raise ValueError("Unverified target frames")
    mask = grid["mask"]
    weights = grid["area"][mask].astype("f8")
    weights /= weights.sum()
    dates = grid["dates"].tolist()
    means = np.zeros((12, mask.sum()), dtype="f8")
    counts = np.zeros(12, dtype=int)
    for i, date in enumerate(dates):
        values = store["variance"][i][mask].astype("f8")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite wet target")
        transformed = (
            np.log1p(values / norm["variance_scale"]) - norm["log_mean"]
        ) / norm["log_std"]
        month = int(date[5:7]) - 1
        means[month] += transformed
        counts[month] += 1
    if not counts.all():
        raise ValueError("Missing calendar-month target coverage")
    spatial_mean = means.sum(0) / counts.sum()
    means /= counts[:, None]
    loss = dict(zero=0.0, static_spatial_mean=0.0, calendar_month_spatial_mean=0.0)
    for i, date in enumerate(dates):
        values = store["variance"][i][mask].astype("f8")
        transformed = (
            np.log1p(values / norm["variance_scale"]) - norm["log_mean"]
        ) / norm["log_std"]
        month = int(date[5:7]) - 1
        loss["zero"] += float((transformed**2 * weights).sum())
        loss["static_spatial_mean"] += float(
            ((transformed - spatial_mean) ** 2 * weights).sum()
        )
        loss["calendar_month_spatial_mean"] += float(
            ((transformed - means[month]) ** 2 * weights).sum()
        )
    loss = {key: value / len(dates) for key, value in loss.items()}
    if not np.isclose(loss["zero"], 1.0, rtol=1e-5):
        raise ValueError(
            "Training target normalization does not reproduce unit variance"
        )
    return dict(
        scope="Training-only target decomposition; fixed maps fitted and measured on the same period. Not held-out predictive skill.",
        interpretation="Fraction of standardized target MSE removed by geography alone or geography plus calendar month; does not demonstrate dynamics or observation benefit.",
        ready_sha256=digest(root / "READY.json"),
        script_sha256=digest(Path(__file__)),
        frames=len(dates),
        period=[dates[0], dates[-1]],
        monthly_counts=counts.tolist(),
        weighted_mse=loss,
        fraction_explained={
            key: 1 - value / loss["zero"]
            for key, value in loss.items()
            if key != "zero"
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(diagnose(args.root), stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
