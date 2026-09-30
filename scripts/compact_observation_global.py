#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Lossless report subsetting and latitude-band diagnostics of global annual exports."""

import argparse
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from collect_observation_global import (  # type: ignore[import-not-found]
    ARMS,
    ORIGINS,
    digest,
)

CHANNELS = [38, 47, 57, 66, 0, 19, 76]


def regional_errors(prediction, reference, mask, lat):
    result: dict[str, Any] = {}
    for region, band in [
        ("global", np.ones_like(lat, bool)),
        ("north", lat > 60),
        ("south", lat < -60),
        ("midlatitudes", np.abs(lat) <= 60),
    ]:
        result[region] = {}
        for channel, name, wet_channel in [(0, "sst", 0), (1, "adt", 6)]:
            support = (
                mask[wet_channel] & band[:, None] & np.isfinite(reference[channel])
            )
            weights = support * np.cos(np.deg2rad(lat))[:, None]
            error = np.where(support, prediction[channel] - reference[channel], 0)
            assert np.isfinite(error).all()
            result[region][name] = dict(
                rmse=float(np.sqrt((weights * error**2).sum() / weights.sum())),
                bias=float((weights * error).sum() / weights.sum()),
                cells=int(support.sum()),
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        run, root, prefix, _ = ARMS[arm]
        destination = args.output / arm
        destination.mkdir(exist_ok=True)
        original_grid = (
            Path(json.loads((run / "manifest.json").read_text())["arguments"]["data"])
            / "grid.npz"
        )
        grid = np.load(original_grid)
        np.savez_compressed(
            destination / "grid.npz",
            lat=grid["lat"],
            lon=grid["lon"],
            mask=grid["mask"][CHANNELS],
            names=grid["names"][CHANNELS],
            channels=CHANNELS,
        )
        record = {
            "extractor_sha256": digest(Path(__file__)),
            "sources": {str(original_grid): digest(original_grid)},
            "regional": {},
            "latent": {},
        }
        for choice in ("endpoint", "selected"):
            directory = root / f"{prefix}-{choice}-annual"
            assert (directory / "COMPLETE.json").exists(), directory
            record["sources"][str(directory / "COMPLETE.json")] = digest(
                directory / "COMPLETE.json"
            )
            for origin in ORIGINS:
                path = directory / (origin + ".npz")
                record["sources"][str(path)] = digest(path)
                with np.load(path) as source:
                    indices = [
                        list(source["leads_days"]).index(day) for day in (30, 365)
                    ]
                    physical = source["state_at_leads"][indices][:, CHANNELS]
                    initial = source["initial"][-1, CHANNELS]
                    assert physical.shape == (2, 7, 180, 360) and initial.shape == (
                        7,
                        180,
                        360,
                    )
                    wet = grid["mask"][CHANNELS]
                    assert (
                        np.isfinite(physical[:, wet]).all()
                        and np.isfinite(initial[wet]).all()
                    )
                    extra = {}
                    if "latent_initial" in source:
                        extra = dict(
                            latent_initial=source["latent_initial"],
                            latent=source["latent_at_leads"][indices],
                        )
                        assert extra["latent_initial"].shape[1] == 10
                        ocean = grid["mask"].any(0)
                        record["latent"][choice + "/" + origin] = {
                            label: {
                                "rms_by_channel": np.sqrt(
                                    (a[:, ocean].astype(np.float64) ** 2).mean(-1)
                                ).tolist(),
                                "max_abs": float(np.abs(a[:, ocean]).max()),
                            }
                            for label, a in [
                                ("initial", extra["latent_initial"][-1]),
                                ("day30", extra["latent"][0]),
                                ("day365", extra["latent"][1]),
                            ]
                        }
                    np.savez_compressed(
                        destination / f"{choice}-{origin}.npz",
                        initial=initial,
                        state=physical,
                        surface=source["surface"][[5, 72]],
                        reference=source["reference"][[5, 72]],
                        leads_days=[30, 365],
                        **extra,
                    )
                    record["regional"][choice + "/" + origin] = {
                        str(day): regional_errors(
                            source["surface"][day // 5 - 1],
                            source["reference"][day // 5 - 1],
                            wet,
                            grid["lat"],
                        )
                        for day in (5, 30, 365)
                    }
        record["files"] = {
            p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
            for p in destination.glob("*.npz")
        }
        (destination / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
        archive_path = args.output / (arm + ".tar")
        with tarfile.open(archive_path, "w") as archive:
            archive.add(destination, arcname=arm)
        receipt = dict(sha256=digest(archive_path), bytes=archive_path.stat().st_size)
        (args.output / (arm + ".json")).write_text(json.dumps(receipt) + "\n")
        print(arm, json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
