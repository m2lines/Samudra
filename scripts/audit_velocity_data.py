#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Read-only, bounded CPU audit of prepared velocity campaign datasets."""

import argparse
import json
from pathlib import Path

import numpy as np

from samudra.experiments.velocity_transfer.data import VelocitySource, eligible_anchors


def audit(root):
    from samudra_rust_loader import FlatOm4ReadPool

    pool = FlatOm4ReadPool(4)
    result = {}
    for name in ("duacs", "om4_1deg", "om4_quarterdeg"):
        source = VelocitySource(root / name, pool)
        if not (np.diff(source.times) > np.timedelta64(0, "s")).all():
            raise ValueError(f"Non-increasing time coordinate: {name}")
        if not np.isfinite(source.static).all() or not np.isfinite(source.std).all():
            raise ValueError(f"Non-finite normalization or conditioning: {name}")
        if not (source.std > 0).all() or not source.mask.any():
            raise ValueError(f"Invalid normalization or wet mask: {name}")
        if not np.isfinite(source.monthly_mean).all():
            raise ValueError(f"Non-finite monthly climatology: {name}")
        splits, samples = {}, []
        for split in ("train", "validation", "test"):
            anchors = eligible_anchors(source.times, split)
            splits[split] = {
                "windows": len(anchors),
                "first_anchor": str(source.times[anchors[0]]) if len(anchors) else None,
                "last_anchor": str(source.times[anchors[-1]]) if len(anchors) else None,
            }
            for anchor in np.unique(
                anchors[
                    np.linspace(0, len(anchors) - 1, min(3, len(anchors)), dtype=int)
                ]
            ):
                fields = np.empty((1, 2, *source.mask.shape), dtype=np.float32)
                source.reader.read_into([int(anchor)], ["u", "v"], fields)
                valid = np.isfinite(fields[0]).all(axis=0) & source.mask
                speed = np.sqrt((fields[0, :, valid] ** 2).sum(axis=-1))
                if not valid.any() or not np.isfinite(speed).all():
                    raise ValueError(f"Invalid sampled velocities: {name}, {anchor}")
                samples.append(
                    {
                        "split": split,
                        "time": str(source.times[anchor]),
                        "valid_cells": int(valid.sum()),
                        "speed_quantiles_m_per_s": dict(
                            zip(
                                ("p50", "p90", "p99", "p99_9", "maximum"),
                                np.quantile(speed, [0.5, 0.9, 0.99, 0.999, 1]).tolist(),
                                strict=True,
                            )
                        ),
                    }
                )
        if not splits["train"]["windows"]:
            raise ValueError(f"No complete training windows: {name}")
        if name == "duacs":
            if not all(splits[s]["windows"] for s in ("validation", "test")):
                raise ValueError("Missing DUACS held-out windows")
        elif source.times[-1] > np.datetime64("2018-09-30T23:59:59"):
            raise ValueError(f"OM4 extends beyond the training cutoff: {name}")
        result[name] = {
            "shape": [len(source.times), *source.mask.shape],
            "start": str(source.times[0]),
            "end": str(source.times[-1]),
            "standard_deviation_m_per_s": source.std[:, 0, 0].tolist(),
            "training_mask_cells": int(source.mask.sum()),
            "training_mask_area_m2": float(
                source.area[source.mask].sum(dtype=np.float64) * 1e10
            ),
            "splits": splits,
            "samples": samples,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.data_root)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
