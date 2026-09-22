#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Explain cross-wave tail-batch differences without relaxing numeric tolerances."""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from samudra.experiments.surface_adaptation_analysis import read_bytes


def batch_sizes(origins, world, batch):
    sizes = {}
    for rank in range(world):
        shard = origins[rank::world]
        for start in range(0, len(shard), batch):
            chunk = shard[start : start + batch]
            sizes.update(dict.fromkeys(chunk, len(chunk)))
    return sizes


def audit(current, matched, previous, output, current_raw, matched_raw):
    previous = Path(previous)
    old_manifest = json.loads(read_bytes(previous / "manifest.json"))
    old = pd.concat(
        [
            pd.read_csv(
                io.BytesIO(read_bytes(previous / f"heldout-origins-rank{r}.csv"))
            )
            for r in range(old_manifest["world_size"])
        ]
    )
    old = old[old["mode"] == "true"]
    keys = ["region", "origin_time", "lead_days", "variable"]
    fields = ["normalized_mse", "physical_mse", "target_second_moment"]
    origins = sorted(old.origin_time.unique())
    old_sizes = batch_sizes(
        origins, old_manifest["world_size"], old_manifest["arguments"]["batch_size"]
    )
    result = {}
    for label, directory, raw in [
        ("current_two_gpu", current, current_raw),
        ("matched_four_gpu", matched, matched_raw),
    ]:
        manifest = json.loads(read_bytes(Path(raw) / "A" / "manifest.json"))
        world = manifest["world_size"]
        batch = manifest["arguments"]["batch_size"]
        if world != (2 if label == "current_two_gpu" else 4):
            raise ValueError("Unexpected reference-check world size")
        frame = pd.read_csv(io.BytesIO(read_bytes(Path(directory) / "origins.csv")))
        frame = frame[
            (frame.run == "A")
            & (frame["mode"] == "true")
            & (frame.lead_days > 0)
            & (frame.variable != "ts")
        ]
        joined = frame.merge(
            old, on=keys, suffixes=("_new", "_old"), validate="one_to_one"
        )
        if len(joined) != len(frame) or set(joined.origin_time) != set(origins):
            raise ValueError("Reference coverage differs")
        sizes = batch_sizes(origins, world, batch)
        changed = sorted(o for o in origins if sizes[o] != old_sizes[o])
        same = ~joined.origin_time.isin(changed)
        differences = {}
        for field in fields:
            a, b = joined[field + "_new"], joined[field + "_old"]
            np.testing.assert_allclose(a[same], b[same], rtol=2e-5, atol=1e-8)
            bad = ~np.isclose(a, b, rtol=2e-5, atol=1e-8)
            if set(joined.loc[bad, "origin_time"]) - set(changed):
                raise ValueError("Difference outside expected tail batches")
            differences[field] = {
                "mismatched_rows": int(bad.sum()),
                "max_absolute_difference": float((a - b).abs().max()),
            }
        result[label] = {
            "rows": len(joined),
            "world_size": world,
            "batch_size": batch,
            "changed_batch_origins": changed,
            "differences": differences,
        }
    if result["matched_four_gpu"]["changed_batch_origins"]:
        raise ValueError("Matching configuration does not match old batches")
    Path(output).write_text(json.dumps(result, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--current", required=True)
    p.add_argument("--matched", required=True)
    p.add_argument("--previous", required=True)
    p.add_argument("--current-raw", required=True)
    p.add_argument("--matched-raw", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    audit(a.current, a.matched, a.previous, a.output, a.current_raw, a.matched_raw)


if __name__ == "__main__":
    main()
