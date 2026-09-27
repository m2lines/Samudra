# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify and summarize native OM4 velocity calibration and spatial structure."""

import argparse
import json
from pathlib import Path
from typing import Any

from samudra.experiments.diffusion_scratch_summary import summarize_structure
from samudra.experiments.diffusion_summary import reduce_records
from samudra.experiments.observation_pilot import atomic_json, digest


def summarize(root):
    protocol = json.loads((root / "protocol.json").read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    if any(complete.get(key) != value for key, value in protocol.items()):
        raise ValueError("Completion contract differs")
    hashes = complete["velocity_sha256"]
    if len(hashes) != 24 or len(set(protocol["origins"])) != 24:
        raise ValueError("Require the complete 24-origin cohort")
    for name, expected in {**hashes, "metrics.csv": complete["metrics_sha256"]}.items():
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError("Native diagnostic hash differs")
    records = [json.loads((root / name).read_text()) for name in hashes]
    if sorted(r["origin"] for r in records) != sorted(protocol["origins"]):
        raise ValueError("Native diagnostic origins differ")
    channels = records[0]["channels"]
    keys = {
        (r, lead)
        for r in ("global", "scored_latitudes", "outside_scored_latitudes")
        for lead in (0, 5, 15, 30)
    }
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {key: [] for key in keys}
    for record in records:
        if record["channels"] != channels or record["units"] != "m/s":
            raise ValueError("Native velocity channel or unit mismatch")
        entries = record["records"]
        if (
            len(entries) != len(keys)
            or {(e["region"], e["lead_days"]) for e in entries} != keys
        ):
            raise ValueError("Incomplete native regions or leads")
        for entry in entries:
            grouped[entry["region"], entry["lead_days"]].append(entry)
    results = []
    for (region, lead), entries in sorted(grouped.items()):
        structure = [
            dict(
                statistics={
                    k: v for k, v in e["statistics"].items() if k.endswith("_structure")
                }
            )
            for e in entries
        ]
        results.append(
            dict(
                region=region,
                lead_days=lead,
                ensemble=reduce_records(entries, "ensemble"),
                zero_velocity=reduce_records(entries, "zero_velocity"),
                structure=summarize_structure(structure),
            )
        )
    return dict(
        protocol=protocol,
        channels=channels,
        results=results,
        scope="Native OM4 reference, not observed velocity skill. Equal supported-channel averages after pooling wet-area sums. Structure describes individual fields; it does not establish dynamical plausibility.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(summarize(args.root), args.output)


if __name__ == "__main__":
    main()
