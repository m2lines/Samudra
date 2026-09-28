# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify and aggregate native latent forecast errors, velocities and time increments."""

import argparse
import json
from pathlib import Path

import numpy as np

from samudra.experiments.diffusion_scratch_summary import summarize_structure
from samudra.experiments.diffusion_summary import reduce_records
from samudra.experiments.observation_pilot import atomic_json, digest


def summarize(root):
    protocol = json.loads((root / "protocol.json").read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    if any(complete.get(k) != v for k, v in protocol.items()):
        raise ValueError("Native latent completion contract differs")
    files = complete["files"]
    if len(files) != 24 or len(set(protocol["origins"])) != 24:
        raise ValueError("Require the complete native 24-origin cohort")
    for name, expected in files.items():
        if Path(name).name != name or digest(root / name) != expected:
            raise ValueError("Native latent diagnostic hash differs")
    records = [json.loads((root / name).read_text()) for name in files]
    if sorted(r["origin"] for r in records) != sorted(protocol["origins"]):
        raise ValueError("Native cohort differs")
    results = []
    for lead in (0, 5, 15, 30):
        entries = [dict(statistics=r["velocity"][str(lead)]) for r in records]
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
                region="scored_latitudes",
                lead_days=lead,
                ensemble=reduce_records(entries, "ensemble"),
                zero_velocity=reduce_records(entries, "zero_velocity"),
                structure=summarize_structure(structure),
            )
        )
    errors = {}
    for key in ("normalized_mse", "physical_mse", "persistence_normalized_mse"):
        values = np.asarray([r[key] for r in records], dtype=float)
        if values.shape != (24, 1, 7, 77) or not np.isfinite(values).all():
            raise ValueError("Incomplete native channel/lead errors")
        errors[key] = values.mean((0, 1)).tolist()
    temporal = {
        key: np.asarray([r["temporal"][key] for r in records], dtype=float)
        .mean(0)
        .tolist()
        for key in records[0]["temporal"]
    }
    return dict(
        protocol=protocol,
        channels=protocol["channels"][:38],
        results=results,
        errors=dict(channels=protocol["channels"], leads=protocol["leads"], **errors),
        temporal=temporal,
        scope="OM4 reference, 24 origins, 60S-60N. Static support permits equal-origin averaging of per-channel MSE. Velocity scores pool area-weighted numerators; temporal correlations are descriptive equal-origin means. Independent latent readouts do not represent coherent uncertain trajectories.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    atomic_json(summarize(args.root), args.output)


if __name__ == "__main__":
    main()
