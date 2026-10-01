#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Plot crossed initializer data/task conditions at one pixel per grid cell."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from plot_observation_missingness import panel_plot  # type: ignore[import-not-found]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.arrays / "COMPLETE.json").read_text())
    for name, info in meta["files"].items():
        assert sha(args.arrays / name) == info["sha256"]
        assert (args.arrays / name).stat().st_size == info["bytes"]
    grid = dict(np.load(args.arrays / "grid.npz"))
    audit = {}
    for origin in meta["cases"]:
        arrays = dict(np.load(args.arrays / (origin + ".npz")))
        panels = [
            (label, arrays[key], grid)
            for key, label in [
                ("om4_om4", "OM4 inputs / OM4 adapter"),
                ("obs_observation", "Obs inputs / observation adapter"),
                ("om4_target", f"OM4 target: {int(origin[:4]) - 1}-12-27 to 12-31"),
                ("om4_observation", "OM4 inputs / observation adapter (swap)"),
                ("obs_om4", "Obs inputs / OM4 adapter (swap)"),
            ]
        ]
        for channel, name in [(0, "zonal"), (1, "meridional")]:
            filename = origin + "-" + name + ".png"
            audit[filename] = panel_plot(
                panels,
                f"Mixed checkpoint: surface {name} velocity before {origin}; frozen weights",
                (channel, name, "m/s", (-0.5, 0.5), "RdBu_r"),
                args.output / filename,
                distinguish_missing=True,
            )
    (args.output / "plot-audit.json").write_text(
        json.dumps(
            {
                "panels": audit,
                "source_sha256": sha(__file__),
                "files": {p.name: sha(p) for p in args.output.glob("*.png")},
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
