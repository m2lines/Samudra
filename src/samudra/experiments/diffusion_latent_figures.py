# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot the verified three-system comparison without pooling independent seeds."""

import argparse
import gzip
import json
from pathlib import Path

from samudra.experiments.diffusion_scratch_figures import annual, profiles

STYLES = (
    ("baseline", "#444444", "^", "Deterministic baseline"),
    ("B-1729", "#286cb0", "o", "Physical diffusion 1729"),
    ("B-1730", "#286cb0", "s", "Physical diffusion 1730"),
    ("D-1729", "#c84638", "D", "Latent + diffusion readout 1729"),
    ("D-1730", "#c84638", "X", "Latent + diffusion readout 1730"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with gzip.open(args.summary, "rt") as stream:
        summary = json.load(stream)
    runs = {}
    for family in ("physical", "latent"):
        runs.update(summary["families"][family])
    expected = {name for name, *_ in STYLES if name != "baseline"}
    if set(runs) != expected:
        raise ValueError("Expected both completed seeds from both model families")
    shared = {"baseline": summary["baseline"], "runs": runs}
    args.output.mkdir(parents=True, exist_ok=True)
    profiles(shared, args.output, styles=STYLES)
    annual(shared, args.output, styles=STYLES)


if __name__ == "__main__":
    main()
