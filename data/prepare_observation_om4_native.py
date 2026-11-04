# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Pack all OM4 frames used by observation training into the shared Rust format."""

import argparse
from pathlib import Path

import numpy as np

from samudra.observations.model_data import ModelMonths
from samudra.observations.native_om4 import prepare_native_om4
from samudra.rust_reader import RustIoRuntime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("om4", type=Path)
    parser.add_argument("grid", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--readers", type=int, default=8)
    args = parser.parse_args()
    with np.load(args.grid) as archive:
        reference = dict(archive)
    original = ModelMonths(args.om4 / "OM4.zarr", "om4", reference, args.om4)
    prepare_native_om4(
        args.om4 / "OM4.zarr", original, args.output, RustIoRuntime(args.readers)
    )
