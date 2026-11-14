# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Precompute tiny display metrics from saved Xray traces; no model inference."""

import argparse
import json

import numpy as np
from step_anatomy import denoiser_rmse

if __name__ == "__main__":
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    for trace in sorted(args.root.glob("*/*/trace.json")):
        calls, errors = denoiser_rmse(str(trace.parent))
        rows = [[float(v) if np.isfinite(v) else None for v in row] for row in errors]
        destination = trace.parent / "denoiser-rmse.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(dict(calls=calls.tolist(), rmse=rows), allow_nan=False)
        )
        temporary.replace(destination)
        print(trace.parent, flush=True)
