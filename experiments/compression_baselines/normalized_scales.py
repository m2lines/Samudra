# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""How big the true five-day change is in Samudra's normalized units.

    python normalized_scales.py CACHE [STDS_ZARR]

Samudra 2 (pred_residuals: false) predicts the full state, normalized per channel
by OM4_stds. This prints, for selected channels, that normalization spread, the
persistence RMSE over the first 200 steps of the test window (the RMS true
five-day change, on every other step, as Samudra 2 is scored) and their ratio: the size of the signal the
model must resolve, in the units its loss sees.
"""

import json
import sys

import numpy as np
import xarray as xr

STDS = (
    "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/OM4_stds.zarr"
)
CHANNELS = [
    "thetao_0", "thetao_7", "thetao_13", "so_0", "so_7", "so_13",
    "uo_0", "uo_7", "uo_13", "zos",
]  # fmt: skip

C = sys.argv[1]
stds = xr.open_zarr(sys.argv[2] if len(sys.argv) > 2 else STDS)
meta = json.load(open(f"{C}/meta.json"))
st = np.load(f"{C}/static.npz")
area = st["area"]
for c in CHANNELS:
    A = np.load(f"{C}/test/{c}.npy", mmap_mode="r")
    wet = st["masks"][meta["levels"][c]]
    d = np.asarray(A[1:200:2]) - np.asarray(A[0:199:2])
    rp = float(np.sqrt(np.average(d[:, wet] ** 2, axis=1, weights=area[wet]).mean()))
    s = float(stds[c].values)
    print(f"{c:10s} norm std {s:.4g}  persistence RMSE {rp:.4g}  ratio {rp / s:.4g}")
