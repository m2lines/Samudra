# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export grid geometry and wet masks without constructing a training config."""

from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]

from samudra.constants import build_om4_layout
from samudra.observations.prepare import save_atomic


def export_grid(om4, output):
    store = zarr.open_consolidated(str(om4), mode="r")
    layout = build_om4_layout()
    names = [f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(19)] + [
        "zos"
    ]
    mask = np.stack(
        [store[f"mask_{i}"][:] for i in range(19)] * 4 + [store["mask_0"][:]]
    ).astype(bool)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(
        output,
        lat=store["y"][:],
        lon=store["x"][:],
        mask=mask,
        mean=np.zeros(77, dtype=np.float32),
        std=np.ones(77, dtype=np.float32),
        names=np.array(names),
        depth=np.asarray(layout.depth_levels),
    )
