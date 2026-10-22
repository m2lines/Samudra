# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export grid geometry and wet masks without constructing a training config."""

from pathlib import Path

import numpy as np
import zarr  # type: ignore[import-untyped]

from samudra.observations.constants import MODEL_DEPTHS, STATE_CHANNELS
from samudra.observations.prepare import save_atomic


def export_grid(om4: str | Path, output: str | Path) -> None:
    # The separate data environment supplies this preprocessing package.
    from ocean_preprocessing import vertical_grid  # type: ignore[import-not-found]

    store = zarr.open_consolidated(str(om4), mode="r")
    depth = store["lev"][:] if "lev" in store else vertical_grid.OM4_DEPTH_LEVELS
    np.testing.assert_allclose(depth, vertical_grid.OM4_DEPTH_LEVELS, rtol=0, atol=1e-6)
    names = [
        f"{v}_{i}" for v in ("uo", "vo", "thetao", "so") for i in range(MODEL_DEPTHS)
    ] + ["zos"]
    mask = np.stack(
        [store[f"mask_{i}"][:] for i in range(MODEL_DEPTHS)] * 4 + [store["mask_0"][:]]
    ).astype(bool)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_atomic(
        output,
        lat=store["y"][:],
        lon=store["x"][:],
        mask=mask,
        mean=np.zeros(STATE_CHANNELS, dtype=np.float32),
        std=np.ones(STATE_CHANNELS, dtype=np.float32),
        names=np.array(names),
        depth=np.asarray(depth),
    )
