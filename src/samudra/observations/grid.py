# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export grid geometry and wet masks without constructing a training config."""

from pathlib import Path

import numpy as np
import xarray as xr
import zarr  # type: ignore[import-untyped]

from samudra.observations.constants import (
    MODEL_DEPTHS,
    MODEL_VARIABLES,
    STATE_CHANNELS,
    STATE_NAMES,
)
from samudra.observations.prepare import save_atomic


def export_grid(
    om4: str | Path, output: str | Path, *, geometry: str | Path | None = None
) -> None:
    # The separate data environment supplies this preprocessing package.
    from ocean_preprocessing import vertical_grid  # type: ignore[import-not-found]

    store = zarr.open_consolidated(str(om4), mode="r")
    depth = store["lev"][:] if "lev" in store else vertical_grid.OM4_DEPTH_LEVELS
    np.testing.assert_allclose(depth, vertical_grid.OM4_DEPTH_LEVELS, rtol=0, atol=1e-6)
    if geometry is None:
        if "lat_b" not in store or "lon_b" not in store:
            raise ValueError(
                "OM4 store lacks cell bounds; pass --grid-geometry with the target grid used during OM4 preparation"
            )
        lat_b, lon_b = store["lat_b"][:], store["lon_b"][:]
    else:
        with xr.open_dataset(geometry) as coordinates:
            lat = coordinates.lat.values
            lon = coordinates.lon.values
            if lat.ndim == 2:
                np.testing.assert_allclose(
                    lat, np.broadcast_to(store["y"][:][:, None], lat.shape)
                )
                np.testing.assert_allclose(
                    lon, np.broadcast_to(store["x"][:][None, :], lon.shape)
                )
            else:
                np.testing.assert_allclose(lat, store["y"][:])
                np.testing.assert_allclose(lon, store["x"][:])
            lat_b, lon_b = coordinates.lat_b.values, coordinates.lon_b.values
    mask = np.stack(
        [store[f"mask_{i}"][:] for i in range(MODEL_DEPTHS)] * len(MODEL_VARIABLES)
        + [store["mask_0"][:]]
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
        names=np.array(STATE_NAMES),
        lat_b=lat_b,
        lon_b=lon_b,
        depth=np.asarray(depth),
    )
