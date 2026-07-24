# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import xarray as xr
from einops import rearrange

from samudra.constants import TensorMap
from samudra.utils.data import Normalize, stack_levels
from samudra.utils.output import ModelInferenceOutput


class ZarrWriter:
    """Writes model prediction outputs to Zarr format for downstream analysis."""

    def __init__(
        self,
        output_dir: str | os.PathLike,
        coords: dict[str, xr.DataArray],
        hist: int,
        model_path: str | os.PathLike,
        time_chunk_size: int,
        normalize: Normalize,
        tensor_map: TensorMap,
    ):
        self.pred_path = os.path.join(output_dir, "predictions.zarr")

        if os.path.exists(self.pred_path):
            raise FileExistsError(
                f"Predictions already exist at {self.pred_path}. Please choose a unique experiment name, output directory, or delete the existing predictions."
            )

        self.hist = hist
        self.buffer: torch.Tensor | None = None
        self.time_buffer: xr.DataArray | None = None
        self.coords = coords
        self.model_path = model_path
        self.time_chunk_size = time_chunk_size

        self.normalize = normalize
        self.tensor_map = tensor_map

    def record_batch(self, IO: ModelInferenceOutput):
        pred_tensor = IO.prediction
        pred_time = IO.time
        pred_tensor = rearrange(
            pred_tensor, "n (hi c) h w -> (n hi) c h w", hi=self.hist + 1
        )
        pred_tensor = self.normalize.unnormalize_tensor_prognostic(
            pred_tensor, fill_value=0.0
        )
        if self.buffer is None:
            self.buffer = pred_tensor
        else:
            self.buffer = torch.cat([self.buffer, pred_tensor], dim=0)

        if self.time_buffer is None:
            self.time_buffer = pred_time
        else:
            self.time_buffer = xr.concat([self.time_buffer, pred_time], dim="time")

    def write(self):
        # Write to zarr
        if self.buffer is None:
            raise ValueError("No tensor to write")

        if self.time_buffer is None:
            raise ValueError("No time buffer to write")

        buffer = self.buffer.cpu().numpy()  # (time, channel, y, x)

        # Lay the flat per-level channels out as a Dataset, then reassemble into
        # depth-stacked variables with `stack_levels` -- the same helper used to
        # put ground-truth inputs into analysis-ready form, so predictions match.
        coords = self._output_coords()
        coords["time"] = self.time_buffer
        per_level = xr.Dataset(
            {
                name: (["time", "y", "x"], buffer[:, channel, :, :])
                for channel, name in enumerate(self.tensor_map.prognostic_var_names)
            },
            coords=coords,
        )
        ds = stack_levels(per_level, self.tensor_map.data_layout)
        ds = ds.transpose("time", "lev", "y", "x", ...)
        ds.attrs["model_path"] = str(self.model_path)
        ds = ds.chunk({"time": self.time_chunk_size})
        if os.path.exists(self.pred_path):
            ds.to_zarr(self.pred_path, mode="a", append_dim="time")
        else:
            ds.to_zarr(
                self.pred_path,
                mode="w",
                encoding={var: {"compressor": None} for var in ds.data_vars},
            )

        # Reset
        self.buffer = None
        self.time_buffer = None

    def _output_coords(self) -> dict:
        """Build output coordinates that match the ground-truth OM4 dataset layout.

        "Ground-truth layout" means the on-disk shape of the preprocessed OM4 data
        (the same data the model is scored against), so that predictions and truth
        can be compared without any reshaping. Concretely that layout uses ``y``/``x``
        horizontal dims with 2D ``lat``/``lon`` coordinates on top of them (not 1D
        ``lat``/``lon`` dims), carries the depth axis ``lev``, and keeps the grid
        metadata needed for physical analysis (``areacello``, ``dz``,
        ``ocean_fraction``, cell bounds ``lat_b``/``lon_b``).
        """
        src = self.coords
        y_vals = np.asarray(src["lat"].values)  # 1-D latitude axis
        x_vals = np.asarray(src["lon"].values)  # 1-D longitude axis
        ny, nx = y_vals.size, x_vals.size

        spec = self.tensor_map.data_layout
        n_levels = spec.num_prognostic_depth_levels

        # 2-D lat/lon on the (y, x) grid, matching the ground-truth layout. Prefer
        # the real coordinates the source preserved (`lat_2d`/`lon_2d`), which are
        # correct on any grid. When they are absent we may only reconstruct them by
        # broadcasting the 1-D axes on a "gaussian" (rectilinear) grid, where that is
        # exact. On a curvilinear grid (e.g. tripolar) broadcasting fabricates wrong
        # geometry, so we refuse rather than silently emit bad coordinates.
        if "lat_2d" in src and "lon_2d" in src:
            lat2d = np.asarray(src["lat_2d"].values)
            lon2d = np.asarray(src["lon_2d"].values)
        elif spec.grid_type == "gaussian":
            lat2d = np.broadcast_to(y_vals[:, None], (ny, nx)).copy()
            lon2d = np.broadcast_to(x_vals[None, :], (ny, nx)).copy()
        else:
            raise ValueError(
                f"Cannot build 2-D lat/lon for grid_type={spec.grid_type!r}: the "
                "source coords carry no real 'lat_2d'/'lon_2d', and broadcasting the "
                "1-D axes is only valid on a 'gaussian' (rectilinear) grid. Preserve "
                "the true 2-D coordinates through preprocessing for curvilinear grids."
            )

        coords: dict[str, tuple] = {
            "y": ("y", y_vals),
            "x": ("x", x_vals),
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
            "lev": ("lev", np.array(spec.depth_levels[:n_levels])),
        }

        rename = {"lat": "y", "lon": "x"}
        for name, da in src.items():
            if name in ("time", "lat", "lon", "lat_2d", "lon_2d", "lev", "y", "x"):
                continue
            # Keep depth-resolved metadata (dz, ocean_fraction) consistent with the
            # emitted levels. A shallow model spec (e.g. thermo_dynamic_5) predicts
            # fewer levels than the source's full-depth grid coords carry, so slice
            # those to the emitted `lev` count instead of conflicting with it.
            if "lev" in da.dims:
                da = da.isel(lev=slice(0, n_levels))
            dims = tuple(rename.get(str(d), str(d)) for d in da.dims)
            coords[name] = (dims, np.asarray(da.values))

        return coords
