# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

import torch
import xarray as xr
from einops import rearrange

from samudra.constants import TensorMap
from samudra.utils.data import Normalize
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

        coords: dict[str, Any] = {k: v for k, v in self.coords.items()}
        coords["time"] = self.time_buffer
        ds = xr.Dataset(
            data_vars={
                var: (["time", "lat", "lon"], self.buffer[:, i, :, :].cpu().numpy())
                for i, var in enumerate(self.tensor_map.prognostic_var_names)
            },
            coords=coords,
        )
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
