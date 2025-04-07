import os
from typing import Any, Dict

import torch
import xarray as xr
from einops import rearrange

from ocean_emulators.constants import TensorMap
from ocean_emulators.utils.data import Normalize
from ocean_emulators.utils.model import InfOutput


class ZarrWriter:
    def __init__(
        self,
        output_dir: str | os.PathLike,
        coords: Dict[str, xr.DataArray],
        hist: int,
        model_path: str | os.PathLike,
    ):
        self.pred_path = os.path.join(output_dir, "predictions.zarr")
        self.hist = hist
        self.buffer: torch.Tensor | None = None
        self.coords = coords
        self.model_path = model_path

        self.normalize = Normalize.get_instance()
        self.tensor_map = TensorMap.get_instance()

    def record_batch(self, IO: InfOutput):
        pred_tensor = IO.prediction
        pred_tensor = pred_tensor.squeeze(0)
        pred_tensor = rearrange(pred_tensor, "(n c) h w -> n c h w", n=self.hist + 1)
        pred_tensor = self.normalize.unnormalize_tensor_prognostic(
            pred_tensor, fill_value=0.0
        )
        if self.buffer is None:
            self.buffer = pred_tensor
        else:
            self.buffer = torch.cat([self.buffer, pred_tensor], dim=0)

    def write(self):
        # Write to zarr
        if self.buffer is None:
            raise ValueError("No tensor to write")

        coords: Dict[str, Any] = {k: v for k, v in self.coords.items()}
        # TODO: Replace by actual time so I dont need to fix this downstream
        coords["time"] = range(self.buffer.shape[0])
        ds = xr.Dataset(
            data_vars={
                var: (["time", "lat", "lon"], self.buffer[:, i, :, :].cpu().numpy())
                for i, var in enumerate(self.tensor_map.prognostic_var_names)
            },
            coords=coords,
        )
        ds.attrs["model_path"] = str(self.model_path)
        ds = ds.chunk({"time": 1, "lat": 180, "lon": 360})
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
