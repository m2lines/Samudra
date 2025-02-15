import os
from typing import Dict

import torch
import xarray as xr
from einops import rearrange

from constants import TensorMap
from utils.data import Normalize
from utils.model import InfOutput


class ZarrWriter:
    def __init__(self, output_dir: str, coords: Dict[str, xr.DataArray], hist: int):
        self.pred_path = os.path.join(output_dir, "predictions.zarr")
        self.hist = hist
        self.acc_tensor = None
        self.coords = coords

        self.normalize = Normalize.get_instance()
        self.tensor_map = TensorMap.get_instance()

    def record_batch(self, IO: InfOutput):
        pred_tensor = IO.prediction
        pred_tensor = pred_tensor.squeeze(0)
        pred_tensor = rearrange(pred_tensor, "(n c) h w -> n c h w", n=self.hist + 1)
        pred_tensor = self.normalize.unnormalize_tensor_outputs(pred_tensor)
        if self.acc_tensor is None:
            self.acc_tensor = pred_tensor.cpu()
        else:
            self.acc_tensor = torch.cat([self.acc_tensor, pred_tensor.cpu()], dim=0)

    def write(self):
        # Write to zarr
        if self.acc_tensor is None:
            raise ValueError("No tensor to write")

        coords = {k: v for k, v in self.coords.items()}
        # TODO: Replace by actual time so I dont need to fix this downstream
        coords["time"] = range(self.acc_tensor.shape[0])
        ds = xr.Dataset(
            data_vars={
                var: (["time", "lat", "lon"], self.acc_tensor[:, i, :, :].numpy())
                for i, var in enumerate(self.tensor_map.outputs)
            },
            coords=coords,
        )

        if os.path.exists(self.pred_path):
            ds.to_zarr(self.pred_path, mode="a", append_dim="time")
        else:
            ds.to_zarr(self.pred_path, mode="w")

        # Reset
        self.acc_tensor = None
