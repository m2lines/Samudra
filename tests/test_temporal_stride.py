import numpy as np
import torch
import xarray as xr

from ocean_emulators.datasets import TorchTrainDataset
from ocean_emulators.utils.data import DataSource, Masks


def test_torch_train_dataset_temporal_stride_subsamples_windows():
    coords = {"time": range(10), "lat": range(1), "lon": range(1)}
    data = xr.Dataset(
        {
            "prognostic1": xr.DataArray(
                np.arange(10, dtype=np.float32).reshape(10, 1, 1),
                dims=["time", "lat", "lon"],
                coords=coords,
            ),
            "boundary1": xr.DataArray(
                (100 + np.arange(10, dtype=np.float32)).reshape(10, 1, 1),
                dims=["time", "lat", "lon"],
                coords=coords,
            ),
        }
    )
    means = xr.Dataset({"prognostic1": 0.0, "boundary1": 0.0})
    stds = xr.Dataset({"prognostic1": 1.0, "boundary1": 1.0})
    masks = Masks(
        prognostic=torch.ones((1, 1, 1), dtype=torch.bool),
        boundary=torch.ones((1, 1), dtype=torch.bool),
    )
    source = DataSource("test", data, means, stds, masks=masks)

    dataset = TorchTrainDataset(
        src=source,
        dst=None,
        prognostic_var_names=["prognostic1"],
        boundary_var_names=["boundary1"],
        hist=1,
        steps=2,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
        temporal_stride=2,
    )

    assert len(dataset) == 3
    assert dataset[0].raw_data[0][0][:, 0, 0, 0].tolist() == [0.0, 1.0]
    assert dataset[1].raw_data[0][0][:, 0, 0, 0].tolist() == [2.0, 3.0]
    assert dataset[2].raw_data[0][0][:, 0, 0, 0].tolist() == [4.0, 5.0]
