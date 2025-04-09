import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.constants import DEPTH_LEVELS, TensorMap
from ocean_emulators.datasets import OM4Dataset, TorchTrainDataset, TrainDataset
from ocean_emulators.utils.data import Normalize, extract_wet_mask, validate_data
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_om4


@pytest.fixture
def data_init():
    with MultitonScope():
        levels = 19
        lats = 3
        lons = 3
        hist = 1
        steps = 4
        stride = 1

        tensor_map = TensorMap.init_instance("thermo_all", "tau_hfds")

        data = xr.Dataset(
            {
                **{
                    f"thetao_{lev}": (
                        ["time", "lat", "lon"],
                        np.random.randn(10, lats, lons) * 10 + 4,
                    )
                    for lev in range(levels)
                },
                **{
                    f"so_{lev}": (
                        ["time", "lat", "lon"],
                        np.random.randn(10, lats, lons) * 30 + 4,
                    )
                    for lev in range(levels)
                },
                "zos": (
                    ["time", "lat", "lon"],
                    np.random.randn(10, lats, lons) * 10 + 4,
                ),
                "tauuo": (
                    ["time", "lat", "lon"],
                    np.random.randn(10, lats, lons) * 10 + 4,
                ),
                "tauvo": (
                    ["time", "lat", "lon"],
                    np.random.randn(10, lats, lons) * 10 + 4,
                ),
                "hfds": (
                    ["time", "lat", "lon"],
                    np.random.randn(10, lats, lons) * 10 + 4,
                ),
                "wetmask": (
                    ["time", "lev", "lat", "lon"],
                    np.random.randint(0, 2, (10, levels, lats, lons)),
                ),
            },
            coords={
                "time": np.arange(10),
                "lev": DEPTH_LEVELS,
                "lat": np.arange(lats),
                "lon": np.arange(lons),
            },
        )
        data_mean = data.mean()
        data_std = data.std()
        data, data_mean, data_std = validate_data(data, data_mean, data_std)
        wet, wet_surface = extract_wet_mask(data, tensor_map.prognostic_var_names, hist)
        wet_without_hist, _ = extract_wet_mask(data, tensor_map.prognostic_var_names, 0)

        _ = Normalize.init_instance(
            data_mean=data_mean,
            data_std=data_std,
            prognostic_var_names=tensor_map.prognostic_var_names,
            boundary_var_names=tensor_map.boundary_var_names,
            wet_mask=wet_without_hist,
        )
        groundtruth_dataset = TrainDataset(
            data,
            tensor_map.prognostic_var_names,
            tensor_map.boundary_var_names,
            wet,
            wet_surface,
            hist,
            steps,
            stride,
        )

        torch_dataset = TorchTrainDataset(
            data,
            tensor_map.prognostic_var_names,
            tensor_map.boundary_var_names,
            wet,
            wet_surface,
            hist,
            steps,
            stride,
        )

        lazy_dataset = OM4Dataset(
            data,
            tensor_map.prognostic_var_names,
            tensor_map.boundary_var_names,
            hist,
            steps,
            stride,
            is_inference=False,
        )

        yield groundtruth_dataset, torch_dataset, lazy_dataset


def test_loaders(data_init):
    groundtruth_dataset, torch_dataset, lazy_dataset = data_init

    # Index 0 test
    gt_td0 = groundtruth_dataset[0]
    torch_td0 = torch_dataset[0]

    input_, label_ = lazy_dataset[0]
    collated_lazy_dataset = collate_om4([(input_, label_)])

    for i in range(len(gt_td0)):
        gt_input = gt_td0.get_input(i)
        torch_input = torch_td0.get_input(i)
        lazy_input = collated_lazy_dataset.get_input(i)
        print(torch.sum(gt_input - torch_input))
        print(torch.sum(gt_input - lazy_input))
        assert torch.allclose(gt_input, torch_input)
        assert torch.allclose(gt_input, lazy_input)

        gt_label = gt_td0.get_label(i)
        torch_label = torch_td0.get_label(i)
        lazy_label = collated_lazy_dataset.get_label(i)
        print(torch.sum(gt_label - torch_label))
        print(torch.sum(gt_label - lazy_label))
        assert torch.allclose(gt_label, torch_label)
        assert torch.allclose(gt_label, lazy_label)

    # TODO: Multiple IndexCollated Test

    # TODO: Parameter-Sweep Test

    # TODO: Inference Test on Lazy Loader
