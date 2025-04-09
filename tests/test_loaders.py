import numpy as np
import pytest
import torch
import xarray as xr

from ocean_emulators.constants import DEPTH_LEVELS, TensorMap
from ocean_emulators.datasets import (
    OM4Dataset,
    TorchTrainDataset,
    TrainData,
    TrainDataset,
)
from ocean_emulators.utils.data import Normalize, extract_wet_mask, validate_data
from ocean_emulators.utils.multiton import MultitonScope
from ocean_emulators.utils.train import collate_om4, collate_train_data


@pytest.fixture
def data_init():
    with MultitonScope():
        levels = 19
        lats = 3
        lons = 3
        hist = 1
        steps = 4
        stride = 1
        total_time_steps = 100

        tensor_map = TensorMap.init_instance("thermo_all", "tau_hfds")

        data = xr.Dataset(
            {
                **{
                    f"thetao_{lev}": (
                        ["time", "lat", "lon"],
                        np.random.randn(total_time_steps, lats, lons) * 10 + 4,
                    )
                    for lev in range(levels)
                },
                **{
                    f"so_{lev}": (
                        ["time", "lat", "lon"],
                        np.random.randn(total_time_steps, lats, lons) * 30 + 4,
                    )
                    for lev in range(levels)
                },
                "zos": (
                    ["time", "lat", "lon"],
                    np.random.randn(total_time_steps, lats, lons) * 10 + 4,
                ),
                "tauuo": (
                    ["time", "lat", "lon"],
                    np.random.randn(total_time_steps, lats, lons) * 10 + 4,
                ),
                "tauvo": (
                    ["time", "lat", "lon"],
                    np.random.randn(total_time_steps, lats, lons) * 10 + 4,
                ),
                "hfds": (
                    ["time", "lat", "lon"],
                    np.random.randn(total_time_steps, lats, lons) * 10 + 4,
                ),
                "wetmask": (
                    ["time", "lev", "lat", "lon"],
                    np.random.randint(0, 2, (total_time_steps, levels, lats, lons)),
                ),
            },
            coords={
                "time": np.arange(total_time_steps),
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

    assert len(groundtruth_dataset) == len(torch_dataset) == len(lazy_dataset)

    # Index 0 test
    gt_td0 = groundtruth_dataset[0]
    torch_td0 = torch_dataset[0]

    input_0, label_0 = lazy_dataset[0]
    collated_lazy_dataset0: TrainData = collate_om4([(input_0, label_0)])

    for i in range(len(gt_td0)):
        gt_input = gt_td0.get_input(i)
        torch_input = torch_td0.get_input(i)
        lazy_input = collated_lazy_dataset0.get_input(i)
        print(torch.sum(gt_input - torch_input))
        print(torch.sum(gt_input - lazy_input))
        assert gt_input.shape == torch_input.shape
        assert gt_input.shape == lazy_input.squeeze().shape
        assert torch.allclose(gt_input, torch_input)
        assert torch.allclose(gt_input, lazy_input)

        gt_label = gt_td0.get_label(i)
        torch_label = torch_td0.get_label(i)
        lazy_label = collated_lazy_dataset0.get_label(i)
        print(torch.sum(gt_label - torch_label))
        print(torch.sum(gt_label - lazy_label))
        assert gt_label.shape == torch_label.shape
        assert gt_label.shape == lazy_label.squeeze().shape
        assert torch.allclose(gt_label, torch_label)
        assert torch.allclose(gt_label, lazy_label)

    # Multiple Index - Collated Test
    gt_td1 = groundtruth_dataset[1]
    gt_td2 = groundtruth_dataset[2]

    torch_td1 = torch_dataset[1]
    torch_td2 = torch_dataset[2]

    collated_gt = collate_train_data([gt_td0, gt_td1, gt_td2])
    collated_torch = collate_train_data([torch_td0, torch_td1, torch_td2])

    input_1, label_1 = lazy_dataset[1]
    input_2, label_2 = lazy_dataset[2]
    collated_lazy_dataset = collate_om4(
        [(input_0, label_0), (input_1, label_1), (input_2, label_2)]
    )

    for i in range(len(collated_gt)):
        gt_input = collated_gt.get_input(i)
        torch_input = collated_torch.get_input(i)
        lazy_input = collated_lazy_dataset.get_input(i)
        print(torch.sum(gt_input - torch_input))
        print(torch.sum(gt_input - lazy_input))
        assert gt_input.shape == torch_input.shape
        assert gt_input.shape == lazy_input.shape
        assert torch.allclose(gt_input, torch_input)
        assert torch.allclose(gt_input, lazy_input)

        gt_label = collated_gt.get_label(i)
        torch_label = collated_torch.get_label(i)
        lazy_label = collated_lazy_dataset.get_label(i)
        print(torch.sum(gt_label - torch_label))
        print(torch.sum(gt_label - lazy_label))
        assert gt_label.shape == torch_label.shape
        assert gt_label.shape == lazy_label.shape
        assert torch.allclose(gt_label, torch_label)
        assert torch.allclose(gt_label, lazy_label)

    # TODO: Parameter-Sweep Test

    # TODO: Inference Test on Lazy Loader
