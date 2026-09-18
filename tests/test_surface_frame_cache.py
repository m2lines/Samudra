# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import xarray as xr

from samudra.constants import build_om4_layout
from samudra.datasets import TorchTrainDataset
from samudra.experiments.frame_cache import PreparedFrameCache
from samudra.experiments.surface_wave import training_batches
from samudra.utils.data import CanonicalSource
from samudra.utils.train import collate_host_batches


@pytest.fixture
def source():
    layout = build_om4_layout(prognostic_vars_key="thetao_1", boundary_vars_key="hfds")
    values = np.arange(20 * 3 * 4, dtype=np.float32).reshape(20, 3, 4)
    wet = np.ones((3, 4), dtype=bool)
    wet[0, 0] = False
    data = xr.Dataset(
        {
            "thetao_0": (("time", "lat", "lon"), values),
            "hfds": (("time", "lat", "lon"), values * 0.5 + 7),
            "mask_0": (("lat", "lon"), wet),
        },
        coords={
            "time": xr.date_range("2000-01-01", periods=20, freq="5D", use_cftime=True),
            "lat": [-60.0, 0.0, 60.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
    )
    selected = data[["thetao_0", "hfds"]]
    return CanonicalSource.from_datasets(
        data,
        selected.mean(("time", "lat", "lon")),
        selected.std(("time", "lat", "lon")),
        data_layout=layout,
        prognostic_var_names=layout.prognostic_var_names,
        boundary_var_names=layout.boundary_var_names,
    )


def dataset(source, steps):
    return TorchTrainDataset(
        input_source=source,
        label_source=None,
        prognostic_var_names=["thetao_0"],
        boundary_var_names=["hfds"],
        input_steps=6,
        output_steps=1,
        steps=steps,
        normalize_before_mask=True,
        masked_fill_value=0.0,
    )


def reference(data, ids):
    host = collate_host_batches([data[i] for i in ids])
    return data.preparer.prepare_host_batch(host, torch.device("cpu"))


@pytest.mark.parametrize("steps", [1, 6])
def test_cache_preserves_prepared_windows_masks_and_forcing(source, steps):
    warm = dataset(source, 1)
    cache = PreparedFrameCache(20, 1, 1, (3, 4), torch.device("cpu"))
    for ids in ([0, 6], [12, 13]):
        cache.record(warm.shard.window_plan(ids), reference(warm, ids))
    assert cache.prognostic_ready.all()
    assert cache.boundary_ready[:-1].all()
    data = dataset(source, steps)
    for ids in ([len(data) - 1, 0], [3, 3], [2, 5]):
        actual = cache.batch(data, ids)
        expected = reference(data, ids)
        for cached_step, expected_step in zip(actual, expected, strict=True):
            for cached, original in zip(cached_step, expected_step, strict=True):
                torch.testing.assert_close(cached, original, rtol=0, atol=0)
        assert torch.count_nonzero(actual.get_initial_input()[0][..., 0, 0]) == 0


def test_cache_rejects_unpopulated_frames(source):
    data = dataset(source, 1)
    cache = PreparedFrameCache(20, 1, 1, (3, 4), torch.device("cpu"))
    cache.record(data.shard.window_plan([0]), reference(data, [0]))
    with pytest.raises(ValueError, match="not populated"):
        cache.batch(data, [len(data) - 1])


def test_one_gpu_resume_preserves_each_global_batch():
    old = [training_batches(2823, 2, 2, rank, 1729, 10) for rank in range(2)]
    new = training_batches(2823, 4, 1, 0, 1729, 10)
    assert len(new) == len(old[0]) == len(old[1])
    for step in range(287, len(new)):
        assert sorted(new[step]) == sorted(old[0][step] + old[1][step])
