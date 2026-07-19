# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import numpy as np
import pytest
import xarray as xr

from samudra.datasets import TrainingShard, season_decade_stratified_indices
from samudra.utils.data import CanonicalDataset


def test_season_decade_stratified_indices_are_balanced_and_deterministic():
    time = xr.DataArray(
        xr.cftime_range("1975-01-03", "2013-10-05", freq="5D", calendar="julian"),
        dims=["time"],
    )

    first = season_decade_stratified_indices(
        time, valid_size=time.size - 4, num_samples=512, seed=17, anchor_offset=2
    )
    repeated = season_decade_stratified_indices(
        time, valid_size=time.size - 4, num_samples=512, seed=17, anchor_offset=2
    )
    different = season_decade_stratified_indices(
        time, valid_size=time.size - 4, num_samples=512, seed=18, anchor_offset=2
    )

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, different)
    assert len(first) == len(np.unique(first)) == 512
    assert np.all(first[:-1] < first[1:])
    assert first.min() >= 0
    assert first.max() < time.size - 4

    strata = Counter(
        (
            int(time.values[index + 2].year) // 10 * 10,
            (int(time.values[index + 2].month) - 1) // 3,
        )
        for index in first
    )
    assert len(strata) == 20
    assert max(strata.values()) - min(strata.values()) <= 1


@pytest.mark.parametrize("data_source", ["mock"], indirect=True)
def test_training_shard_maps_selected_samples_to_contiguous_windows(
    data_source: CanonicalDataset,
):
    shard = TrainingShard(
        src=data_source,
        dst=None,
        prognostic_var_names=data_source.dataset_spec.prognostic_var_names,
        boundary_var_names=data_source.dataset_spec.boundary_var_names,
        hist=1,
        steps=1,
        normalize_before_mask=True,
        masked_fill_value=0.0,
        stride=1,
        sample_num=4,
        sample_seed=9,
    )

    assert len(shard) == 4
    assert shard.sample_indices is not None
    for logical_index, selected_start in enumerate(shard.sample_indices):
        window = shard.window_indices(logical_index, step=0)
        np.testing.assert_array_equal(
            window, np.arange(selected_start, selected_start + 4)
        )
