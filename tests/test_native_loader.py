# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared native pipeline contracts, exercised without the Rust extension."""

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch

from samudra.config import JulianDate, Om4TimeConfig, RustDataLoadingConfig
from samudra.datasets import TorchTrainDataset, TrainingWindows
from samudra.utils.data import CanonicalReader, CanonicalReadRequest, CanonicalSource
from samudra.utils.train import collate_host_batches
from tests.test_canonical_dataset import _equivalent_om4_sources


@dataclass(frozen=True)
class ArrayBulkReader:
    """Test implementation of the bulk contract with no native adapter ancestry."""

    root: CanonicalSource
    view: CanonicalReader
    indices: np.ndarray
    calls: list[tuple[np.ndarray, tuple[str, ...]]]

    @property
    def channels(self):
        return self.view.channels

    @property
    def time(self):
        return self.view.time

    @property
    def resolution(self):
        return self.view.resolution

    @property
    def attrs(self):
        return self.view.attrs

    def statistics(self, channels):
        return self.view.statistics(channels)

    def coordinates(self):
        return self.view.coordinates()

    def metadata(self, data_layout):
        return self.view.metadata(data_layout)

    def slice_time(self, time):
        view = self.view.slice_time(time)
        positions = self.time.to_index().get_indexer(view.time.to_index())
        return replace(self, view=view, indices=self.indices[positions])

    def read(self, request: CanonicalReadRequest) -> np.ndarray:
        pytest.fail("Native batching should use bulk reads")

    @property
    def storage_id(self) -> int:
        return id(self.root)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.root.grid_size

    def physical_indices(self, relative: np.ndarray) -> np.ndarray:
        return self.indices[relative]

    def read_into(self, indices, channels, output):
        self.calls.append((indices.copy(), channels))
        np.copyto(output, self.root.read(indices, channels))


@pytest.mark.parametrize("normalize_before_mask", [True, False])
def test_shared_pipeline_accepts_an_independent_bulk_reader(
    normalize_before_mask, monkeypatch
):
    source, _ = _equivalent_om4_sources()
    data, means, stds = source._xarray_datasets_for_testing()
    calls: list[tuple[np.ndarray, tuple[str, ...]]] = []
    native = CanonicalSource.from_datasets(
        data,
        means,
        stds,
        data_layout=source.data_layout,
        prognostic_var_names=["so_0", "so_2", "zos"],
        boundary_var_names=["hfds"],
        reader_factory=lambda metadata: ArrayBulkReader(
            source, metadata, np.arange(source.time.size), calls
        ),
    )
    period = Om4TimeConfig(start=JulianDate("2000-01-02"), end=JulianDate("2000-01-08"))

    def windows(view):
        return TrainingWindows(
            input_source=view.slice_time(period),
            label_source=None,
            prognostic_var_names=["so_0", "so_2", "zos"],
            boundary_var_names=["hfds"],
            input_steps=2,
            output_steps=1,
            steps=2,
            normalize_before_mask=normalize_before_mask,
            masked_fill_value=-1.0,
        )

    schedule = [[1, 0], [2]]
    reference = TorchTrainDataset(windows(source))

    def forbidden_import(*args, **kwargs):
        pytest.fail("Shared batch construction imported the Rust extension")

    monkeypatch.setattr("samudra.rust_reader._load_extension", forbidden_import)
    loader = RustDataLoadingConfig(prefetch_batches=2).build_batch_loader(
        [windows(native)],
        schedule,
        torch.device("cpu"),
        pin_memory=False,
        multiprocessing_context=None,
        worker_seed=0,
        concurrent_compute=False,
    )
    try:
        for actual, indices in zip(loader, schedule, strict=True):
            expected = reference.to_model_batch(
                collate_host_batches([reference[index] for index in indices]),
                torch.device("cpu"),
            )
            for actual_step, expected_step in zip(
                actual.steps, expected.steps, strict=True
            ):
                for actual_tensor, expected_tensor in zip(
                    actual_step, expected_step, strict=True
                ):
                    torch.testing.assert_close(
                        actual_tensor, expected_tensor, rtol=0, atol=0
                    )
    finally:
        loader.close()

    # One prognostic and one boundary read per batch, deduplicated across all steps.
    assert len(calls) == 2 * len(schedule)
    np.testing.assert_array_equal(calls[0][0], [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(calls[1][0], [1, 2, 3, 4])
    assert calls[0][1] == ("so_0", "so_2", "zos")
    assert calls[1][1] == ("hfds",)
