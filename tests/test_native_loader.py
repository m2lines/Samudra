# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared native pipeline contracts, exercised without the Rust extension."""

from dataclasses import dataclass, replace

import numpy as np
import pytest
import torch
from jaxtyping import Int

from samudra.config import JulianDate, Om4TimeConfig, RustDataLoadingConfig
from samudra.constants import CanonicalPlanes, TimeIndices
from samudra.datasets import TorchTrainDataset, TrainingWindows
from samudra.native_loader import NativeBatchLoader
from samudra.utils.data import CanonicalReader, CanonicalReadRequest, CanonicalSource
from samudra.utils.train import collate_host_batches
from tests.test_canonical_dataset import _equivalent_om4_sources


@dataclass(frozen=True)
class ArrayBulkReader:
    """Test implementation of the bulk contract with no native adapter ancestry."""

    root: CanonicalSource
    view: CanonicalReader
    indices: Int[np.ndarray, " source_time"]
    calls: list[tuple[TimeIndices, tuple[str, ...]]]

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

    def read(self, request: CanonicalReadRequest) -> CanonicalPlanes:
        pytest.fail("Native batching should use bulk reads")

    @property
    def storage_id(self) -> int:
        return id(self.root)

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.root.grid_size

    def read_into(
        self,
        indices: Int[np.ndarray, " time"],
        channels: tuple[str, ...],
        output: CanonicalPlanes,
    ) -> None:
        self.calls.append((indices.copy(), channels))
        np.copyto(output, self.root.read(self.indices[indices], channels))


@pytest.mark.parametrize("normalize_before_mask", [True, False])
@pytest.mark.parametrize("prefetch_to_device", [True, False])
@pytest.mark.parametrize(
    "device_type", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]
)
def test_shared_pipeline_accepts_an_independent_bulk_reader(
    normalize_before_mask, prefetch_to_device, device_type, monkeypatch
):
    source, _ = _equivalent_om4_sources()
    calls: list[tuple[TimeIndices, tuple[str, ...]]] = []
    native = replace(
        source,
        _reader=ArrayBulkReader(
            source, source.reader, np.arange(source.time.size), calls
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
    device = torch.device(device_type)
    loader = RustDataLoadingConfig(
        prefetch_batches=2, prefetch_to_device=prefetch_to_device
    ).build_batch_loader(
        [windows(native)],
        schedule,
        device,
        seed=17,
    )
    assert isinstance(loader, NativeBatchLoader)
    original_prepare = loader._prepare_batch
    observed_pinned: list[bool] = []

    def record_prepare(loaded):
        _, raw = loaded
        observed_pinned.extend(tensor.is_pinned() for tensor in raw.groups)
        return original_prepare(loaded)

    monkeypatch.setattr(loader, "_prepare_batch", record_prepare)
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
                    assert actual_tensor.device.type == device_type
                    torch.testing.assert_close(
                        actual_tensor.cpu(), expected_tensor, rtol=0, atol=0
                    )
    finally:
        loader.close()

    # One prognostic and one boundary read per batch, deduplicated across all steps.
    assert len(calls) == 2 * len(schedule)
    assert observed_pinned == [device_type == "cuda"] * (2 * len(schedule))
    np.testing.assert_array_equal(calls[0][0], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(calls[1][0], [0, 1, 2, 3])
    assert calls[0][1] == ("so_0", "so_2", "zos")
    assert calls[1][1] == ("hfds",)


def test_native_bulk_read_rejects_duplicate_channels_before_io():
    from unittest.mock import Mock

    from samudra.native_reader import NativeOm4Reader, PhysicalVariable
    from samudra.utils.data import XarrayCanonicalReader

    source, _ = _equivalent_om4_sources()
    assert isinstance(source.reader, XarrayCanonicalReader)
    planes = Mock()
    reader = NativeOm4Reader(
        xarray_reader=source.reader,
        channels=source.channels,
        path="unused",
        _planes=planes,
        _reader_variables={"so_0": PhysicalVariable("so_0")},
        _physical_time_indices=np.arange(source.time.size),
        _spatial_shape=source.grid_size,
    )
    output = np.full((1, 2, *source.grid_size), np.nan, dtype=np.float32)
    with pytest.raises(ValueError, match="channel names must be unique"):
        reader.read_into(np.array([0]), ("so_0", "so_0"), output)
    planes.read_into.assert_not_called()
    assert np.isnan(output).all()
