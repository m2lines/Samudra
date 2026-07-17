# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-specific construction of canonical training sources."""

from typing import Protocol

from samudra.config import BaseDataLoadingConfig, RustDataLoadingConfig
from samudra.constants import DatasetSpec
from samudra.utils.data import CanonicalDataset
from samudra.utils.location import LocalLocation, ResolvedLocation


class TrainingSourceBackend(Protocol):
    def validate(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        dataset_spec: DatasetSpec,
    ) -> None: ...

    def prepare(
        self,
        dataset: CanonicalDataset,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        dataset_spec: DatasetSpec,
    ) -> CanonicalDataset: ...


class _PythonSourceBackend:
    def validate(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        dataset_spec: DatasetSpec,
    ) -> None:
        pass

    def prepare(
        self,
        dataset: CanonicalDataset,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        dataset_spec: DatasetSpec,
    ) -> CanonicalDataset:
        return dataset


class _RustOm4SourceBackend:
    def __init__(self, max_concurrent_reads: int) -> None:
        from samudra.rust_data import create_rust_io_runtime

        self._runtime = create_rust_io_runtime(max_concurrent_reads)

    def validate(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        dataset_spec: DatasetSpec,
    ) -> None:
        derived = [
            name
            for name in dataset_spec.boundary_var_names
            if name.endswith("_anomalies")
        ]
        if derived:
            raise ValueError(
                "loading.type='rust' does not yet support derived boundary "
                f"variables {derived}; select physical boundary variables or use "
                "loading.type='cpu'"
            )
        locations = {
            "data_location": data_location,
            "data_means_location": means_location,
            "data_stds_location": stds_location,
        }
        for field_name, location in locations.items():
            if not isinstance(location, LocalLocation):
                raise ValueError(
                    "loading.type='rust' currently requires local data, "
                    f"but {field_name} resolved to {location}"
                )

    def prepare(
        self,
        dataset: CanonicalDataset,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        dataset_spec: DatasetSpec,
    ) -> CanonicalDataset:
        from samudra.rust_data import native_om4_dataset

        assert isinstance(data_location, LocalLocation)
        return native_om4_dataset(dataset, data_location, self._runtime)


def build_training_source_backend(
    loading: BaseDataLoadingConfig,
) -> TrainingSourceBackend:
    if isinstance(loading, RustDataLoadingConfig):
        return _RustOm4SourceBackend(loading.max_concurrent_reads)
    return _PythonSourceBackend()
