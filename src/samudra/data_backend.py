# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-specific decoration of canonical training sources."""

from typing import Protocol

from samudra.config import BaseDataLoadingConfig, RustDataLoadingConfig
from samudra.utils.data import CanonicalSource
from samudra.utils.location import LocalLocation, ResolvedLocation


class TrainingSourceBackend(Protocol):
    """Build-lifetime policy for canonical training readers."""

    def validate_locations(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        source_type: str,
    ) -> None: ...

    def prepare(
        self,
        source: CanonicalSource,
        *,
        data_location: ResolvedLocation,
        source_type: str,
    ) -> CanonicalSource: ...


class _PythonSourceBackend:
    def validate_locations(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        source_type: str,
    ) -> None:
        pass

    def prepare(
        self,
        source: CanonicalSource,
        *,
        data_location: ResolvedLocation,
        source_type: str,
    ) -> CanonicalSource:
        return source


class _RustOm4SourceBackend:
    def __init__(self, max_concurrent_reads: int) -> None:
        from samudra.rust_data import create_rust_io_runtime

        self._runtime = create_rust_io_runtime(max_concurrent_reads)

    def validate_locations(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        source_type: str,
    ) -> None:
        if source_type != "om4":
            raise ValueError(
                "loading.type='rust' currently supports OM4 sources only; "
                f"got {source_type!r}"
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
        source: CanonicalSource,
        *,
        data_location: ResolvedLocation,
        source_type: str,
    ) -> CanonicalSource:
        assert source_type == "om4"
        derived = [
            name
            for name in source.data_layout.boundary_var_names
            if name.endswith("_anomalies")
        ]
        if derived:
            raise ValueError(
                "loading.type='rust' does not yet support derived boundary "
                f"variables {derived}; select physical boundary variables or use "
                "loading.type='cpu'"
            )

        from samudra.rust_data import native_om4_source

        assert isinstance(data_location, LocalLocation)
        return native_om4_source(source, data_location, self._runtime)


def build_training_source_backend(
    loading: BaseDataLoadingConfig,
) -> TrainingSourceBackend:
    if isinstance(loading, RustDataLoadingConfig):
        return _RustOm4SourceBackend(loading.max_concurrent_reads)
    return _PythonSourceBackend()
