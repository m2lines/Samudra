# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Backend-specific decoration of canonical training sources."""

from typing import TYPE_CHECKING, Protocol

from samudra.config import (
    BaseDataLoadingConfig,
    NativeDataLoadingConfig,
    TensorStoreDataLoadingConfig,
)
from samudra.utils.data import CanonicalSource
from samudra.utils.location import LocalLocation, ResolvedLocation

if TYPE_CHECKING:
    from samudra.rust_data import Om4IoRuntime


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


class _NativeOm4SourceBackend:
    def __init__(self, loading: NativeDataLoadingConfig) -> None:
        self._loading = loading
        self._runtime: Om4IoRuntime | None = None

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
                "Native loading currently supports OM4 sources only; "
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
                    "Native loading currently requires local data, "
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
                "Native loading does not yet support derived boundary "
                f"variables {derived}; select physical boundary variables or use "
                "loading.type='cpu'"
            )

        from samudra.rust_data import create_rust_io_runtime, native_om4_source

        assert isinstance(data_location, LocalLocation)
        if self._runtime is None:
            if isinstance(self._loading, TensorStoreDataLoadingConfig):
                from samudra.tensorstore_data import TensorStoreIoRuntime

                self._runtime = TensorStoreIoRuntime(self._loading.max_concurrent_reads)
            else:
                self._runtime = create_rust_io_runtime(
                    self._loading.max_concurrent_reads
                )
        return native_om4_source(source, data_location, self._runtime)


def build_training_source_backend(
    loading: BaseDataLoadingConfig,
) -> TrainingSourceBackend:
    if isinstance(loading, NativeDataLoadingConfig):
        return _NativeOm4SourceBackend(loading)
    return _PythonSourceBackend()
