# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Source construction policies selected by the concrete loading configs."""

from collections.abc import Callable
from typing import Protocol

from samudra.config import DataSourceType
from samudra.constants import DataLayout
from samudra.native_reader import Om4IoRuntime, build_om4_reader
from samudra.utils.data import CanonicalReader, XarrayCanonicalReader
from samudra.utils.location import LocalLocation, ResolvedLocation


class TrainingSourceBackend(Protocol):
    def validate_source(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        source_type: DataSourceType,
        data_layout: DataLayout,
    ) -> None: ...

    def build_reader(
        self,
        xarray_reader: XarrayCanonicalReader,
        *,
        data_location: ResolvedLocation,
        data_layout: DataLayout,
    ) -> CanonicalReader: ...


class PythonSourceBackend:
    def validate_source(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        source_type: DataSourceType,
        data_layout: DataLayout,
    ) -> None:
        pass

    def build_reader(
        self,
        xarray_reader: XarrayCanonicalReader,
        *,
        data_location: ResolvedLocation,
        data_layout: DataLayout,
    ) -> CanonicalReader:
        return xarray_reader


class NativeOm4SourceBackend:
    """Validate native OM4 inputs and share a lazily constructed I/O runtime."""

    def __init__(self, name: str, runtime_factory: Callable[[], Om4IoRuntime]) -> None:
        self._name = name
        self._runtime_factory = runtime_factory
        self._runtime: Om4IoRuntime | None = None

    def validate_source(
        self,
        *,
        data_location: ResolvedLocation,
        means_location: ResolvedLocation,
        stds_location: ResolvedLocation,
        source_type: DataSourceType,
        data_layout: DataLayout,
    ) -> None:
        if source_type != "om4":
            raise ValueError(
                f"loading.type={self._name!r} currently supports OM4 sources only; "
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
                    f"loading.type={self._name!r} currently requires local data, "
                    f"but {field_name} resolved to {location}"
                )
        derived = [
            name
            for name in data_layout.boundary_var_names
            if name.endswith("_anomalies")
        ]
        if derived:
            raise ValueError(
                f"loading.type={self._name!r} does not yet support derived boundary "
                f"variables {derived}; select physical boundary variables or use "
                "loading.type='cpu'"
            )

    def build_reader(
        self,
        xarray_reader: XarrayCanonicalReader,
        *,
        data_location: ResolvedLocation,
        data_layout: DataLayout,
    ) -> CanonicalReader:
        assert isinstance(data_location, LocalLocation)
        if self._runtime is None:
            self._runtime = self._runtime_factory()
        return build_om4_reader(
            xarray_reader,
            data_location,
            data_layout,
            self._runtime,
            backend=self._name,
        )
