# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote, urljoin, urlparse

import xarray as xr
from pydantic import (
    BaseModel,
    BeforeValidator,
    PlainSerializer,
    SerializationInfo,
    WithJsonSchema,
    model_serializer,
    model_validator,
)


class UnresolvedLocation(BaseModel):
    """Representation for a raw string in a Location config.

    This is expected to be a relative or absolute path.
    It is flexibily interepreted as a url or local path depending
    on what kind of location it is resolved against.
    """

    path: str

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if urlparse(self.path).scheme:
            raise ValueError(
                "Absolute urls are not supported, please use a "
                "relative path or set type = 's3' or 'local'"
            )
        return self

    @model_serializer
    def seralize(self) -> Any:
        return self.path


class ResolvedLocation(ABC):
    """A location which is ready to be opened or resolved against."""

    @abstractmethod
    def open(self, chunks: dict[str, int] | None = None) -> xr.Dataset:
        pass

    @abstractmethod
    def resolve(self, location: "Location") -> "ResolvedLocation":
        pass

    def __truediv__(self, other: "Location") -> "ResolvedLocation":
        return self.resolve(other)


class S3Location(ResolvedLocation, BaseModel):
    """An S3 bucket, assuming credentials in your environment.

    Set ``anon: true`` to read a public bucket without credentials (e.g. the
    open datasets on OSN); leave it false to sign requests with the usual
    ``AWS_*`` environment variables.

    For example:
    ```yaml
    data_location:
      type: s3
      endpoint_url: https://nyu1.osn.mghpcc.org
      anon: true
      bucket: m2lines-pubs
      path: Samudra/v2026-09/om4_halfdeg/OM4.zarr
    ```
    """

    type: Literal["s3"] = "s3"
    endpoint_url: str | None = None
    anon: bool = False
    bucket: str
    path: str

    def open(self, chunks: dict[str, int] | None = None) -> xr.Dataset:
        # TODO(jder): could consider passing credentials here
        # rather than relying on the environment
        storage_options: dict[str, Any] = {"endpoint_url": self.endpoint_url}
        if self.anon:
            storage_options["anon"] = True

        return xr.open_dataset(
            self.url(),
            backend_kwargs={"storage_options": storage_options},
            engine="zarr",
            chunks=chunks,
        )

    def url(self) -> str:
        path = quote(self.path.lstrip("/"))
        bucket = quote(self.bucket, safe="")
        return f"s3://{bucket}/{path}"

    def resolve(self, location: "Location") -> "ResolvedLocation":
        if isinstance(location, UnresolvedLocation):
            return S3Location(
                endpoint_url=self.endpoint_url,
                anon=self.anon,
                bucket=self.bucket,
                path=urljoin(self.path + "/", location.path),
            )
        return location

    def __str__(self) -> str:
        return self.url()


class LocalLocation(ResolvedLocation, BaseModel):
    """A local absolute filesystem path.

    For example:
    ```yaml
    data_location:
      type: local
      path: /path/to/data
    ```
    """

    type: Literal["local"] = "local"
    path: Path

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        if not self.path.is_absolute():
            raise ValueError(
                "Locations with type: 'local' must be absolute. "
                "For relative paths, use a string instead of a structured location. "
                "i.e. 'my/relative/path' instead of "
                "{type: 'local', path: 'my/relative/path'}"
            )
        return self

    def open(self, chunks: dict[str, int] | None = None) -> xr.Dataset:
        engine = "netcdf4" if self.path.suffix == ".nc" else "zarr"
        return xr.open_dataset(self.path, engine=engine, chunks=chunks)

    def resolve(self, location: "Location") -> "ResolvedLocation":
        if isinstance(location, UnresolvedLocation):
            return LocalLocation(path=self.path / location.path)
        return location

    def __str__(self) -> str:
        return str(self.path)


def string_to_unresolved(data: Any) -> Any:
    """Turns a string into an UnresolvedLocation."""
    # TODO(jder): we could support other fsspec or universal_pathlib URLs here
    if isinstance(data, str):
        return UnresolvedLocation(path=data)
    return data


def serialize_location(location: Any, info: SerializationInfo) -> Any:
    """Preserve structured locations when serializing a Location union."""
    if isinstance(location, UnresolvedLocation):
        return location.path
    return location.model_dump(mode=info.mode)


Location = Annotated[
    Annotated[UnresolvedLocation, WithJsonSchema({"type": "string"})]
    | S3Location
    | LocalLocation,
    BeforeValidator(string_to_unresolved),
    PlainSerializer(serialize_location),
]


def sibling_location(location: Location, suffix: str) -> Location:
    """The store beside ``location`` named ``<stem>_<suffix><ext>``.

    This is the release layout's naming convention for a dataset's companion
    stores: ``OM4.zarr`` sits beside ``OM4_means.zarr`` and ``OM4_stds.zarr``.
    """
    match location:
        case UnresolvedLocation(path=path):
            p = PurePosixPath(path)
            return UnresolvedLocation(
                path=str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))
            )
        case LocalLocation(path=path):
            return LocalLocation(
                path=path.with_name(f"{path.stem}_{suffix}{path.suffix}")
            )
        case S3Location(path=path):
            p = PurePosixPath(path)
            return location.model_copy(
                update={"path": str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))}
            )
    raise TypeError(f"Unsupported location type: {type(location).__name__}")
