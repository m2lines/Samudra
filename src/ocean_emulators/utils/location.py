import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import quote, urljoin, urlparse

import xarray as xr
from pydantic import (
    BaseModel,
    BeforeValidator,
    WithJsonSchema,
    model_serializer,
    model_validator,
)

from ocean_emulators.utils.device import using_gpu

logger = logging.getLogger(__name__)
_ZARR_GPU_LOGGED = False

import zarr

try:  # Registers GPU-capable backends (e.g., kvikio) when available.
    import cupy_xarray  # noqa: F401
except ImportError:
    cupy_xarray = None


def enable_zarr_gpu():
    global _ZARR_GPU_LOGGED
    if not using_gpu():
        return nullcontext()
    if not _ZARR_GPU_LOGGED:
        logger.info("Enabling zarr GPU array support for xarray opens.")
        _ZARR_GPU_LOGGED = True
    return zarr.config.enable_gpu()


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

    @abstractmethod
    def supports_fork(self) -> bool:
        pass

    def __truediv__(self, other: "Location") -> "ResolvedLocation":
        return self.resolve(other)


class S3Location(ResolvedLocation, BaseModel):
    """An S3 bucket, assuming credentials in your environment.

    For example:
    ```yaml
    data_location:
      type: s3
      bucket: emulators
      path: sd5313/OM4_highres/om4_halfdeg.zarr
    ```
    """

    type: Literal["s3"] = "s3"
    endpoint_url: str | None = None
    bucket: str
    path: str

    def open(self, chunks: dict[str, int] | None = None) -> xr.Dataset:
        # TODO(jder): could consider passing credentials here
        # rather than relying on the environment
        enable_zarr_gpu()
        return xr.open_dataset(
            self.url(),
            backend_kwargs={
                "storage_options": {"endpoint_url": self.endpoint_url}
            },
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
                bucket=self.bucket,
                path=urljoin(self.path + "/", location.path),
            )
        return location

    def supports_fork(self) -> bool:
        return False  # s3fs does not support forking

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
        import kvikio.zarr
        zarr.config.enable_gpu()
        store = kvikio.zarr.GDSStore(root=self.path)
        dataset = xr.open_dataset(filename_or_obj=store, engine="zarr", decode_cf=False, create_default_indexes=False)
        # Debug print disabled per request.
        # if dataset.data_vars:
        #     var_name = next(iter(dataset.data_vars))
        #     data = dataset.data_vars[var_name].data
        #     print("type of random data var is", var_name, type(data))
        return dataset

    def resolve(self, location: "Location") -> "ResolvedLocation":
        if isinstance(location, UnresolvedLocation):
            return LocalLocation(path=self.path / location.path)
        return location

    def supports_fork(self) -> bool:
        return True

    def __str__(self) -> str:
        return str(self.path)


def string_to_unresolved(data: Any) -> Any:
    """Turns a string into an UnresolvedLocation."""
    # TODO(jder): we could support other fsspec or universal_pathlib URLs here
    if isinstance(data, str):
        return UnresolvedLocation(path=data)
    return data


Location = Annotated[
    Annotated[UnresolvedLocation, WithJsonSchema({"type": "string"})]
    | S3Location
    | LocalLocation,
    BeforeValidator(string_to_unresolved),
]
