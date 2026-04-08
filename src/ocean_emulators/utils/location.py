import contextlib
import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Callable
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

logger = logging.getLogger(__name__)

_DEFAULT_KVIKIO_TASK_SIZE = 64 * 1024 * 1024
_DEFAULT_KVIKIO_NUM_THREADS = 8


def _zarr_gpu_decode_context(
    use_gpu_zarr_decode: bool,
) -> contextlib.AbstractContextManager[Any]:
    if not use_gpu_zarr_decode:
        return contextlib.nullcontext()

    try:
        import zarr  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "data.loading.type=gpu requires the zarr package to be installed."
        ) from exc

    config = getattr(zarr, "config", None)
    enable_gpu = getattr(config, "enable_gpu", None)
    if enable_gpu is None:
        raise RuntimeError(
            "data.loading.type=gpu requires a zarr build that provides "
            "`zarr.config.enable_gpu()`."
        )
    return enable_gpu()


def _configure_kvikio_for_gpu_decode(
    *,
    kvikio_task_size: int | None = None,
    kvikio_num_threads: int | None = None,
) -> None:
    try:
        import kvikio.defaults as kvikio_defaults  # type: ignore[import-not-found,import-untyped]
    except ModuleNotFoundError:
        return

    task_size = (
        kvikio_task_size
        if kvikio_task_size is not None
        else int(os.environ.get("OE_KVIKIO_TASK_SIZE", str(_DEFAULT_KVIKIO_TASK_SIZE)))
    )
    num_threads = (
        kvikio_num_threads
        if kvikio_num_threads is not None
        else int(
            os.environ.get("OE_KVIKIO_NUM_THREADS", str(_DEFAULT_KVIKIO_NUM_THREADS))
        )
    )
    if task_size <= 0:
        raise ValueError(f"OE_KVIKIO_TASK_SIZE must be > 0, got {task_size}.")
    if num_threads <= 0:
        raise ValueError(f"OE_KVIKIO_NUM_THREADS must be > 0, got {num_threads}.")

    current_task_size = kvikio_defaults.get("task_size")
    current_num_threads = kvikio_defaults.get("num_threads")
    if current_task_size == task_size and current_num_threads == num_threads:
        return

    kvikio_defaults.set(
        {
            "task_size": task_size,
            "num_threads": num_threads,
        }
    )
    logger.info(
        "Configured kvikIO for GPU zarr decode: task_size=%d num_threads=%d "
        "(was task_size=%d num_threads=%d)",
        task_size,
        num_threads,
        current_task_size,
        current_num_threads,
    )


def _local_gds_store(
    path: Path,
    *,
    kvikio_task_size: int | None = None,
    kvikio_num_threads: int | None = None,
) -> Any | None:
    try:
        import kvikio.zarr as kvikio_zarr  # type: ignore[import-not-found,import-untyped]
    except ModuleNotFoundError:
        logger.warning(
            "kvikio is not installed; falling back to standard zarr reads for %s",
            path,
        )
        return None

    try:
        _configure_kvikio_for_gpu_decode(
            kvikio_task_size=kvikio_task_size,
            kvikio_num_threads=kvikio_num_threads,
        )
        return kvikio_zarr.GDSStore(str(path))
    except Exception as exc:
        logger.warning(
            "Failed to initialize kvikio GDSStore for %s; falling back to standard "
            "zarr reads (%s)",
            path,
            exc,
        )
        return None


def _open_with_optional_gpu_decode(
    *,
    use_gpu_zarr_decode: bool,
    location: str,
    open_dataset: Callable[[], xr.Dataset],
) -> xr.Dataset:
    try:
        with _zarr_gpu_decode_context(use_gpu_zarr_decode):
            return open_dataset()
    except Exception as exc:
        if not use_gpu_zarr_decode:
            raise

        raise RuntimeError(
            "GPU zarr decode failed while opening "
            f"{location} with xarray. "
            "This usually means xarray/zarr GPU-buffer integration is incompatible "
            "with the current store or environment. "
            "Set data.loading.type=cpu to continue on the CPU path."
        ) from exc


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
    def open(
        self,
        chunks: dict[str, int] | None = None,
        *,
        use_gpu_zarr_decode: bool = False,
        kvikio_task_size: int | None = None,
        kvikio_num_threads: int | None = None,
    ) -> xr.Dataset:
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

    def open(
        self,
        chunks: dict[str, int] | None = None,
        *,
        use_gpu_zarr_decode: bool = False,
        kvikio_task_size: int | None = None,
        kvikio_num_threads: int | None = None,
    ) -> xr.Dataset:
        del kvikio_task_size, kvikio_num_threads
        # TODO(jder): could consider passing credentials here
        # rather than relying on the environment
        open_dataset_kwargs: dict[str, Any] = {
            "backend_kwargs": {"storage_options": {"endpoint_url": self.endpoint_url}},
            "engine": "zarr",
            "chunks": chunks,
        }
        if use_gpu_zarr_decode:
            open_dataset_kwargs["decode_cf"] = False
            open_dataset_kwargs["create_default_indexes"] = False
        return _open_with_optional_gpu_decode(
            use_gpu_zarr_decode=use_gpu_zarr_decode,
            location=self.url(),
            open_dataset=lambda: xr.open_dataset(self.url(), **open_dataset_kwargs),
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

    def open(
        self,
        chunks: dict[str, int] | None = None,
        *,
        use_gpu_zarr_decode: bool = False,
        kvikio_task_size: int | None = None,
        kvikio_num_threads: int | None = None,
    ) -> xr.Dataset:
        engine = "netcdf4" if self.path.suffix == ".nc" else "zarr"
        if engine == "netcdf4":
            del kvikio_task_size, kvikio_num_threads
            return xr.open_dataset(self.path, engine=engine, chunks=chunks)

        path_or_store: Path | Any = self.path
        if use_gpu_zarr_decode and (
            gds_store := _local_gds_store(
                self.path,
                kvikio_task_size=kvikio_task_size,
                kvikio_num_threads=kvikio_num_threads,
            )
        ):
            path_or_store = gds_store

        open_dataset_kwargs: dict[str, Any] = {
            "engine": engine,
            "chunks": chunks,
        }
        if use_gpu_zarr_decode:
            open_dataset_kwargs["decode_cf"] = False
            open_dataset_kwargs["create_default_indexes"] = False

        return _open_with_optional_gpu_decode(
            use_gpu_zarr_decode=use_gpu_zarr_decode,
            location=str(self.path),
            open_dataset=lambda: xr.open_dataset(path_or_store, **open_dataset_kwargs),
        )

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
