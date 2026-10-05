# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""TensorStore plane reads for the batch pipeline introduced by the Rust loader."""

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import zarr  # type: ignore[import-untyped]

from samudra.constants import CanonicalPlanes
from samudra.native_reader import Om4IoRuntime, PhysicalVariable


class TensorStoreIoRuntime(Om4IoRuntime):
    """Share native file I/O and copy concurrency across a rank's readers."""

    def __init__(self, max_concurrent_reads: int) -> None:
        if max_concurrent_reads < 1:
            raise ValueError("max_concurrent_reads must be positive")
        try:
            self._ts = import_module("tensorstore")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "TensorStore loading requires `pip install samudra[tensorstore]`."
            ) from error
        self._context = self._ts.Context(
            {
                "file_io_concurrency": {"limit": max_concurrent_reads},
                "data_copy_concurrency": {"limit": max_concurrent_reads},
            }
        )

    def open(
        self, path: Path, variables: Sequence[PhysicalVariable]
    ) -> "TensorStoreOm4Reader":
        return TensorStoreOm4Reader(path, variables, self._ts, self._context)


class TensorStoreOm4Reader:
    """Persistent float32 Zarr v2 views with a caller-owned output buffer."""

    def __init__(
        self,
        path: Path,
        variables: Sequence[PhysicalVariable],
        ts: Any,
        context: Any,
    ) -> None:
        self._ts = ts
        self._context = context
        self.shape: tuple[int, int, int]
        self._views: dict[tuple[str, int | None], Any] = {}
        group = zarr.open_group(str(path), mode="r")
        arrays: dict[str, Any] = {}
        for variable in variables:
            name, level = variable.name, variable.level
            metadata = group[name]
            if metadata.dtype != np.dtype("float32"):
                raise ValueError(f"TensorStore OM4 requires float32 arrays: {name}")
            dims = tuple(
                {"y": "lat", "x": "lon"}.get(dim, dim)
                for dim in metadata.attrs["_ARRAY_DIMENSIONS"]
            )
            expected = (
                (("time", "lat", "lon"),)
                if level is None
                else (("time", "lev", "lat", "lon"), ("lev", "time", "lat", "lon"))
            )
            if dims not in expected:
                raise ValueError(f"Unsupported OM4 dimensions for {name}: {dims}")
            # Native reads bypass Xarray's CF decoding; reject encoded values
            # rather than silently changing canonical data semantics.
            for attr in ("scale_factor", "add_offset"):
                if attr in metadata.attrs:
                    raise ValueError(f"TensorStore OM4 does not decode {attr}: {name}")
            for fill in (
                metadata.fill_value,
                metadata.attrs.get("missing_value"),
                metadata.attrs.get("_FillValue"),
            ):
                if fill is not None and not np.all(np.isnan(fill)):
                    raise ValueError(
                        f"TensorStore OM4 requires NaN fill values: {name}"
                    )
            if name not in arrays:
                arrays[name] = ts.open(
                    {
                        "driver": "zarr",
                        "kvstore": {"driver": "file", "path": str(path / name) + "/"},
                    },
                    open=True,
                    read=True,
                    write=False,
                    context=context,
                ).result()
            view = arrays[name]
            if level is not None:
                axis = dims.index("lev")
                if level < 0 or level >= metadata.shape[axis]:
                    raise ValueError(f"OM4 level out of range: {name}[{level}]")
                selection: list[slice | int] = [slice(None)] * len(dims)
                selection[axis] = level
                view = view[tuple(selection)]
            shape = (int(view.shape[0]), int(view.shape[1]), int(view.shape[2]))
            if any(size < 1 for size in shape):
                raise ValueError(f"OM4 arrays must have nonempty dimensions: {name}")
            if self._views and shape != self.shape:
                raise ValueError(f"OM4 arrays have inconsistent shapes: {name}")
            self.shape = shape
            self._views[name, level] = view
        if not self._views:
            raise ValueError("At least one OM4 variable must be selected")

    def read_into(
        self,
        indices: list[int],
        variables: Sequence[PhysicalVariable],
        output: CanonicalPlanes,
    ) -> None:
        selectors = [(v.name, v.level) for v in variables]
        expected = (len(indices), len(selectors), *self.shape[1:])
        if output.shape != expected or output.dtype != np.float32:
            raise ValueError(f"Expected a float32 output buffer with shape {expected}")
        if not output.flags.writeable:
            raise ValueError("Output buffer must be writable")
        if any(index < 0 or index >= self.shape[0] for index in indices):
            raise IndexError("OM4 time index out of range")
        views = [self._views[selector] for selector in selectors]
        if not indices or not selectors:
            return
        futures = []
        errors = []
        try:
            for channel, view in enumerate(views):
                # ts.array retains the NumPy view of the pinned Torch allocation.
                # Writes target memory only; the Zarr store is opened read-only.
                target = self._ts.array(
                    output[:, channel, :, :], context=self._context, copy=False
                )
                futures.append(target.write(view[indices, :, :]))
        except BaseException as error:
            errors.append(error)
        # Drain every submitted read, including on failure, before the caller
        # can release the buffer lease and reuse its memory for another batch.
        for future in futures:
            while True:
                try:
                    future.result()
                    break
                except BaseException as error:
                    errors.append(error)
                    # An interrupted wait can leave the copy running. Keep
                    # waiting until the buffer is safe to release.
                    if future.done():
                        break
        if errors:
            raise BaseExceptionGroup("TensorStore OM4 reads failed", errors)
