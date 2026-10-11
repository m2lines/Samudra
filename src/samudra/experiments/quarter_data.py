# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded OM4 windows for a native-grid observation experiment.

``QuarterOm4Windows.from_compact(source, path, ...)`` decorates an already
canonicalized source with the existing persistent Rust reader. The source may
be a recent/early/validation time slice; physical storage indices remain the
reader's responsibility. ``origins`` optionally restricts admissible window
starts within that slice. No training model, full-period field cache, worker,
or prefetch queue is constructed here.

``model_sample([index])`` returns the six tensors consumed by the existing
OM4 objective: surface, past forcing, context, truth, future forcing, labels.
The index addresses ``origins``, not the physical Zarr time axis. A runner must
adapt its dataset argument to this interface and qualify its source contract.
Only microbatch one and six forecast steps are supported. Memory scales with
one 25-frame window, not the number of years. At 77 state channels the three
reads total 726 fp32 spatial planes (2.80 GiB at 720x1440), excluding native
decompression buffers, normalization temporaries and returned CUDA tensors.

This module is not yet wired into training. It does not implement DDP, source
provenance checks, observation loading, asynchronous prefetch, or GPU fit gates.
"""

import datetime
import math
from collections.abc import Sequence
from typing import Protocol, cast

import numpy as np
import torch

from samudra.experiments.surface_state import geographic_features
from samudra.utils.data import CanonicalSource, ChannelStatistics


class WindowReader(Protocol):
    """Narrow native-reader seam, also injectable for independent CPU tests."""

    def physical_indices(self, relative: np.ndarray) -> np.ndarray: ...

    def read_unique(
        self, physical_time_indices: np.ndarray, *, channels: tuple[str, ...]
    ) -> torch.Tensor: ...


class QuarterOm4Windows:
    def __init__(
        self,
        source: CanonicalSource,
        reader: WindowReader,
        *,
        device: str | torch.device = "cpu",
        observation_statistics: ChannelStatistics | None = None,
        origins: Sequence[int] | np.ndarray | None = None,
    ):
        self.device = torch.device(device)
        self.reader = reader
        self._runtime: object | None = None
        self.names = tuple(source.data_layout.prognostic_var_names)
        self.forcing_names = ("tauuo", "tauvo", "hfds")
        self.surface_ids = [self.names.index("thetao_0"), self.names.index("zos")]
        self.surface_names = tuple(self.names[i] for i in self.surface_ids)
        self.times = np.array(source.time.values, copy=True)
        if len(self.times) < 25:
            raise ValueError("OM4 source needs at least 25 frames")
        # Timing is cheap metadata; never silently bridge a missing five-day bin.
        for delta in np.diff(self.times):
            expected = (
                np.timedelta64(5, "D")
                if isinstance(delta, np.timedelta64)
                else datetime.timedelta(days=5)
            )
            if delta != expected:
                raise ValueError("OM4 windows require contiguous five-day frames")
        candidates = (
            np.arange(len(self.times) - 24) if origins is None else np.asarray(origins)
        )
        if (
            candidates.ndim != 1
            or not np.issubdtype(candidates.dtype, np.integer)
            or not len(candidates)
            or np.any(candidates < 0)
            or np.any(candidates >= len(self.times) - 24)
        ):
            raise ValueError("Origins must index complete 25-frame windows")
        self.origins = candidates.astype(np.int64, copy=True)
        self.origins.setflags(write=False)
        self.mask = source.masks.prognostic.to(self.device).bool()
        self.boundary_mask = source.masks.boundary.to(self.device).bool()
        lat, lon = source.resolution
        self.lat, self.lon = lat.to(self.device).float(), lon.to(self.device).float()
        self.shape = (len(self.lat), len(self.lon))
        if tuple(self.mask.shape) != (len(self.names), *self.shape):
            raise ValueError("State mask does not match channel order and grid")
        if tuple(self.boundary_mask.shape) not in (
            self.shape,
            (1, *self.shape),
            (3, *self.shape),
        ):
            raise ValueError("Boundary mask does not match forcing channels and grid")
        self.geo = geographic_features(self.lat, self.lon)
        self.weights = self.mask.float() * self.lat.deg2rad().cos()[None, :, None]
        self.native_mean, self.native_std = self._statistics(
            source.statistics(self.names), len(self.names)
        )
        self.forcing_mean, self.forcing_std = self._statistics(
            source.statistics(self.forcing_names), 3
        )
        self.observation_normalization = observation_statistics is not None
        self.mean, self.std = (
            self._statistics(observation_statistics, len(self.names))
            if observation_statistics is not None
            else (self.native_mean, self.native_std)
        )

    @classmethod
    def from_compact(cls, source, path, *, max_concurrent_reads=4, **kwargs):
        """Attach the existing compact-aware Rust backend without reading fields.

        ``path`` is the physical OM4.zarr, while ``source`` supplies canonical
        masks/statistics and the desired time slice. Keep this object alive for
        the reader's lifetime; each rank should construct its own instance.
        """
        from samudra.rust_data import create_rust_io_runtime, native_om4_source
        from samudra.utils.location import LocalLocation

        runtime = create_rust_io_runtime(max_concurrent_reads)
        native = native_om4_source(source, LocalLocation(path=path), runtime)
        # native_om4_source guarantees these extra native methods, while its
        # return annotation exposes the storage-independent CanonicalReader.
        result = cls(native, cast(WindowReader, native.reader), **kwargs)
        result._runtime = runtime
        return result

    def __len__(self):
        return len(self.origins)

    def _statistics(self, statistics, channels):
        mean = torch.as_tensor(statistics.mean, device=self.device)
        std = torch.as_tensor(statistics.std, device=self.device)
        if (
            mean.shape != (channels,)
            or std.shape != (channels,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or not (std > 0).all()
        ):
            raise ValueError("Expected finite channel statistics and positive scales")
        return mean, std

    def _read(self, start, stop, channels):
        relative = np.arange(start, stop, dtype=np.int64)
        values = self.reader.read_unique(
            self.reader.physical_indices(relative), channels=channels
        )
        if values.dtype != torch.float32 or tuple(values.shape) != (
            stop - start,
            len(channels),
            *self.shape,
        ):
            raise ValueError("Native window read must return ordered fp32 planes")
        return values.to(self.device)

    @staticmethod
    def _normalize(values, mean, std, mask):
        # Same operation order and NaN handling as TrainBatchPreparer. In
        # particular, normalize before masking and leave valid native NaNs at 0.
        mean = mean.to(values.dtype)[None, :, None, None]
        std = std.to(values.dtype)[None, :, None, None]
        normalized = ((values - mean) / std).nan_to_num(nan=0.0)
        return torch.where(mask, normalized, 0.0)

    def _state(self, values, channels):
        native = self._normalize(
            values,
            self.native_mean[channels],
            self.native_std[channels],
            self.mask[channels],
        )
        if not self.observation_normalization:
            return native
        # Preserve InitializerWave.model_sample's round trip exactly; replacing
        # this with direct physical -> observation normalization changes rounding.
        shape = (1, -1, 1, 1)
        return (
            (
                native * self.native_std[channels].reshape(shape)
                + self.native_mean[channels].reshape(shape)
                - self.mean[channels].reshape(shape)
            )
            / self.std[channels].reshape(shape)
        ) * self.mask[channels]

    def model_sample(self, ids):
        """Read one indexed window; no sample tensors survive in this object."""
        indices = np.asarray(ids)
        if (
            indices.shape != (1,)
            or not np.issubdtype(indices.dtype, np.integer)
            or indices[0] < 0
            or indices[0] >= len(self)
        ):
            raise ValueError("Provide exactly one valid window index")
        start = int(self.origins[indices[0]])
        surface = self._state(
            self._read(start, start + 19, self.surface_names), self.surface_ids
        )
        state = self._state(self._read(start + 17, start + 25, self.names), slice(None))
        forcing = self._normalize(
            self._read(start, start + 24, self.forcing_names),
            self.forcing_mean,
            self.forcing_std,
            self.boundary_mask,
        )
        date = self.times[start + 18]
        if isinstance(date, np.datetime64):
            day = (
                int(
                    (
                        date.astype("datetime64[D]") - date.astype("datetime64[Y]")
                    ).astype(int)
                )
                + 1
            )
        else:
            day = date.dayofyr if hasattr(date, "dayofyr") else date.timetuple().tm_yday
        phase = 2 * math.pi * (day - 1) / 365.25
        season = surface.new_tensor([math.sin(phase), math.cos(phase)])[:, None, None]
        context = torch.cat((self.geo, season.expand(2, *self.shape)), 0)[None]
        return (
            surface[None],
            forcing[None, :19],
            context,
            state[None, :2],
            forcing[None, 18:24],
            state[None, 2:],
        )
