# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Native-calendar OM4/CM4 windows on the verified common Gaussian grid."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, TypedDict

import cftime
import numpy as np
import torch
import zarr  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray

from samudra.constants import build_om4_layout
from samudra.observations.constants import (
    FORECAST_STEPS,
    HISTORY_STEPS,
    MAX_REPLAY_MONTHS,
    MODEL_DEPTHS,
    MODEL_FORCING_VARIABLES,
    MODEL_VARIABLES,
    STEP_DAYS,
    TEMPERATURE_OFFSET,
)
from samudra.observations.intervals import CalendarDate, calendar_weights

if TYPE_CHECKING:
    from samudra.observations.shared_era5 import SharedEra5
    from samudra.rust_reader import RustIoRuntime


class ContinuationLayout(TypedDict):
    start_lead: int
    end_lead: int
    next_start_lead: int


class ModelSample(ContinuationLayout):
    """One monthly model window, with normalized fields and native calendar timing."""

    fields: torch.Tensor  # [batch=1, history+forecast, channel, y, x]
    forcing: torch.Tensor  # [batch=1, history+forecast, forcing_channel, y, x]
    contexts: torch.Tensor  # [batch=1, history+forecast, context_channel, y, x]
    mask: torch.Tensor  # [channel, y, x]
    weights: torch.Tensor  # [forecast_step], calendar-month overlap fractions
    duration: int  # Calendar-month duration in days.
    first: str  # First forecast interval start in the source calendar.
    origin_index: int  # Original forecast start index in the source store.


class ModelWindow(NamedTuple):
    start: int
    stop: int
    weights: NDArray[np.float32]
    duration: int


class ContinuationWindow(NamedTuple):
    start: int
    stop: int
    weights: NDArray[np.float32]
    duration: int
    origin_start: int
    next_start: int


def normalize_physical_fields(
    values: NDArray, mask: NDArray[np.bool_], mean: NDArray, std: NDArray
) -> NDArray[np.float32]:
    """Normalize an owned read buffer with the original NumPy rounding order.

    Reuse storage when each operation preserves dtype. Avoid gathering all wet
    values or allocating several full history/target arrays just to mask land.
    """
    if not np.all(np.isfinite(values), where=mask[None]):
        raise ValueError("Nonfinite model target on declared wet support")
    mean, std = mean[:, None, None], std[:, None, None]
    shifted = np.subtract(
        values,
        mean,
        out=values if np.result_type(values, mean) == values.dtype else None,
    )
    scaled = np.divide(
        shifted,
        std,
        out=shifted if np.result_type(shifted, std) == shifted.dtype else None,
    )
    np.copyto(scaled, 0, where=~mask)
    return scaled.astype(np.float32, copy=False)


def calendar_context(
    lat: ArrayLike, lon: ArrayLike, dates: Sequence[CalendarDate]
) -> NDArray[np.float32]:
    """Return [time, context_channel=5, y, x] position and annual-phase planes."""
    latitude, longitude = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    xyz = np.stack(
        [
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ]
    )
    result = []
    for stamp in dates:
        first = type(stamp)(stamp.year, 1, 1)
        last = type(stamp)(stamp.year + 1, 1, 1)
        phase = (
            2 * np.pi * (stamp - first).total_seconds() / (last - first).total_seconds()
        )
        season = np.broadcast_to(
            np.array([np.sin(phase), np.cos(phase)])[:, None, None],
            (2, *latitude.shape),
        )
        result.append(np.concatenate((xyz, season)))
    return np.asarray(result, dtype=np.float32)


class ModelMonths:
    """Read physical fields using shared observation scales and native forcing/masks.

    OM4 uses the established pre-validation cutoff. CM4 uses its first 180 model
    years for training, reserving the final twenty for future diagnostics.
    """

    def __init__(
        self,
        path: str | Path,
        kind: Literal["om4", "cm4"],
        reference: Mapping[str, NDArray],
        om4_statistics: str | Path,
        *,
        runtime: RustIoRuntime | None = None,
    ) -> None:
        self.era5: SharedEra5 | None = None
        if kind not in ("om4", "cm4"):
            raise ValueError(kind)
        self.kind = kind
        self.source_metadata_sha256 = hashlib.sha256(
            (Path(path) / ".zmetadata").read_bytes()
        ).hexdigest()
        self.store = zarr.open_consolidated(str(path), mode="r")
        self.grid = reference
        self.names = reference["names"].tolist()
        if self.names != build_om4_layout().prognostic_var_names:
            raise ValueError("Unexpected physical channel order")
        z = self.store
        self.planes = None
        if runtime is not None:
            from samudra.native_reader import PhysicalVariable

            if kind != "om4":
                raise ValueError("Native observation training currently supports OM4")
            self.physical_variables = [PhysicalVariable(n) for n in self.names]
            self.planes = runtime.open(Path(path), self.physical_variables)
        lat = z["lat"][:, 0] if kind == "cm4" else z["y"][:]
        lon = z["lon"][0] if kind == "cm4" else z["x"][:]
        np.testing.assert_allclose(lat, reference["lat"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(lon, reference["lon"], atol=1e-6, rtol=0)
        self.lat, self.lon = lat, lon
        if kind == "cm4":
            np.testing.assert_allclose(
                z["lat"][:],
                np.broadcast_to(lat[:, None], (len(lat), len(lon))),
                atol=1e-6,
                rtol=0,
            )
            np.testing.assert_allclose(
                z["lon"][:],
                np.broadcast_to(lon[None], (len(lat), len(lon))),
                atol=1e-6,
                rtol=0,
            )
            np.testing.assert_allclose(
                z["dz"][:], build_om4_layout().depth_thickness, rtol=0, atol=1e-6
            )
            wet = z["wetmask"][:].astype(bool)
            self.mask = np.concatenate([wet] * len(MODEL_VARIABLES) + [wet[:1]])
            bounds = z["time_bnds"]
            self.bounds = cftime.num2date(
                bounds[:], bounds.attrs["units"], bounds.attrs["calendar"]
            )
            labels = cftime.num2date(
                z["time"][:], z["time"].attrs["units"], z["time"].attrs["calendar"]
            )
            if not np.array_equal(labels, self.bounds[:, 1]):
                raise ValueError("CM4 timestamps are not verified interval ends")
        else:
            self.mask = np.stack(
                [z[f"mask_{i}"][:] for i in range(MODEL_DEPTHS)] * len(MODEL_VARIABLES)
                + [z["mask_0"][:]]
            ).astype(bool)
            labels = cftime.num2date(
                z["time"][:],
                z["time"].attrs["units"],
                z["time"].attrs.get("calendar", "standard"),
            )
            # Published OM4 five-day means are centered at 12:00 on day three.
            self.bounds = np.array(
                [
                    [
                        t - timedelta(days=STEP_DAYS / 2),
                        t + timedelta(days=STEP_DAYS / 2),
                    ]
                    for t in labels
                ],
                dtype=object,
            )
        lengths = [(b - a).total_seconds() for a, b in self.bounds]
        if any(v != STEP_DAYS * 86400 for v in lengths):
            raise ValueError("Non-five-day source intervals")
        gaps = self.bounds[1:, 0] != self.bounds[:-1, 1]
        if kind == "cm4" and gaps.any():
            raise ValueError("CM4 interval bounds are not contiguous")
        self.midpoints = [a + (b - a) / 2 for a, b in self.bounds]
        statistics = Path(om4_statistics)
        mean = zarr.open_consolidated(str(statistics / "OM4_means.zarr"), mode="r")
        std = zarr.open_consolidated(str(statistics / "OM4_stds.zarr"), mode="r")
        self.forcing_mean = np.array(
            [float(mean[n][...]) for n in MODEL_FORCING_VARIABLES], dtype=np.float32
        )
        self.forcing_std = np.array(
            [float(std[n][...]) for n in MODEL_FORCING_VARIABLES], dtype=np.float32
        )
        if not np.isfinite(self.forcing_mean).all() or not (self.forcing_std > 0).all():
            raise ValueError("Invalid forcing normalization")
        self.windows: list[ModelWindow] = []
        self.window_months: list[CalendarDate] = []
        self.excluded_gap_months: list[str] = []
        first_year = self.bounds[0, 0].year
        for year in range(first_year, self.bounds[-1, 1].year + 1):
            for month in range(1, 13):
                first = type(self.midpoints[0])(year, month, 1)
                last = type(first)(year + (month == 12), month % 12 + 1, 1)
                indices = np.arange(
                    np.searchsorted(self.bounds[:, 1], first, side="right"),
                    np.searchsorted(self.bounds[:, 0], last, side="left"),
                )
                if (
                    not len(indices)
                    or indices[0] < HISTORY_STEPS
                    or self.bounds[indices[-1], 1] < last
                ):
                    continue
                if kind == "om4" and (
                    year < 1975
                    or (year, month) > (2013, 7)
                    or self.bounds[indices[0] - HISTORY_STEPS, 0].year < 1975
                ):
                    continue
                if kind == "cm4" and year >= first_year + 180:
                    continue
                # History frames may straddle missing days. Every forecast
                # transition, including the last history frame -> first target,
                # must still span exactly one contiguous five-day interval.
                # TODO: add temporal encoding and random history dropout during
                # training to improve initializer robustness to missing days.
                if gaps[indices[0] - 1 : indices[-1]].any():
                    self.excluded_gap_months.append(str(first))
                    continue
                weights = calendar_weights(
                    self.bounds[indices, 0], self.bounds[indices, 1], first, last
                )
                self.window_months.append(first)
                self.windows.append(
                    ModelWindow(
                        int(indices[0]),
                        int(indices[-1] + 1),
                        weights,
                        (last - first).days,
                    )
                )
        if not self.windows:
            raise ValueError("No training months")

    def sample(self, index: int, device: torch.device) -> ModelSample:
        start, stop, weights, duration = self.windows[index % len(self.windows)]
        return self._read(
            start,
            stop,
            weights,
            duration,
            device,
            origin_start=start,
            next_start=self._next_start(self.window_months[index % len(self.windows)]),
        )

    def _next_start(self, first: CalendarDate) -> int:
        last = type(first)(first.year + (first.month == 12), first.month % 12 + 1, 1)
        return int(
            np.searchsorted(
                self.bounds[:, 0], np.asarray(last, dtype=object), side="right"
            )
            - 1
        )

    def _continuation_window(
        self, index: int, month_offset: int
    ) -> ContinuationWindow | None:
        if month_offset < 0 or month_offset >= MAX_REPLAY_MONTHS:
            raise ValueError("Replay month offset must be in 0..11")
        origin_start = self.windows[index][0]
        original = self.window_months[index]
        month_id = original.year * 12 + original.month - 1 + month_offset
        first = type(original)(month_id // 12, month_id % 12 + 1, 1)
        candidates = [i for i, month in enumerate(self.window_months) if month == first]
        if not candidates:
            return None
        start, stop, weights, duration = self.windows[candidates[0]]
        if any(
            self.bounds[k, 1] != self.bounds[k + 1, 0]
            # Check from the original forecast boundary, not its input history.
            for k in range(origin_start - 1, stop - 1)
        ):
            return None
        if stop - origin_start > FORECAST_STEPS:
            return None
        return ContinuationWindow(
            start, stop, weights, duration, origin_start, self._next_start(first)
        )

    def continuation_layout(
        self, index: int, month_offset: int
    ) -> ContinuationLayout | None:
        """Inspect eligibility without reading any physical field data."""
        window = self._continuation_window(index, month_offset)
        if window is None:
            return None
        start, stop, _, _, origin_start, next_start = window
        return dict(
            start_lead=int(start - origin_start),
            end_lead=int(stop - origin_start),
            next_start_lead=int(next_start - origin_start),
        )

    def continuation(
        self, index: int, month_offset: int, device: torch.device
    ) -> ModelSample:
        window = self._continuation_window(index, month_offset)
        if window is None:
            raise ValueError(
                "Replay crosses a source gap, training cutoff, or one-year limit"
            )
        start, stop, weights, duration, origin_start, next_start = window
        return self._read(
            start,
            stop,
            weights,
            duration,
            device,
            origin_start=origin_start,
            next_start=next_start,
        )

    def _read(
        self,
        start: int,
        stop: int,
        weights: NDArray[np.float32],
        duration: int,
        device: torch.device,
        *,
        origin_start: int,
        next_start: int,
    ) -> ModelSample:
        interval = slice(start - HISTORY_STEPS, stop)
        z = self.store
        if self.planes is not None:
            values = np.empty(
                (stop - start + 19, len(self.names), *self.mask.shape[-2:]), np.float32
            )
            self.planes.read_into(
                list(range(start - 19, stop)), self.physical_variables, values
            )
        elif self.kind == "cm4":
            values = np.concatenate(
                [z[v][interval] for v in MODEL_VARIABLES]
                + [z["zos"][interval][:, None]],
                axis=1,
            )
        else:
            values = np.stack([z[n][interval] for n in self.names], axis=1)
        fields = normalize_physical_fields(
            values, self.mask, self.grid["mean"], self.grid["std"]
        )
        if self.era5 is not None:
            forcing = self.era5.load(start - HISTORY_STEPS, stop)
        else:
            forcing = np.stack(
                [z[n][interval] for n in MODEL_FORCING_VARIABLES], axis=1
            )
            if not np.isfinite(forcing[:, :, self.mask[TEMPERATURE_OFFSET]]).all():
                raise ValueError("Nonfinite own-model forcing on wet support")
            forcing = np.where(
                self.mask[TEMPERATURE_OFFSET],
                (forcing - self.forcing_mean[:, None, None])
                / self.forcing_std[:, None, None],
                0,
            ).astype(np.float32)

        def tensor(value: ArrayLike) -> torch.Tensor:
            return torch.as_tensor(value, device=device)

        return dict(
            fields=tensor(fields)[None],
            forcing=tensor(forcing)[None],
            contexts=tensor(
                calendar_context(
                    self.lat, self.lon, self.midpoints[start - HISTORY_STEPS : stop]
                )
            )[None],
            mask=tensor(self.mask),
            weights=tensor(np.asarray(weights, dtype=np.float32)),
            duration=int(duration),
            first=str(self.bounds[start, 0]),
            start_lead=int(start - origin_start),
            end_lead=int(stop - origin_start),
            next_start_lead=int(next_start - origin_start),
            origin_index=int(origin_start),
        )
