# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Annual observation scoring with Samudra's spatial-RMSE-first reduction.

Predictions and targets enter in physical units. The selection metric divides
per-field RMSE by the observation-fitted training standard deviation, never by
climatology error. Because SST/SSH are five-day means and T/S are monthly means,
we average available times within each channel before averaging channels equally.
This equals Samudra's channel/time mean for fully observed, equal-cadence data.

Only 29 channels select a checkpoint: SST, SSH, temperature depths 1..13 and
salinity depths 0..13. Monthly temperature depth 0 is an optional diagnostic,
not a second SST vote. Velocities and unobserved deep layers are never scored.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike

from samudra.aggregator.metrics import area_weighted_rmse

SURFACE_INDICES = (38, 76)
PROFILE_INDICES = tuple(range(38, 52)) + tuple(range(57, 71))
SURFACE_NAMES = ("thetao_0", "zos")
PROFILE_NAMES = tuple(f"thetao_{i}" for i in range(14)) + tuple(
    f"so_{i}" for i in range(14)
)


class ObservedChannelRMSE:
    """Stream raw fields shaped ``[..., channel, y, x]`` with explicit support.

    Missing targets do not contribute. Nonfinite predictions on supported target
    cells raise, rather than silently shrinking the scored observation footprint.
    Spatial RMSE uses the same area-weighted kernel as Samudra validation.
    """

    def __init__(self, names: Sequence[str], std: torch.Tensor, area: torch.Tensor):
        self.names = tuple(names)
        if not self.names or len(set(self.names)) != len(self.names):
            raise ValueError("Channel names must be nonempty and unique")
        self.std = torch.as_tensor(std).detach().flatten()
        self.area = torch.as_tensor(area).detach()
        if self.std.shape != (len(self.names),):
            raise ValueError(
                "Need one observation-fitted standard deviation per channel"
            )
        if not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError("Standard deviations must be finite and positive")
        if self.area.ndim != 2 or not torch.isfinite(self.area).all():
            raise ValueError("Area must be a finite two-dimensional array")
        if (self.area < 0).any() or not (self.area > 0).any():
            raise ValueError("Area must be nonnegative with positive total support")
        self._sum = torch.zeros(len(self.names), dtype=torch.float64)
        self._count = torch.zeros(len(self.names), dtype=torch.int64)
        self._cells = torch.zeros(len(self.names), dtype=torch.int64)

    @torch.no_grad()
    def record(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        support: torch.Tensor | None = None,
    ) -> None:
        prediction = torch.as_tensor(prediction).detach()
        target = torch.as_tensor(target, device=prediction.device)
        if prediction.shape != target.shape or prediction.ndim < 3:
            raise ValueError(
                "Prediction and target must share shape [..., channel, y, x]"
            )
        if prediction.shape[-3:] != (len(self.names), *self.area.shape):
            raise ValueError("Prediction channels/grid do not match metric definition")
        present = torch.isfinite(target) & (self.area.to(prediction.device) > 0)
        if support is not None:
            support = torch.as_tensor(support, device=prediction.device)
            if support.dtype != torch.bool:
                raise ValueError("Observation support must be boolean")
            present &= support
        if (~torch.isfinite(prediction) & present).any():
            raise ValueError("Nonfinite prediction at a supported observation target")
        error = (prediction - target).masked_fill(~present, float("nan"))
        rmse = area_weighted_rmse(torch.zeros_like(error), error, self.area)
        rmse = rmse.reshape(-1, len(self.names))
        available = present.flatten(-2).any(-1).reshape(-1, len(self.names))
        if not torch.isfinite(rmse[available]).all():
            raise ValueError("Nonfinite RMSE at a supported observation target")
        self._sum += rmse.masked_fill(~available, 0).sum(0).double().cpu()
        self._count += available.sum(0).cpu()
        self._cells += (
            present.reshape(-1, len(self.names), *self.area.shape).sum((0, 2, 3)).cpu()
        )

    def means(self, required: Sequence[int] | None = None) -> torch.Tensor:
        indices = list(range(len(self.names))) if required is None else list(required)
        missing = [self.names[i] for i in indices if self._count[i] == 0]
        if missing:
            raise ValueError(f"No observation support for scored channels: {missing}")
        return self._sum / self._count

    def logs(
        self, prefix: str, required: Sequence[int] | None = None
    ) -> dict[str, float]:
        raw = self.means(required)
        indices = list(range(len(self.names))) if required is None else list(required)
        logs = {
            f"{prefix}/normalized_rmse/channel_mean": float(
                (raw / self.std.double().cpu())[indices].mean()
            )
        }
        for i, name in enumerate(self.names):
            if self._count[i] > 0:
                logs[f"{prefix}/weighted_rmse/{name}"] = float(raw[i])
                logs[f"{prefix}/normalized_rmse/{name}"] = float(
                    raw[i] / self.std[i].cpu()
                )
            logs[f"{prefix}/target_times/{name}"] = int(self._count[i])
            logs[f"{prefix}/target_cells/{name}"] = int(self._cells[i])
        return logs


def complete_month_weights(
    interval_starts: ArrayLike, interval_ends: ArrayLike
) -> tuple[list[str], np.ndarray]:
    """Exact calendar-overlap averaging, dropping partial boundary months.

    Every prediction represents the mean over its supplied half-open interval.
    Splitting such a mean at a month boundary is a piecewise-constant assumption,
    matching the prepared five-day target operator; no daily detail is invented.
    """
    starts = np.asarray(interval_starts, dtype="datetime64[ns]")
    ends = np.asarray(interval_ends, dtype="datetime64[ns]")
    if starts.ndim != 1 or not len(starts) or starts.shape != ends.shape:
        raise ValueError("Need matching nonempty one-dimensional interval arrays")
    if np.isnat(starts).any() or np.isnat(ends).any():
        raise ValueError("Intervals cannot contain missing dates")
    if (ends <= starts).any() or (starts[1:] != ends[:-1]).any():
        raise ValueError("Prediction intervals must be positive and contiguous")
    months, rows = [], []
    first = starts[0].astype("datetime64[M]")
    stop = ends[-1].astype("datetime64[M]") + np.timedelta64(1, "M")
    for month in np.arange(first, stop, dtype="datetime64[M]"):
        begin = month.astype("datetime64[ns]")
        end = (month + np.timedelta64(1, "M")).astype("datetime64[ns]")
        if begin < starts[0] or end > ends[-1]:
            continue
        overlap = np.maximum(
            (np.minimum(ends, end) - np.maximum(starts, begin)).astype(np.int64), 0
        )
        rows.append(overlap / float((end - begin).astype(np.int64)))
        months.append(str(month))
    return months, np.asarray(rows, dtype=np.float64).reshape(len(rows), len(starts))


class AnnualObservationEvaluator:
    """Consume one continuous year's raw predictions, one five-day step at a time.

    ``trajectory`` is returned by ``Samples.trajectory``. All target values are
    raw; its monthly matrix must equal exact calendar overlaps. The caller owns
    autoregression and must initialize only once, with no later ocean resets.
    Finalization refuses partial years and any scored channel without targets.
    """

    def __init__(
        self,
        trajectory: Mapping[str, Any],
        state_std: torch.Tensor,
        area_weights: torch.Tensor,
        *,
        surface_mask: torch.Tensor | None = None,
        profile_mask: torch.Tensor | None = None,
    ):
        self._starts = np.asarray(trajectory["interval_starts"], dtype="datetime64[ns]")
        self._ends = np.asarray(trajectory["interval_ends"], dtype="datetime64[ns]")
        months, matrix = complete_month_weights(self._starts, self._ends)
        if len(self._starts) != 73 or not np.all(
            self._ends - self._starts == np.timedelta64(5, "D")
        ):
            raise ValueError(
                "Annual validation requires 73 contiguous five-day intervals"
            )
        if self._starts[0] < np.datetime64("2013-08-05") or self._ends[
            -1
        ] > np.datetime64("2015-01-01"):
            raise ValueError(
                "Validation targets must be after training and strictly before 2015"
            )
        if [str(m)[:7] for m in trajectory["months"]] != months:
            raise ValueError(
                "Monthly targets must cover exactly the complete forecast months"
            )
        given_matrix = (
            torch.as_tensor(trajectory["month_matrix"]).detach().cpu().numpy()
        )
        if given_matrix.shape != matrix.shape or not np.allclose(
            given_matrix, matrix, atol=1e-7, rtol=1e-6
        ):
            raise ValueError(
                "Monthly averaging weights do not match calendar intervals"
            )
        self._matrix = torch.as_tensor(matrix, dtype=torch.float32)
        self._surface_targets = torch.as_tensor(trajectory["raw_surface"])
        self._profile_targets = torch.as_tensor(trajectory["monthly_targets"])
        shape = self._surface_targets.shape
        if len(shape) != 5 or shape[1:3] != (73, 2):
            raise ValueError("raw_surface must have shape [batch, 73, 2, y, x]")
        if self._profile_targets.shape != (shape[0], len(months), 28, *shape[-2:]):
            raise ValueError(
                "monthly_targets must have shape [batch, months, 28, y, x]"
            )
        self._std = torch.as_tensor(state_std).flatten()
        if self._std.shape != (77,):
            raise ValueError("Expected the canonical 77-channel state normalization")
        self._surface = ObservedChannelRMSE(
            SURFACE_NAMES, self._std[list(SURFACE_INDICES)], area_weights
        )
        self._profile = ObservedChannelRMSE(
            PROFILE_NAMES, self._std[list(PROFILE_INDICES)], area_weights
        )
        self._surface_mask = surface_mask
        self._profile_mask = profile_mask
        self._monthly_prediction: torch.Tensor | None = None
        self._recorded_months: set[int] = set()
        self._step = 0

    @torch.no_grad()
    def record(self, prediction_raw: torch.Tensor) -> None:
        """Record the next five-day forecast; prediction shape is [B, 77, Y, X]."""
        if self._step == 73:
            raise ValueError("Annual rollout already contains all 73 predictions")
        prediction = torch.as_tensor(prediction_raw).detach()
        expected = (
            self._surface_targets.shape[0],
            77,
            *self._surface_targets.shape[-2:],
        )
        if prediction.shape != expected:
            raise ValueError(
                f"Expected raw prediction shape {expected}, got {prediction.shape}"
            )
        self._surface.record(
            prediction[:, list(SURFACE_INDICES)],
            self._surface_targets[:, self._step],
            self._surface_mask,
        )
        if self._monthly_prediction is None:
            self._monthly_prediction = torch.zeros(
                self._profile_targets.shape,
                dtype=prediction.dtype,
                device=prediction.device,
            )
        # Add only overlapping months: multiplying NaN by a zero weight would
        # incorrectly poison every month of the rollout.
        for month in torch.nonzero(self._matrix[:, self._step]).flatten().tolist():
            self._monthly_prediction[:, month] += (
                prediction[:, list(PROFILE_INDICES)]
                * self._matrix[month, self._step].item()
            )
            if not (self._matrix[month, self._step + 1 :] > 0).any():
                self._profile.record(
                    self._monthly_prediction[:, month],
                    self._profile_targets[:, month],
                    self._profile_mask,
                )
                self._recorded_months.add(month)
        self._step += 1

    def result(self, label: str = "rollout_val/365d") -> dict[str, float]:
        if self._step != 73 or len(self._recorded_months) != len(self._matrix):
            raise ValueError(
                "Cannot select a checkpoint from an incomplete annual rollout"
            )
        surface_raw = self._surface.means()
        profile_raw = self._profile.means(required=range(1, 28))
        normalized = torch.cat(
            (
                surface_raw / self._surface.std.cpu(),
                (profile_raw / self._profile.std.cpu())[1:],
            )
        )
        logs = self._surface.logs(label)
        logs.update(self._profile.logs(f"{label}/monthly", required=range(1, 28)))
        # Keep canonical Samudra per-channel raw metric names for scored fields.
        for i, name in enumerate(PROFILE_NAMES[1:], 1):
            logs[f"{label}/weighted_rmse/{name}"] = float(profile_raw[i])
        logs[f"{label}/normalized_rmse/channel_mean"] = float(normalized.mean())
        logs[f"{label}/scored_channels"] = 29
        logs[f"{label}/forecast_steps"] = self._step
        logs[f"{label}/complete_months"] = len(self._recorded_months)
        return logs


def evaluate_annual_predictions(
    prediction_raw: torch.Tensor,
    trajectory: Mapping[str, Any],
    state_std: torch.Tensor,
    area_weights: torch.Tensor,
    **kwargs: Any,
) -> dict[str, float]:
    """Convenience wrapper for an already-collected [B, 73, 77, Y, X] rollout."""
    if prediction_raw.ndim != 5:
        raise ValueError("Expected predictions with shape [batch, time, 77, y, x]")
    evaluator = AnnualObservationEvaluator(
        trajectory, state_std, area_weights, **kwargs
    )
    for step in range(prediction_raw.shape[1]):
        evaluator.record(prediction_raw[:, step])
    return evaluator.result()
