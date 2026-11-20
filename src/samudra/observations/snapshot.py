# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed-cadence LLC/observation windows and objectives, independent of months.

Arrays use the existing ForecastSample contract, with history followed by future
frames. Ocean input history consists of surface temperature and SSH only. Daily
analysis products are assigned explicit slots by preparation; no interpolation or
repetition occurs here. ERA5 preparation owns the forcing interval convention.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import Field

from samudra.config_base import BaseConfig
from samudra.models.surface_initialized import SurfaceInitializedModel
from samudra.observations.constants import MODEL_VARIABLES, SURFACE_NAMES, ModelTask
from samudra.observations.losses import structured_visibility
from samudra.observations.sample_types import ForecastSample

LLC_STATE_NAMES = tuple(
    f"{variable}_{level}" for variable in MODEL_VARIABLES for level in range(51)
) + ("zos",)


class SnapshotTaskConfig(BaseConfig):
    """The first experiment has fixed six-hour dynamics and daily OBS labels."""

    step_hours: Literal[6] = 6
    history_steps: Literal[29] = 29
    forecast_steps: Literal[12] = 12
    observation_stride: Literal[4] = 4
    reconstruction_weight: float = Field(default=0.1, ge=0)
    completion_weight: float = Field(default=0.1, ge=0)

    def offsets(self) -> NDArray[np.timedelta64]:
        """Frame offsets relative to the initialized state at t, including t."""
        return np.arange(
            1 - self.history_steps, self.forecast_steps + 1
        ) * np.timedelta64(self.step_hours, "h")

    def target_steps(self, source: Literal["obs", "llc"]) -> NDArray[np.int64]:
        if source not in ("obs", "llc"):
            raise ValueError("Snapshot source must be obs or llc")
        stride = self.observation_stride if source == "obs" else 1
        return np.arange(stride, self.forecast_steps + 1, stride)


def eligible_origins(
    times: NDArray[np.datetime64],
    start: np.datetime64,
    end: np.datetime64,
    task: SnapshotTaskConfig,
    *,
    observation: bool,
    excluded: Sequence[tuple[np.datetime64, np.datetime64]] = (),
) -> NDArray[np.int64]:
    """Indices with complete contiguous six-hour windows in [start, end).

    Observation origins use midnight analysis slots, so future daily targets are
    steps 4, 8, 12. Exclusions are half-open intervals and cover history as well
    as labels. The caller supplies the timestamp index of the prepared six-hour
    archive, not the daily observation index or the native hourly LLC index.
    """
    times = np.asarray(times, dtype="datetime64[ns]")
    if (
        times.ndim != 1
        or np.isnat(times).any()
        or np.any(np.diff(times) <= np.timedelta64(0, "ns"))
    ):
        raise ValueError("Timestamps must be finite, unique and increasing")
    bounds = np.asarray(
        [start, end, *(value for pair in excluded for value in pair)],
        dtype="datetime64[ns]",
    )
    if np.isnat(bounds).any():
        raise ValueError("Split and exclusion bounds must be finite")
    if start >= end or any(a >= b for a, b in excluded):
        raise ValueError("Split and exclusion bounds must be increasing")
    indices = []
    for index in range(task.history_steps - 1, len(times) - task.forecast_steps):
        window = times[index - task.history_steps + 1 : index + task.forecast_steps + 1]
        if not np.array_equal(window, times[index] + task.offsets()):
            continue
        if window[0] < start or window[-1] >= end:
            continue
        if observation and times[index] != times[index].astype("datetime64[D]"):
            continue
        if any(window[0] < b and window[-1] >= a for a, b in excluded):
            continue
        indices.append(index)
    return np.asarray(indices, dtype=np.int64)


def masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    wet_area: torch.Tensor,
    *,
    allow_empty: bool = False,
) -> torch.Tensor:
    """Equal-weight channel MSE after wet-area reduction, ignoring missing cells.

    Each channel is averaged over its available batch/time slots, then channels
    are averaged. Missing slots never contribute a zero error. ``wet_area`` is
    channel/y/x, or any shape broadcastable to the field arrays.
    """
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("Prediction, target and validity shapes must agree")
    if not torch.isfinite(wet_area).all() or (wet_area < 0).any():
        raise ValueError("Wet areas must be finite and nonnegative")
    support = valid.bool() & (wet_area > 0)
    if not torch.isfinite(target[support]).all():
        raise ValueError("Nonfinite target on declared valid support")
    if not torch.isfinite(prediction[support]).all():
        raise FloatingPointError("Nonfinite prediction on scored support")
    error = torch.where(support, prediction.float() - target.float(), 0).square()
    weight = support * wet_area
    denominator = weight.sum((-2, -1))
    present = denominator > 0
    mse = (error * weight).sum((-2, -1)) / denominator.clamp_min(1e-12)
    count = present.sum((0, 1))
    if not allow_empty and (count == 0).any():
        raise ValueError("Every scored channel needs at least one target slot")
    channels = (mse * present).sum((0, 1)) / count.clamp_min(1)
    return channels[count > 0].mean() if (count > 0).any() else error.sum() * 0


def snapshot_objective(
    model: SurfaceInitializedModel,
    sample: ForecastSample,
    wet_area: torch.Tensor,
    source: Literal["obs", "llc"],
    seed: int,
    task: SnapshotTaskConfig = SnapshotTaskConfig(),
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """Forecast every six hours; score OBS only daily, without monthly branches.

    The loader must supply finite normalized LLC fields on the physical wet mask,
    and original observation availability in ``validity``. Artificial corruption
    affects initializer inputs only. Forecast targets never enter initialization.
    """
    if source not in ("obs", "llc"):
        raise ValueError("Snapshot source must be obs or llc")
    history = task.history_steps
    if model.history_steps != history or tuple(model.names) != LLC_STATE_NAMES:
        raise ValueError(
            "Snapshot task requires 29-frame history and all 205 LLC fields"
        )
    if model.forcing_mode != "shared_era5":
        raise ValueError("Snapshot task requires shared eight-field ERA5 conditioning")
    if sample["surface"].shape[1] != history + task.forecast_steps:
        raise ValueError("Snapshot sample requires history plus 12 forecast frames")
    if sample["surface"].shape != sample["validity"].shape:
        raise ValueError("Surface availability must match surface frames")
    if (
        sample["atmosphere"].shape[1] != sample["surface"].shape[1]
        or sample["contexts"].shape[1] != sample["surface"].shape[1]
    ):
        raise ValueError("Forcing and contexts must span the complete window")
    if wet_area.shape != sample["mask"].shape:
        raise ValueError("Snapshot wet areas must match the channel/y/x physical mask")
    surface_indices = [model.names.index(name) for name in SURFACE_NAMES]
    available = sample["validity"].bool() & sample["mask"][surface_indices].bool()
    visible = structured_visibility(available[:, :history], seed)
    inputs = torch.where(visible, sample["surface"][:, :history], 0)
    model_task: ModelTask = "observation" if source == "obs" else "llc"
    initial = model.initialize(
        inputs,
        sample["atmosphere"][:, :history],
        sample["contexts"][:, history - 1],
        sample["mask"],
        visible,
        model_task,
    )
    prediction, final = model.continue_forecast(
        initial,
        sample["atmosphere"][:, history:],
        sample["contexts"][:, history:],
        sample["mask"],
        task=model_task,
    )
    area = wet_area * sample["mask"].bool()
    parts = {
        "completion": masked_mse(
            initial[:, :, surface_indices],
            sample["surface"][:, history - 2 : history],
            available[:, history - 2 : history] & ~visible[:, -2:],
            area[surface_indices],
            allow_empty=True,
        )
    }
    if source == "obs":
        steps = task.target_steps(source) - 1
        selected = torch.as_tensor(steps, device=prediction.device)
        parts["forecast"] = masked_mse(
            prediction[:, selected][:, :, surface_indices],
            sample["surface"][:, history + selected],
            available[:, history + selected],
            area[surface_indices],
        )
    else:
        groups = []
        interior = []
        for variable in (*MODEL_VARIABLES, "zos"):
            indices = [
                i
                for i, name in enumerate(model.names)
                if name == variable or name.startswith(variable + "_")
            ]
            valid = (
                sample["mask"][indices]
                .bool()
                .expand_as(sample["target_fields"][:, :, indices])
            )
            groups.append(
                masked_mse(
                    prediction[:, :, indices],
                    sample["target_fields"][:, :, indices],
                    valid,
                    area[indices],
                )
            )
            inferred = [i for i in indices if i not in surface_indices]
            if inferred:
                initial_target = sample["initial_fields"][:, :, inferred]
                interior.append(
                    masked_mse(
                        initial[:, :, inferred],
                        initial_target,
                        sample["mask"][inferred].bool().expand_as(initial_target),
                        area[inferred],
                    )
                )
        parts["forecast"] = torch.stack(groups).mean()
        parts["reconstruction"] = torch.stack(interior).mean()
    loss = parts["forecast"] + task.completion_weight * parts["completion"]
    if source == "llc":
        loss = loss + task.reconstruction_weight * parts["reconstruction"]
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite snapshot objective")
    return loss, parts, final.detach()
