"""Autoregressive rollout validation.

Ported from the Samudra `RolloutValidationAggregator`
(https://github.com/m2lines/Samudra/pull/770, "validation rollout sliced") and
reshaped for this codebase. Instead of per-variable depth-band tables, each
rollout horizon reports a single mean loss -- computed with the same loss
function as one-step validation, so the two are directly comparable -- plus an
RMSE-vs-rollout-step curve so error compounding over the rollout is visible.
"""

import dataclasses
import logging
import random

import torch

from ocean_emulators.aggregator.metrics import weighted_mean
from ocean_emulators.aggregator.plotting import plot_metric_by_rollout_step
from ocean_emulators.utils.loss import LossFn
from ocean_emulators.utils.output import ModelInferenceOutput
from ocean_emulators.utils.wandb import Metrics, MetricsDict

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class RolloutWindow:
    """One autoregressive validation run over the validation time axis."""

    start_index: int
    """Index of the run's first input timestep, relative to the start of val_time."""

    num_timesteps: int
    """Timesteps the run consumes: its input history plus all of its targets."""


@dataclasses.dataclass(frozen=True)
class RolloutValidationPlan:
    """Where the runs of one autoregressive validation start, and how long they are."""

    label: str
    num_steps: int
    """Autoregressive steps every run takes. Shared so runs can be averaged."""

    windows: tuple[RolloutWindow, ...]

    @property
    def enabled(self) -> bool:
        return bool(self.windows) and self.num_steps > 0


def _empty_plan(label: str) -> RolloutValidationPlan:
    return RolloutValidationPlan(label=label, num_steps=0, windows=())


def plan_rollout_windows(
    *,
    label: str,
    total_timesteps: int,
    hist: int,
    num_steps: int,
    num_runs: int,
    seed: int,
) -> RolloutValidationPlan:
    """Lay out autoregressive validation runs inside val_time.

    When the runs fit without overlapping, the validation range is cut into
    `num_runs` equal blocks and one run is placed at a random offset inside its
    own block. That spreads the initial conditions around rather than pinning
    them to block starts, and the offsets come from an explicitly seeded
    generator, so the same initial conditions are scored every epoch. That is
    what makes the per-epoch mean loss comparable across epochs for checkpoint
    selection, and what lets the per-epoch RMSE-vs-step curves be read as one
    series.

    When they do not fit, all `num_runs` runs are still scored: their starts are
    spread evenly from the beginning of val_time to the latest start that still
    ends at its end. That overlaps the runs, which is a better trade than
    dropping validation, but it does mean their errors are partly correlated.

    `num_steps` is shortened, with a warning, only when val_time cannot fit even
    a single run.

    Args:
        label: Name of this validation, used only in log messages.
        total_timesteps: Timesteps available in the validation time range.
        hist: Number of extra history timesteps per model step.
        num_steps: Requested autoregressive steps per run.
        num_runs: Requested number of runs.
        seed: Seed for the within-block offsets.

    Returns:
        The resolved plan. `RolloutValidationPlan.enabled` is False when
        val_time cannot fit even a single step.
    """
    if num_steps <= 0 or num_runs <= 0:
        return _empty_plan(label)

    # One model step consumes `hist + 1` timesteps, and a run also needs its
    # own input history before the first target.
    stride = hist + 1

    def span_for(steps: int) -> int:
        return stride * (steps + 1)

    span = span_for(num_steps)
    max_runs = total_timesteps // span
    if max_runs == 0:
        fitted_steps = total_timesteps // stride - 1
        if fitted_steps < 1:
            logger.warning(
                f"Skipping {label} autoregressive validation: val_time has "
                f"{total_timesteps} timesteps, which is not enough for one "
                f"autoregressive step at hist={hist}."
            )
            return _empty_plan(label)
        logger.warning(
            f"{label} autoregressive validation requested {num_steps} steps, but "
            f"val_time only has {total_timesteps} timesteps. Shortening to "
            f"{fitted_steps} steps. Widen val_time to score the full horizon."
        )
        num_steps = fitted_steps
        span = span_for(num_steps)
        max_runs = 1

    if num_runs <= max_runs:
        rng = random.Random(seed)
        block = total_timesteps // num_runs
        slack = block - span
        starts = [run * block + rng.randint(0, slack) for run in range(num_runs)]
    else:
        # No room to keep them apart, so spread them from the first possible
        # start to the last one that still ends at the end of val_time.
        last_start = total_timesteps - span
        # Deduplicated: when a run already fills val_time there is only one
        # window to score, and repeating it would score the same rollout twice.
        starts = sorted(
            {round(run * last_start / (num_runs - 1)) for run in range(num_runs)}
        )
        logger.warning(
            f"{label} autoregressive validation requested {num_runs} runs of "
            f"{num_steps} steps, but only {max_runs} fit without overlapping in "
            f"the {total_timesteps} timesteps of val_time. Running {len(starts)} "
            f"with overlap; widen val_time to keep them independent."
        )

    windows = tuple(
        RolloutWindow(start_index=start, num_timesteps=span) for start in starts
    )
    logger.info(
        f"{label} autoregressive validation plan: {len(windows)} run(s) of "
        f"{num_steps} steps, starting at val_time indices {starts}."
    )
    return RolloutValidationPlan(label=label, num_steps=num_steps, windows=windows)


class RolloutValidationAggregator:
    """Aggregates per-rollout-step loss and RMSE over autoregressive validation runs.

    Every run contributes one value per rollout step, and the runs are averaged
    step by step. That keeps the reported curve a single line per epoch no
    matter how many initial conditions were rolled out.

    Metrics are not reduced across ranks: the rollouts run on the main process
    only, which is where wandb logging and checkpoint selection happen.
    """

    def __init__(
        self,
        *,
        num_steps: int,
        area_weights: torch.Tensor,
        wet: torch.Tensor,
        loss_fn: LossFn,
        device: torch.device,
        ownership: torch.Tensor | None = None,
    ):
        """`ownership` is a `[tile, 1, lat, lon]` indicator for grouped rollouts.

        A group's tiles overlap, so scoring each of them in full would weight the
        shared cells twice and bias every metric toward the seams. The indicator
        gives each cell exactly one owner. Without it a rollout is a single
        field, and everything here behaves exactly as it did before.
        """
        if num_steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        self._num_steps = num_steps
        self._area_weights = area_weights
        self._wet = wet.bool()
        self._loss_fn = loss_fn
        self._ownership = ownership
        self._loss_sum = torch.zeros(num_steps, device=device)
        self._rmse_sum = torch.zeros(num_steps, device=device)
        self._n_runs = 0

    @property
    def n_runs(self) -> int:
        return self._n_runs

    @torch.no_grad()
    def record_run(self, data: ModelInferenceOutput, *, step_offset: int = 0) -> None:
        """Record one chunk of a single rollout, starting at `step_offset`.

        `data.prediction` and `data.target` are `[step, channel, lat, lon]`, the
        shape `BaseModel.inference` returns. A run is recorded across as many
        calls as it took chunks to roll out; `finish_run` closes it.
        """
        prediction = data.prediction
        target = data.target
        if prediction.shape != target.shape:
            raise RuntimeError(
                f"prediction and target must have the same shape, got "
                f"{prediction.shape} and {target.shape}"
            )
        num_steps = prediction.shape[0]
        if step_offset + num_steps > self._num_steps:
            raise ValueError(
                f"Recording steps {step_offset}..{step_offset + num_steps - 1} "
                f"exceeds the planned {self._num_steps} rollout steps."
            )

        for step in range(num_steps):
            # A grouped rollout carries a tile axis, so the step slice is already
            # a batch of tiles. Ungrouped, keep the batch dimension: `loss_fn`
            # broadcasts its wet mask over it.
            if prediction.ndim == 5:
                gen = prediction[step]
                label = target[step]
                loss = torch.mean(
                    self._loss_fn(gen, label, sample_weight=self._ownership)
                ).detach()
            else:
                gen = prediction[step : step + 1]
                label = target[step : step + 1]
                loss = torch.mean(self._loss_fn(gen, label)).detach()
            self._loss_sum[step_offset + step] += loss.to(self._loss_sum.device)
            rmse = self._area_weighted_rmse(gen, label)
            self._rmse_sum[step_offset + step] += rmse.to(self._rmse_sum.device)

    def finish_run(self) -> None:
        """Close the run whose chunks were just recorded."""
        self._n_runs += 1

    @torch.no_grad()
    def _area_weighted_rmse(
        self, gen: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Channel-mean, area-weighted RMSE for one rollout step.

        Land is masked to NaN rather than left at the normalized fill value so
        it does not dilute the ocean error, matching how the one-step
        aggregators build their dicts.
        """
        squared_error = (gen - target).square()
        if self._ownership is not None and squared_error.shape[0] > 1:
            # Every cell has one owner, so summing the owned parts of each tile
            # covers the domain exactly once. Disowned cells go to NaN alongside
            # land so neither dilutes the mean.
            owned = self._ownership.to(squared_error.device) > 0
            squared_error = torch.where(
                self._wet.unsqueeze(0) & owned, squared_error, torch.nan
            )
            rmse_per_channel = torch.stack(
                [
                    weighted_mean(squared_error[tile], self._area_weights)
                    for tile in range(squared_error.shape[0])
                ]
            ).nanmean(dim=0).sqrt()
            return rmse_per_channel.nanmean().detach()

        squared_error = squared_error.squeeze(0)
        squared_error = torch.where(self._wet, squared_error, torch.nan)
        rmse_per_channel = weighted_mean(squared_error, self._area_weights).sqrt()
        return rmse_per_channel.nanmean().detach()

    def _mean_by_step(self, total: torch.Tensor) -> torch.Tensor:
        if self._n_runs == 0:
            raise ValueError("No autoregressive validation runs have been recorded.")
        return total / self._n_runs

    def loss_by_step(self) -> torch.Tensor:
        """Loss at each rollout step, averaged over runs."""
        return self._mean_by_step(self._loss_sum)

    def rmse_by_step(self) -> torch.Tensor:
        """Area-weighted RMSE at each rollout step, averaged over runs."""
        return self._mean_by_step(self._rmse_sum)

    def mean_loss(self) -> float:
        """Mean loss over every rollout step of every run."""
        return float(self.loss_by_step().mean().cpu())

    @torch.no_grad()
    def get_logs(self, label: str) -> Metrics:
        """Return the wandb/log metrics for this rollout horizon.

        Args:
            label: Horizon name, e.g. `short` or `long`.
        """
        rmse_by_step = self.rmse_by_step().cpu().numpy()
        caption = (
            f"{label} autoregressive validation: area-weighted RMSE (normalized "
            f"units, channel mean) against rollout step, averaged over "
            f"{self._n_runs} initial condition(s)."
        )
        logs: MetricsDict = {
            f"val/mean/{label}-autoregressive-loss": self.mean_loss(),
            f"val/autoregressive-loss/{label}": plot_metric_by_rollout_step(
                rmse_by_step,
                title=f"{label} autoregressive validation RMSE",
                caption=caption,
            ),
        }
        return logs
