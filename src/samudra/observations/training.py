# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation source windows, objectives and monthly replay training.

The optimizer loop remains separate from the epoch-based Samudra Trainer because
its samples combine distinct observation cadences and cached forecast states.
"""

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import torch

from samudra.models.surface_initialized import SurfaceInitializedModel
from samudra.observations.archive import SPLITS, validate_normalization_policy
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.constants import (
    FORECAST_STEPS,
    HISTORY_STEPS,
    INITIAL_STATE_START,
    LAST_HISTORY_STEP,
    MAX_REPLAY_MONTHS,
    STEP_DAYS,
    SURFACE_CHANNELS,
)
from samudra.observations.evaluate import AnnualObservationEvaluator
from samudra.observations.intervals import replay_month
from samudra.observations.losses import balanced_loss, completion_loss, corrupt_sample
from samudra.observations.model_data import ModelMonths
from samudra.observations.replay import ReplayEntry, states_at
from samudra.observations.samples import Samples
from samudra.observations.state import TrainingState, write_json
from samudra.utils.device import autocast
from samudra.utils.train import CheckpointPaths


class ReplayLayout(TypedDict):
    """Five-day bin offsets from the original forecast origin."""

    start_lead: int
    end_lead: int
    next_start_lead: int


class ForecastSample(TypedDict, total=False):
    """A source window; tensors have batch/time/channel/latitude/longitude axes.

    Observation windows carry surface and monthly profile labels; model windows
    additionally carry dense physical fields. Validity stays separate from values.
    """

    surface: torch.Tensor
    atmosphere: torch.Tensor
    contexts: torch.Tensor
    validity: torch.Tensor
    mask: torch.Tensor
    month_weights: torch.Tensor
    fields: torch.Tensor
    interior: torch.Tensor
    raw_surface: torch.Tensor
    completion_target: torch.Tensor
    completion_valid: torch.Tensor
    start_lead: int
    end_lead: int
    next_start_lead: int


Origin = str | int  # Observation calendar month, or index into native model windows.
Metrics = dict[str, float | int]
Manifest = dict[
    str, Any
]  # JSON provenance includes nested hashes and runtime metadata.

SELECTION_METRIC = "rollout_val/365d/normalized_rmse/channel_mean"


class Sources:
    """One observation normalization shared by the explicitly opened sources."""

    def __init__(
        self, observations: Samples, models: Mapping[str, ModelMonths]
    ) -> None:
        self.obs, self.models = observations, models
        self.paths = observations.paths("train")
        expected = [str(m) for m in pd.period_range(*SPLITS["train"], freq="M")]
        if [p.stem for p in self.paths] != expected:
            raise ValueError("Observation training cohort is incomplete")
        self.names = ("obs", *models)

    def fresh_origin(self, source: str, rng: np.random.Generator) -> Origin:
        if source == "obs":
            return self.paths[int(rng.integers(len(self.paths)))].stem
        return int(rng.integers(len(self.models[source].windows)))

    def layout(self, source: str, origin: Origin, month: int) -> ReplayLayout | None:
        if month >= MAX_REPLAY_MONTHS:
            return None
        if source != "obs":
            return self.models[source].continuation_layout(int(origin), month)
        target = pd.Timestamp(origin) + pd.DateOffset(months=month)
        if not SPLITS["train"][0] <= target.strftime("%Y-%m") <= SPLITS["train"][1]:
            return None
        start, stop, next_start = replay_month(origin, target)
        if stop > FORECAST_STEPS:
            return None
        return dict(start_lead=start, end_lead=stop, next_start_lead=next_start)

    def load(self, source: str, origin: Origin, month: int = 0) -> ForecastSample:
        if source == "obs":
            target = pd.Timestamp(origin) + pd.DateOffset(months=month)
            if month:
                sample = self.obs.trajectory(origin, target_month=target, split="train")
            else:
                sample = self.obs.load(
                    self.obs.root / "train" / (target.strftime("%Y-%m") + ".npz")
                )
                layout = self.layout(source, origin, month)
                if layout is None:
                    raise ValueError(
                        "Observation window is outside the training cohort"
                    )
                sample.update(layout)
            sample["mask"] = self.obs.mask
            return sample
        raw = self.models[source].continuation(int(origin), month, self.obs.device)
        fields = raw["fields"]
        return {
            **raw,
            "surface": fields[:, :, SURFACE_CHANNELS],
            "atmosphere": raw["forcing"],
            "month_weights": raw["weights"],
            "validity": raw["mask"][SURFACE_CHANNELS]
            .bool()
            .expand_as(fields[:, :, SURFACE_CHANNELS]),
        }

    def auxiliary(
        self, source: str, origin: Origin, month: int, sample: ForecastSample
    ) -> ForecastSample:
        if source != "obs" or month == 0:
            return sample
        target = pd.Timestamp(origin) + pd.DateOffset(months=month)
        result = self.obs.load(
            self.obs.root / "train" / (target.strftime("%Y-%m") + ".npz")
        )
        result["mask"] = self.obs.mask
        return result


def objective(
    model: SurfaceInitializedModel,
    data: Samples,
    source: str,
    sample: ForecastSample,
    auxiliary: ForecastSample,
    seed: int,
    *,
    cached: torch.Tensor | None = None,
    coverage: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """One forecast window plus separate initializer reconstruction/completion.

    Replay never sees the auxiliary branch's contemporaneous ocean observations.
    Only its own detached physical history, future forcing and calendar advance it.
    """
    task = "observation" if source == "obs" else "om4"
    auxiliary = corrupt_sample(auxiliary, seed)
    if source != "obs":
        if coverage is None:
            raise ValueError("Model initialization needs observation coverage")
        visible = auxiliary["validity"].clone()
        visible[:, :HISTORY_STEPS] &= coverage[:, :HISTORY_STEPS].bool()
        auxiliary["validity"] = visible
        auxiliary["surface"] = torch.where(visible, auxiliary["surface"], 0)
        auxiliary["completion_valid"] = (
            sample["mask"][SURFACE_CHANNELS]
            .bool()
            .expand_as(visible[:, INITIAL_STATE_START:HISTORY_STEPS])
            & ~visible[:, INITIAL_STATE_START:HISTORY_STEPS]
        )
    args = (
        auxiliary["surface"],
        auxiliary["atmosphere"],
        auxiliary["contexts"],
        auxiliary["mask"],
        auxiliary["validity"],
    )
    if cached is None:
        prediction, initial = model.forecast(*args, task=task)
        replay_initial = initial
    else:
        replay_initial = cached.detach().to(data.device)
        prediction, _ = model.continue_forecast(
            replay_initial,
            sample["atmosphere"][:, HISTORY_STEPS:],
            sample["contexts"][:, HISTORY_STEPS:],
            sample["mask"],
            start_lead=sample["start_lead"] + 1,
            task=task,
        )
        initial = model.initialize(
            args[0][:, :HISTORY_STEPS],
            args[1][:, :HISTORY_STEPS],
            args[2][:, LAST_HISTORY_STEP],
            args[3],
            args[4][:, :HISTORY_STEPS],
            task,
        )
    parts = {
        "completion": completion_loss(
            initial[:, :, SURFACE_CHANNELS],
            auxiliary["completion_target"],
            auxiliary["completion_valid"],
            data.grid["lat"],
        )
    }
    if source == "obs":
        parts.update(data.forecast_losses(prediction, sample))
        reconstructed = model.reconstruct_month(*args, auxiliary["month_weights"])
        parts["reconstruction"] = data.interior_loss(reconstructed, auxiliary)
        forecast = 0.8 * parts["monthly_ts"] + 0.1 * parts["sst"] + 0.1 * parts["ssh"]
    else:
        weights = sample["mask"] * data.area
        parts["forecast"] = balanced_loss(
            prediction, sample["fields"][:, HISTORY_STEPS:], weights, model.names
        )
        parts["reconstruction"] = balanced_loss(
            initial,
            auxiliary["fields"][:, INITIAL_STATE_START:HISTORY_STEPS],
            weights,
            model.names,
            interior_only=True,
        )
        forecast = parts["forecast"]
    loss = forecast + 0.1 * parts["reconstruction"] + 0.1 * parts["completion"]
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Nonfinite {source} objective")
    boundary = sample["next_start_lead"] - sample["start_lead"]
    next_states = states_at(replay_initial, prediction, boundary).detach()
    return loss, parts, next_states


@torch.no_grad()
def annual_metrics(
    model: SurfaceInitializedModel, data: Samples, trajectory: ForecastSample
) -> dict[str, float]:
    model.eval()
    evaluator = AnnualObservationEvaluator(
        trajectory,
        data.std,
        data.area.expand(data.mask.shape[-2:]),
        surface_mask=data.mask[SURFACE_CHANNELS].bool(),
        profile_mask=data.ts_mask.bool(),
    )
    with autocast(torch.device(data.device).type == "cuda", torch.bfloat16):
        states = model.initialize(
            trajectory["surface"][:, :HISTORY_STEPS],
            trajectory["atmosphere"][:, :HISTORY_STEPS],
            trajectory["contexts"][:, LAST_HISTORY_STEP],
            data.mask,
            trajectory["validity"][:, :HISTORY_STEPS],
        )
        for step in range(FORECAST_STEPS):
            prediction, states = model.continue_forecast(
                states,
                trajectory["atmosphere"][
                    :, HISTORY_STEPS + step : HISTORY_STEPS + step + 1
                ],
                trajectory["contexts"][
                    :, HISTORY_STEPS + step : HISTORY_STEPS + step + 1
                ],
                data.mask,
                start_lead=step + 1,
            )
            evaluator.record(data.physical(prediction)[:, 0])
    return evaluator.result()


def train(
    cfg: ObservationTrainConfig,
    state: TrainingState,
    sources: Sources,
    output: Path,
    manifest: Manifest,
    validate: Callable[[SurfaceInitializedModel], dict[str, float]],
    emit: Callable[[Metrics], None],
    *,
    stop_after: int | None = None,
    stopping: Callable[[], bool] = lambda: False,
) -> None:
    """The complete scientific sequence is visible and editable here."""
    total = cfg.total_updates
    checkpoints = CheckpointPaths(output)
    limit = total if stop_after is None else min(stop_after, total)
    if limit < state.updates:
        raise ValueError("Stop limit precedes the restored checkpoint")
    for update in range(state.updates, limit):
        # Specialize to observations; keep Adam moments, reset EMA exactly once.
        if update == cfg.main_updates and not state.cooled:
            state.ema = state.new_ema()
            state.cooled = True
        source = cfg.source_at(update)
        pool = state.pools[source]
        for group in state.optimizer.param_groups:
            group["lr"] = cfg.learning_rate_at(update)
        state.model.train()
        state.optimizer.zero_grad(set_to_none=True)
        record: Metrics = {
            "lr": cfg.learning_rate_at(update),
            "max_start_lead_days": 0,
        }
        for example in range(cfg.gradient_accumulation_steps):
            entry = pool.draw(update) if example >= cfg.fresh_examples else None
            origin = (
                sources.fresh_origin(source, state.rng)
                if entry is None
                else entry.origin
            )
            month = 0 if entry is None else entry.month
            sample = sources.load(source, origin, month)
            if entry is not None and entry.start_lead != sample["start_lead"]:
                raise ValueError("Replay state time differs from target window")
            auxiliary = sources.auxiliary(source, origin, month, sample)
            coverage = None
            if source != "obs":
                coverage_origin = sources.fresh_origin("obs", state.rng)
                coverage = sources.load("obs", coverage_origin)["validity"]
            # A fresh signed-32-bit seed controls artificial gaps for this example.
            seed = int(state.rng.integers(2**31))
            with autocast(
                torch.device(sources.obs.device).type == "cuda",
                torch.bfloat16,
            ):
                loss, parts, next_states = objective(
                    state.model,
                    sources.obs,
                    source,
                    sample,
                    auxiliary,
                    seed,
                    cached=None if entry is None else entry.states,
                    coverage=coverage,
                )
            (loss / cfg.gradient_accumulation_steps).backward()
            exposure = "fresh" if entry is None else "replay"
            state.exposure[source][exposure] += 1
            record[exposure] = record.get(exposure, 0) + 1
            record["max_start_lead_days"] = max(
                record["max_start_lead_days"], sample["start_lead"] * STEP_DAYS
            )
            for name, value in {"loss": loss, **parts}.items():
                record[name] = (
                    record.get(name, 0.0)
                    + float(value.detach()) / cfg.gradient_accumulation_steps
                )
            following = sources.layout(source, origin, month + 1)
            if following is not None:
                if following["start_lead"] != sample["next_start_lead"]:
                    raise ValueError(
                        "Replay boundary differs from next month's input time"
                    )
                pool.add(
                    ReplayEntry(
                        source,
                        origin,
                        month + 1,
                        following["start_lead"],
                        update if entry is None else entry.origin_update,
                        next_states,
                    )
                )
        norm = torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), cfg.gradient_clip, error_if_nonfinite=True
        )
        state.optimizer.step()
        state.ema(state.model)
        state.updates = update + 1
        state.exposure[source]["updates"] += 1
        emit(
            {
                "updates": state.updates,
                "train/gradient_norm": float(norm),
                **{f"train/{source}/{key}": value for key, value in record.items()},
                **{
                    f"exposure/{name}/{key}": value
                    for name, counts in state.exposure.items()
                    for key, value in counts.items()
                },
            }
        )
        if state.updates % cfg.validation_freq == 0 or state.updates == total:
            with state.averaged_model():
                metrics = validate(state.model)
            score = metrics[SELECTION_METRIC]
            if not math.isfinite(score):
                raise FloatingPointError("Annual selection score is nonfinite")
            if score < state.best_score:
                state.best_score = score
                state.save(checkpoints.best_validation_checkpoint_path, manifest)
            emit({"updates": state.updates, **metrics})
        if state.updates == cfg.main_updates:
            state.save(output / "main.pt", manifest)
        if state.updates % cfg.save_freq == 0 or state.updates == limit or stopping():
            state.save(checkpoints.latest_checkpoint_path, manifest)
        if stopping():
            break
    if state.updates == total:
        state.save(output / "endpoint.pt", manifest)
        write_json(
            {"updates": total, "exposure": state.exposure}, output / "COMPLETE.json"
        )


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_normalization(directory: Path) -> None:
    """Read back fitted samples; a saved manifest alone is not proof."""
    proof = json.loads((directory / "normalization.json").read_text())
    validate_normalization_policy(proof)
    for path, expected in proof["sha256"].items():
        if digest(path) != expected:
            raise ValueError(f"Normalization input changed: {path}")


def archive_identity(root: Path) -> dict[str, str | int]:
    """Change detection, not a full content read-back of the daily archive.

    Record names, sizes, mtimes and preparation manifests. Content hashes where
    provided by preprocessing remain in the included manifests.
    """
    identity = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".npz", ".json"):
            stat = path.stat()
            signature = (str(path.relative_to(root)), stat.st_size, stat.st_mtime_ns)
            identity.update(json.dumps(signature).encode())
            if path.suffix == ".json":
                identity.update(path.read_bytes())
            count += 1
    if count == 0:
        raise ValueError("Daily observation archive is empty")
    return {"inventory_sha256": identity.hexdigest(), "files": count}


def run_manifest(
    args: ObservationTrainConfig, data: Samples, sources: Sources
) -> Manifest:
    package = Path(__file__).resolve().parents[1]
    normalization = data.normalization
    verify_normalization(normalization)
    files = [data.root / "grid.npz"] + [
        normalization / n
        for n in ("normalization.npz", "normalization.json", "statistics.npz")
    ]
    for kind in sources.models:
        root = (
            args.data.local_path("om4", args.experiment) / "OM4.zarr"
            if kind == "om4"
            else args.data.local_path("cm4", args.experiment)
        )
        files.append(root / ".zmetadata")
    if sources.models:
        files.extend(
            args.data.local_path("om4", args.experiment) / name / ".zmetadata"
            for name in ("OM4_means.zarr", "OM4_stds.zarr")
        )
    return {
        "version": 1,
        "mix": args.mix,
        "config": args.scientific_config(),
        "selection_metric": SELECTION_METRIC,
        "runtime": {"torch": str(torch.__version__), "numpy": np.__version__},
        "channels": data.grid["names"].tolist(),
        "shape": list(data.mask.shape),
        "observations": str(data.root.resolve()),
        "daily": str(data.daily.resolve()),
        "daily_identity": archive_identity(data.daily),
        "model_identity_scope": "Zarr metadata; field chunks checked for support and finiteness when read",
        "data_sha256": {str(path.resolve()): digest(path) for path in files},
        "code_sha256": {
            str(path.relative_to(package)): digest(path)
            for path in sorted(package.rglob("*.py"))
        },
    }
