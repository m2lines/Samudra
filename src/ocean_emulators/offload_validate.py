"""Long-lived validation worker for per-epoch EMA training snapshots.

Training in ``validation_mode=offload`` writes immutable
``ema_ckpt_epNNNN.pt`` files. This module watches one or more run directories,
validates those snapshots FIFO on a face-parallel GPU group, and atomically
records enough state to resume after preemption without losing an epoch.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime
import json
import logging
import math
import os
import re
import shutil
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal, Mapping, Sequence, cast

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from ocean_emulators.aggregator.validate.main import ONE_STEP_LOSS_KEY
from ocean_emulators.aggregator.validate.rollout import RolloutValidationAggregator
from ocean_emulators.config import TrainConfig
from ocean_emulators.train import INITIAL_BEST_VAL_LOSS, Trainer
from ocean_emulators.utils.distributed import get_rank, is_main_process
from ocean_emulators.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)

SNAPSHOT_RE = re.compile(r"^ema_ckpt_ep(\d+)\.pt$")
STATE_VERSION = 1
CONTROL_TIMEOUT = datetime.timedelta(hours=2)
WandbMode = Literal["online", "offline", "disabled"]


@dataclass(frozen=True)
class OffloadOptions:
    runs: tuple[Path, ...]
    once: bool
    poll_seconds: float
    idle_timeout_seconds: float
    wandb_mode: WandbMode


@dataclass(frozen=True)
class SnapshotJob:
    run_index: int
    run_dir: Path
    snapshot: Path
    epoch: int
    state: dict[str, Any]


@dataclass
class ValidationResult:
    epoch: int
    num_batches_seen: int
    metrics: dict[str, Any]
    scalar_metrics: dict[str, float | None]
    curves: dict[str, dict[str, list[float]]]
    combined_score: float | None
    strict_score: float | None
    rejection_reason: str | None
    contributors: list[str]
    one_step_seconds: float
    autoregressive_seconds: float
    one_step_peak_gib: float
    autoregressive_peak_gib: float


def _parse_offload_args(
    argv: Sequence[str] | None = None,
) -> tuple[OffloadOptions, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--offload.runs", required=True)
    parser.add_argument("--offload.once", action="store_true")
    parser.add_argument("--offload.poll-seconds", type=float, default=60.0)
    parser.add_argument("--offload.idle-timeout-seconds", type=float, default=0.0)
    parser.add_argument(
        "--offload.wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    known, remaining = parser.parse_known_args(argv)
    runs = tuple(
        Path(value.strip()).resolve()
        for value in known.__dict__["offload.runs"].split(",")
        if value.strip()
    )
    if not runs:
        parser.error("--offload.runs must name at least one run directory")
    if known.__dict__["offload.poll_seconds"] <= 0:
        parser.error("--offload.poll-seconds must be positive")
    if known.__dict__["offload.idle_timeout_seconds"] < 0:
        parser.error("--offload.idle-timeout-seconds cannot be negative")
    return (
        OffloadOptions(
            runs=runs,
            once=known.__dict__["offload.once"],
            poll_seconds=known.__dict__["offload.poll_seconds"],
            idle_timeout_seconds=known.__dict__["offload.idle_timeout_seconds"],
            wandb_mode=cast(WandbMode, known.__dict__["offload.wandb_mode"]),
        ),
        remaining,
    )


def _jsonable_config(cfg: TrainConfig) -> dict[str, Any]:
    return json.loads(cfg.model_dump_json())


def _validation_signature(raw: Mapping[str, Any]) -> dict[str, Any]:
    """The configuration fields that can change validation numerics.

    Data locations are intentionally excluded: the validator reads a second,
    byte-equivalent face cache so it does not contend with H200 training I/O.
    Geometry, masks, normalization locations, variables, model, loss, seeded
    draws, and validation horizons still have to match.
    """
    model = copy.deepcopy(raw.get("model", {}))
    model.pop("checkpointing", None)

    data = copy.deepcopy(raw.get("data", {}))
    for key in (
        "data_location",
        "boundary_data_location",
        "num_workers",
        "prefetch_factor",
        "rust_read_threads",
        "concurrent_compute",
        "loader_backend",
    ):
        data.pop(key, None)

    experiment = raw.get("experiment", {})
    replay = raw.get("replay", {})
    face_parallel = raw.get("face_parallel", {})
    return {
        "model": model,
        "data": data,
        "loss": raw.get("loss"),
        "experiment": {
            key: experiment.get(key)
            for key in (
                "prognostic_vars_key",
                "boundary_vars_key",
                "rand_seed",
                "data_root",
            )
        },
        "val_time": raw.get("val_time"),
        "one_step_val_num": raw.get("one_step_val_num"),
        "short_autoregressive_val_length": raw.get(
            "short_autoregressive_val_length"
        ),
        "short_autoregressive_val_num": raw.get("short_autoregressive_val_num"),
        "long_autoregressive_val_length": raw.get("long_autoregressive_val_length"),
        "long_autoregressive_val_num": raw.get("long_autoregressive_val_num"),
        "long_autoregressive_val_start_epoch": raw.get(
            "long_autoregressive_val_start_epoch"
        ),
        "autoregressive_val_steps_forward": raw.get(
            "autoregressive_val_steps_forward"
        ),
        "surface_snapshot": raw.get("surface_snapshot"),
        "debug": raw.get("debug"),
        "data_stride": raw.get("data_stride"),
        "temporal_stride": raw.get("temporal_stride"),
        "temporal_stride_transition": raw.get("temporal_stride_transition"),
        "steps": raw.get("steps"),
        "step_transition": raw.get("step_transition"),
        "replay": {
            key: replay.get(key) for key in ("enabled", "grouped", "blend_window")
        },
        "face_parallel": {"enabled": face_parallel.get("enabled")},
    }


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(child, path))
    return flattened


def validation_config_mismatches(
    run_config: Mapping[str, Any], validator_config: Mapping[str, Any]
) -> list[str]:
    run = _flatten(_validation_signature(run_config))
    validator = _flatten(_validation_signature(validator_config))
    mismatches = [
        f"{path}: run={run.get(path)!r}, validator={validator.get(path)!r}"
        for path in sorted(set(run) | set(validator))
        if run.get(path) != validator.get(path)
    ]

    guards = {
        "validation_mode": run_config.get("validation_mode") in {"offload", "both"},
        "test_using_ema": run_config.get("test_using_ema") is True,
        "inference_epochs": run_config.get("inference_epochs") == [],
        "temporal_stride_transition": run_config.get(
            "temporal_stride_transition"
        )
        == [],
        "step_transition": run_config.get("step_transition") == [],
        "face_parallel.enabled": run_config.get("face_parallel", {}).get("enabled")
        is True,
    }
    mismatches.extend(
        f"{name}: unsupported run value {run_config.get(name)!r}"
        for name, accepted in guards.items()
        if not accepted
    )
    return mismatches


def _default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "validated_epochs": [],
        "best_epoch": None,
        "best_score": None,
        "combined_loss_contributors": [],
        "wandb_id": None,
        "rejections": {},
        "config_rejection_reason": None,
    }


def _state_path(run_dir: Path) -> Path:
    return run_dir / "offload_val" / "state.json"


def load_state(run_dir: Path) -> dict[str, Any]:
    path = _state_path(run_dir)
    if not path.exists():
        return _default_state()
    with path.open() as handle:
        state = json.load(handle)
    if state.get("version") != STATE_VERSION:
        raise ValueError(
            f"Unsupported offload validation state version in {path}: "
            f"{state.get('version')!r}"
        )
    merged = _default_state()
    merged.update(state)
    return merged


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(value).__name__}")
    return value


def _snapshot_candidates(
    run_dir: Path, state: Mapping[str, Any]
) -> list[tuple[int, Path]]:
    validated = {int(epoch) for epoch in state.get("validated_epochs", [])}
    candidates = []
    for path in (run_dir / "saved_nets").glob("ema_ckpt_ep*.pt"):
        match = SNAPSHOT_RE.match(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch not in validated:
            candidates.append((epoch, path))
    return sorted(candidates)


class SnapshotScanner:
    def __init__(
        self, runs: Sequence[Path], validator_config: Mapping[str, Any]
    ) -> None:
        self.runs = tuple(runs)
        self.validator_config = validator_config
        self._compatible: set[Path] = set()
        self._reported_missing: set[Path] = set()

    def _check_config(self, run_dir: Path, state: dict[str, Any]) -> bool:
        if run_dir in self._compatible:
            return True
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            if run_dir not in self._reported_missing:
                logger.info("Waiting for training config: %s", config_path)
                self._reported_missing.add(run_dir)
            return False
        try:
            run_config = _load_yaml(config_path)
        except (OSError, yaml.YAMLError, ValueError) as error:
            # The trainer creates its directory before writing config.yaml. A
            # concurrently started watcher can observe that small window (or a
            # partially written YAML file), so wait and retry like any other
            # not-yet-published input.
            logger.warning("Training config is not ready at %s: %s", config_path, error)
            return False
        mismatches = validation_config_mismatches(run_config, self.validator_config)
        if mismatches:
            reason = "Validation configuration mismatch:\n" + "\n".join(mismatches)
            if state.get("config_rejection_reason") != reason:
                state["config_rejection_reason"] = reason
                _atomic_write_json(_state_path(run_dir), state)
            logger.error("Rejecting run %s: %s", run_dir, reason)
            return False
        if state.get("config_rejection_reason") is not None:
            state["config_rejection_reason"] = None
            _atomic_write_json(_state_path(run_dir), state)
        self._compatible.add(run_dir)
        logger.info("Validation configuration matches run: %s", run_dir)
        return True

    def next_job(self) -> SnapshotJob | None:
        available: list[tuple[float, SnapshotJob]] = []
        for run_index, run_dir in enumerate(self.runs):
            state = load_state(run_dir)
            if not self._check_config(run_dir, state):
                continue
            candidates = _snapshot_candidates(run_dir, state)
            if not candidates:
                continue
            epoch, snapshot = candidates[0]
            available.append(
                (
                    snapshot.stat().st_mtime,
                    SnapshotJob(run_index, run_dir, snapshot, epoch, state),
                )
            )
        if not available:
            return None
        return min(available, key=lambda entry: entry[0])[1]


def _broadcast_command(command: dict[str, Any] | None, control_group) -> dict[str, Any]:
    if not dist.is_available() or not dist.is_initialized():
        assert command is not None
        return command
    payload = [command if is_main_process() else None]
    dist.broadcast_object_list(payload, src=0, group=control_group)
    assert isinstance(payload[0], dict)
    return payload[0]


def _model_without_ddp(trainer: Trainer) -> torch.nn.Module:
    if isinstance(trainer.model, DistributedDataParallel):
        return trainer.model.module
    return trainer.model


def _load_snapshot(trainer: Trainer, job: SnapshotJob) -> int:
    checkpoint = torch.load(
        job.snapshot,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    if int(checkpoint.get("epoch", -1)) != job.epoch:
        raise ValueError(
            f"{job.snapshot} names epoch {job.epoch}, but stores "
            f"epoch={checkpoint.get('epoch')!r}"
        )
    if checkpoint.get("epoch_complete") is not True:
        raise ValueError(f"{job.snapshot} is not an epoch-complete checkpoint")
    ema = checkpoint.get("ema", {})
    if isinstance(ema, Mapping) and "ema_params" in ema:
        raise ValueError(
            f"{job.snapshot} carries ema_params; an inference EMA snapshot must "
            "already store EMA weights directly in model"
        )

    state = {
        name.removeprefix("module."): tensor
        for name, tensor in checkpoint["model"].items()
    }
    _model_without_ddp(trainer).load_state_dict(state, strict=True)
    num_batches_seen = int(checkpoint.get("num_batches_seen", 0))
    del state, checkpoint
    _verify_parameter_checksum(trainer)
    return num_batches_seen


@torch.no_grad()
def _verify_parameter_checksum(trainer: Trainer) -> None:
    model = _model_without_ddp(trainer)
    checksum = torch.zeros(3, dtype=torch.float64, device=trainer.device)
    for parameter in model.parameters():
        values = parameter.detach().to(dtype=torch.float64)
        checksum[0] += values.sum()
        checksum[1] += values.square().sum()
        checksum[2] += values.numel()
    if not dist.is_available() or not dist.is_initialized():
        return
    gathered = [torch.empty_like(checksum) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, checksum)
    if any(not torch.equal(candidate, gathered[0]) for candidate in gathered[1:]):
        raise RuntimeError(
            "Snapshot parameters differ between validator ranks after loading: "
            f"{[candidate.tolist() for candidate in gathered]}"
        )


def _reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gib(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    value = torch.tensor(
        torch.cuda.max_memory_reserved(device) / 2**30,
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return float(value.cpu())


@contextlib.contextmanager
def _capture_rollout_curves() -> Iterator[dict[str, dict[str, list[float]]]]:
    curves: dict[str, dict[str, list[float]]] = {}
    original = RolloutValidationAggregator.get_logs

    def capture(self, label: str):
        curves[label] = {
            "loss_by_step": self.loss_by_step().detach().cpu().tolist(),
            "rmse_by_step": self.rmse_by_step().detach().cpu().tolist(),
        }
        return original(self, label)

    aggregator_class = cast(Any, RolloutValidationAggregator)
    aggregator_class.get_logs = capture
    try:
        yield curves
    finally:
        aggregator_class.get_logs = original


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, torch.Tensor) and value.numel() == 1:
            number = float(value.detach().cpu())
        else:
            continue
        result[name] = number if math.isfinite(number) else None
    return result


def strict_validation_score(
    trainer: Trainer,
    epoch: int,
    one_step_loss: float,
    autoregressive_metrics: Mapping[str, Any],
    combined_score: float,
) -> tuple[float | None, str | None]:
    active: list[tuple[str, Any]] = [("one-step", one_step_loss)]
    for spec in trainer.due_autoregressive_val_specs(epoch):
        key = f"val/mean/{spec.label}-autoregressive-loss"
        active.append((spec.label, autoregressive_metrics.get(key)))
    invalid = [
        name
        for name, value in active
        if value is None or not math.isfinite(float(value))
    ]
    if invalid:
        return None, f"non-finite or missing active validation: {', '.join(invalid)}"
    if not math.isfinite(float(combined_score)):
        return None, "combined validation score is non-finite"
    return float(combined_score), None


def validate_snapshot(trainer: Trainer, job: SnapshotJob) -> ValidationResult:
    num_batches_seen = _load_snapshot(trainer, job)
    if isinstance(trainer.val_sampler, DistributedSampler):
        trainer.val_sampler.set_epoch(job.epoch)

    prior_best = job.state.get("best_score")
    trainer.best_val_loss = (
        float(prior_best) if prior_best is not None else INITIAL_BEST_VAL_LOSS
    )
    trainer._combined_loss_contributors = frozenset(
        job.state.get("combined_loss_contributors", [])
    )

    _reset_peak_memory(trainer.device)
    one_step_start = time.perf_counter()
    with torch.no_grad():
        one_step_metrics = dict(trainer.validate_one_epoch(job.epoch))
    one_step_seconds = time.perf_counter() - one_step_start
    one_step_peak_gib = _peak_memory_gib(trainer.device)

    _reset_peak_memory(trainer.device)
    autoregressive_start = time.perf_counter()
    with _capture_rollout_curves() as curves, torch.no_grad():
        autoregressive_metrics = dict(
            trainer.validate_autoregressive_one_epoch(job.epoch)
        )
    autoregressive_seconds = time.perf_counter() - autoregressive_start
    autoregressive_peak_gib = _peak_memory_gib(trainer.device)

    one_step_loss = float(one_step_metrics[ONE_STEP_LOSS_KEY])
    combined_score, combined_metrics = trainer.combined_validation_loss(
        one_step_loss, autoregressive_metrics
    )
    strict_score, rejection_reason = strict_validation_score(
        trainer,
        job.epoch,
        one_step_loss,
        autoregressive_metrics,
        combined_score,
    )
    metrics = {
        **one_step_metrics,
        **autoregressive_metrics,
        **combined_metrics,
        "offload/strict-combined-loss": (
            strict_score if strict_score is not None else float("inf")
        ),
        "offload/one-step-seconds": one_step_seconds,
        "offload/autoregressive-seconds": autoregressive_seconds,
        "offload/one-step-peak-reserved-gib": one_step_peak_gib,
        "offload/autoregressive-peak-reserved-gib": autoregressive_peak_gib,
        "epoch": job.epoch,
    }
    return ValidationResult(
        epoch=job.epoch,
        num_batches_seen=num_batches_seen,
        metrics=metrics,
        scalar_metrics=_scalar_metrics(metrics),
        curves=curves,
        combined_score=(
            float(combined_score) if math.isfinite(float(combined_score)) else None
        ),
        strict_score=strict_score,
        rejection_reason=rejection_reason,
        contributors=sorted(trainer._combined_loss_contributors),
        one_step_seconds=one_step_seconds,
        autoregressive_seconds=autoregressive_seconds,
        one_step_peak_gib=one_step_peak_gib,
        autoregressive_peak_gib=autoregressive_peak_gib,
    )


def _replace_metrics_record(path: Path, record: Mapping[str, Any]) -> None:
    records: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                records.append(json.loads(line))
    records = [item for item in records if item.get("epoch") != record.get("epoch")]
    records.append(_finite_json_value(dict(record)))
    records.sort(key=lambda item: int(item["epoch"]))
    text = "".join(
        json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in records
    )
    _atomic_write_text(path, text)


def _finite_json_value(value: Any) -> Any:
    """Replace non-finite leaves with JSON-safe nulls.

    Diverged rollout curves are still useful evidence, but strict JSON does
    not represent NaN or infinity. Preserve every finite point and mark only
    the invalid positions as null so committing a rejected epoch cannot fail.
    """
    if isinstance(value, Mapping):
        return {key: _finite_json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json_value(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    staging.unlink(missing_ok=True)
    try:
        os.link(source, staging)
    except OSError:
        shutil.copy2(source, staging)
    os.replace(staging, destination)


def _log_to_wandb(
    *,
    run_dir: Path,
    state: dict[str, Any],
    result: ValidationResult,
    mode: WandbMode,
    cfg: TrainConfig,
) -> None:
    if mode == "disabled":
        return
    try:
        import wandb

        run_name = f"{run_dir.name}-val"
        run = wandb.init(
            id=state.get("wandb_id"),
            resume="allow",
            reinit=True,
            name=run_name,
            group=run_dir.name,
            project=cfg.experiment.wandb.project,
            entity=cfg.experiment.wandb.entity,
            mode=mode,
            dir=str(run_dir / "offload_val"),
            tags=cfg.experiment.wandb.tags,
            notes=cfg.experiment.wandb.notes,
        )
        if run is not None:
            state["wandb_id"] = run.id
            wandb.log(result.metrics, step=result.num_batches_seen)
        wandb.finish()
    except Exception:
        # Validation results are durable locally and must not be held hostage
        # by a transient tracking-service failure.
        logger.exception("W&B logging failed for %s epoch %d", run_dir, result.epoch)


def commit_result(
    job: SnapshotJob,
    result: ValidationResult,
    *,
    wandb_mode: WandbMode,
    cfg: TrainConfig,
) -> None:
    output_dir = job.run_dir / "offload_val"
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "epoch": result.epoch,
        "snapshot": str(job.snapshot),
        "num_batches_seen": result.num_batches_seen,
        "metrics": result.scalar_metrics,
        "curves": result.curves,
        "combined_score": result.combined_score,
        "strict_score": result.strict_score,
        "rejection_reason": result.rejection_reason,
        "one_step_seconds": result.one_step_seconds,
        "autoregressive_seconds": result.autoregressive_seconds,
        "one_step_peak_reserved_gib": result.one_step_peak_gib,
        "autoregressive_peak_reserved_gib": result.autoregressive_peak_gib,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    _replace_metrics_record(output_dir / "metrics.jsonl", record)

    state = copy.deepcopy(job.state)
    state["combined_loss_contributors"] = result.contributors
    rejections = dict(state.get("rejections", {}))
    if result.rejection_reason is None:
        rejections.pop(str(result.epoch), None)
    else:
        rejections[str(result.epoch)] = result.rejection_reason
    state["rejections"] = rejections

    previous_best = state.get("best_score")
    contributors_joined = bool(
        set(result.contributors)
        - set(job.state.get("combined_loss_contributors", []))
    ) and bool(job.state.get("combined_loss_contributors", []))
    if contributors_joined:
        previous_best = None
    if result.strict_score is not None and (
        previous_best is None or result.strict_score <= float(previous_best)
    ):
        state["best_score"] = result.strict_score
        state["best_epoch"] = result.epoch
        _atomic_link_or_copy(
            job.snapshot,
            job.run_dir / "saved_nets" / "best_validation_ema_ckpt.pt",
        )

    _log_to_wandb(
        run_dir=job.run_dir,
        state=state,
        result=result,
        mode=wandb_mode,
        cfg=cfg,
    )
    state["validated_epochs"] = sorted(
        {int(epoch) for epoch in state.get("validated_epochs", [])} | {result.epoch}
    )
    _atomic_write_json(_state_path(job.run_dir), state)
    logger.info(
        "Validated %s epoch %d: combined=%s strict=%s best_epoch=%s",
        job.run_dir.name,
        result.epoch,
        result.combined_score,
        result.strict_score,
        state.get("best_epoch"),
    )


def _job_to_command(job: SnapshotJob) -> dict[str, Any]:
    return {
        "kind": "job",
        "run_index": job.run_index,
        "run_dir": str(job.run_dir),
        "snapshot": str(job.snapshot),
        "epoch": job.epoch,
        "state": job.state,
    }


def _job_from_command(command: Mapping[str, Any]) -> SnapshotJob:
    return SnapshotJob(
        run_index=int(command["run_index"]),
        run_dir=Path(command["run_dir"]),
        snapshot=Path(command["snapshot"]),
        epoch=int(command["epoch"]),
        state=dict(command["state"]),
    )


def _force_validator_settings(cfg: TrainConfig, options: OffloadOptions) -> None:
    cfg.resume_ckpt_path = None
    cfg.finetune = False
    cfg.preemptible = False
    cfg.emergency_checkpoint_interval_minutes = 0
    cfg.inference_epochs = []
    cfg.validation_mode = "inline"
    cfg.test_using_ema = False
    cfg.face_parallel.blend_scope = "face"
    cfg.replay.checkpoint_buffer = False
    cfg.experiment.wandb.mode = "disabled"
    cfg.experiment.name = f"offload-val-{os.environ.get('SLURM_JOB_ID', 'manual')}"
    cfg.experiment.base_output_dir = str(options.runs[0] / "offload_val" / "runtime")


def run_validator(
    cfg: TrainConfig,
    options: OffloadOptions,
    validator_config: Mapping[str, Any],
) -> None:
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    trainer = Trainer(cfg)
    trainer.test_using_ema = False
    # The snapshot's ``model`` entry already contains EMA weights. Trainer
    # creates a fresh EMA copy during normal startup, but retaining that second
    # w672 parameter set would waste several GiB on each H100 and it is never
    # consulted when test_using_ema is false.
    trainer._ema = None  # type: ignore[assignment]
    if trainer.device.type == "cuda":
        torch.cuda.empty_cache()
    stride = trainer.get_current_temporal_stride(1)
    trainer.temporal_stride = stride
    trainer.init_data_loaders(max(cfg.replay.max_lead_steps), stride)

    control_group = None
    if dist.is_available() and dist.is_initialized():
        control_group = dist.new_group(backend="gloo", timeout=CONTROL_TIMEOUT)

    def request_stop(signum: int, _frame) -> None:
        trainer._stop_requested = True
        trainer._stop_reason = signal.Signals(signum).name
        logger.warning(
            "Validator received %s; stopping between snapshots.",
            trainer._stop_reason,
        )

    for signum in (signal.SIGTERM, signal.SIGINT, getattr(signal, "SIGUSR1", None)):
        if signum is not None:
            signal.signal(signum, request_stop)

    scanner = (
        SnapshotScanner(options.runs, validator_config)
        if is_main_process()
        else None
    )
    idle_since = time.monotonic()
    completed = 0
    try:
        while True:
            command = None
            if is_main_process():
                assert scanner is not None
                if trainer._stop_requested:
                    command = {"kind": "stop", "reason": trainer._stop_reason}
                else:
                    job = scanner.next_job()
                    if job is not None:
                        command = _job_to_command(job)
                        idle_since = time.monotonic()
                    elif (
                        options.idle_timeout_seconds > 0
                        and time.monotonic() - idle_since
                        >= options.idle_timeout_seconds
                    ):
                        command = {"kind": "stop", "reason": "idle timeout"}
                    else:
                        command = {"kind": "idle"}

            command = _broadcast_command(command, control_group)
            kind = command["kind"]
            if kind == "stop":
                logger.info("Stopping offloaded validator: %s", command.get("reason"))
                break
            if kind == "idle":
                time.sleep(options.poll_seconds)
                continue
            if kind != "job":
                raise RuntimeError(f"Unknown validator command: {command!r}")

            job = _job_from_command(command)
            logger.info("Validating %s", job.snapshot)
            result = validate_snapshot(trainer, job)
            if is_main_process():
                commit_result(
                    job,
                    result,
                    wandb_mode=options.wandb_mode,
                    cfg=cfg,
                )
            completed += 1
            if options.once and completed >= 1:
                stop = {"kind": "stop", "reason": "--offload.once completed"}
                _broadcast_command(stop if is_main_process() else None, control_group)
                break
    finally:
        trainer.finish()
        if control_group is not None:
            dist.destroy_process_group(control_group)


def main(argv: Sequence[str] | None = None) -> None:
    options, remaining = _parse_offload_args(argv)
    cfg = TrainConfig.from_yaml_and_cli(remaining)
    validator_config = _jsonable_config(cfg)
    _force_validator_settings(cfg, options)
    run_validator(cfg, options, validator_config)


if __name__ == "__main__":
    main()
