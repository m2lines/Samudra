# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run a matched loader qualification job and persist rank-local evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import statistics
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from samudra.config import CpuDataLoadingConfig, RustDataLoadingConfig, TrainConfig
from samudra.train import Trainer
from samudra.utils.distributed import destroy_distributed_mode
from samudra.utils.logging import handle_logging, handle_warnings


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        return value.detach().item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "max": max(values) if values else None,
    }


def _rss_bytes() -> int:
    statm = Path("/proc/self/statm")
    if not statm.exists():
        return 0
    resident_pages = int(statm.read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


class SystemSampler:
    def __init__(self, path: Path, interval_seconds: float, device: torch.device):
        self.path = path
        self.interval_seconds = interval_seconds
        self.device = device
        self.samples: list[dict[str, float | int | None]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="rust-loader-qualification-system-sampler",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        previous_wall = time.perf_counter()
        previous_cpu = time.process_time()
        with self.path.open("w", buffering=1) as output:
            while not self._stop.wait(self.interval_seconds):
                now_wall = time.perf_counter()
                now_cpu = time.process_time()
                elapsed = now_wall - previous_wall
                sample: dict[str, float | int | None] = {
                    "monotonic_seconds": now_wall,
                    "cpu_percent": (
                        100 * (now_cpu - previous_cpu) / elapsed if elapsed else 0
                    ),
                    "rss_bytes": _rss_bytes(),
                    "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * 1024,
                }
                if self.device.type == "cuda":
                    sample.update(
                        {
                            "gpu_utilization_percent": torch.cuda.utilization(
                                self.device
                            ),
                            "gpu_memory_utilization_percent": torch.cuda.memory_usage(
                                self.device
                            ),
                            "torch_device_allocated_bytes": torch.cuda.memory_allocated(
                                self.device
                            ),
                            "torch_device_reserved_bytes": torch.cuda.memory_reserved(
                                self.device
                            ),
                        }
                    )
                    host_stats = torch.cuda.host_memory_stats()
                    sample["torch_pinned_allocated_bytes"] = host_stats.get(
                        "allocated_bytes.all.current"
                    )
                    sample["torch_pinned_peak_bytes"] = host_stats.get(
                        "allocated_bytes.all.peak"
                    )
                self.samples.append(sample)
                output.write(json.dumps(sample, sort_keys=True) + "\n")
                previous_wall = now_wall
                previous_cpu = now_cpu


class MetricRecorder:
    def __init__(
        self,
        path: Path,
        run_label: str,
        schedule_provider: Callable[[], dict[str, Any]],
    ):
        self.path = path
        self.run_label = run_label
        self.schedule_provider = schedule_provider
        self.records: list[dict[str, Any]] = []
        self._output = path.open("w", buffering=1)

    def close(self) -> None:
        self._output.close()

    def record(
        self, metrics: Mapping[str, Any], step: int | None, **_kwargs: Any
    ) -> None:
        serialized = {
            key: converted
            for key, value in metrics.items()
            if (converted := _json_value(value)) is not None
        }
        record = {
            "run_label": self.run_label,
            "rank": _rank(),
            "step": step,
            "monotonic_seconds": time.perf_counter(),
            "metrics": serialized,
        }
        if "epoch" in serialized:
            record["schedule"] = self.schedule_provider()
        self.records.append(record)
        self._output.write(json.dumps(record, sort_keys=True) + "\n")


def _metric_values(records: list[dict[str, Any]], name: str) -> list[float]:
    return [
        float(value)
        for record in records
        if (value := record["metrics"].get(name)) is not None
    ]


def _system_values(
    samples: list[dict[str, float | int | None]], name: str
) -> list[float]:
    return [
        float(value) for sample in samples if (value := sample.get(name)) is not None
    ]


def _schedule_evidence(trainer: Trainer) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name in ("train", "val"):
        sampler = getattr(trainer, f"{name}_sampler")
        batches = getattr(sampler, "last_batches", [])
        payload = json.dumps(batches, separators=(",", ":")).encode()
        evidence[name] = {
            "batch_count": len(batches),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    return evidence


def _loader_memory_evidence(trainer: Trainer) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for name in ("train", "val"):
        loader = getattr(trainer, f"{name}_loader")
        evidence[name] = getattr(loader, "pinned_pool_stats", None)
    return evidence


def _build_summary(
    recorder: MetricRecorder,
    sampler: SystemSampler,
    loader: str,
    backend: str,
    elapsed_seconds: float,
    loader_memory: dict[str, Any],
    epoch_schedules: list[dict[str, Any]],
) -> dict[str, Any]:
    batch_loss = _metric_values(recorder.records, "train/batch/loss")
    iteration = _metric_values(recorder.records, "train/batch/iter_time")
    summary = {
        "run_label": recorder.run_label,
        "rank": _rank(),
        "loader": loader,
        "backend": backend,
        "elapsed_seconds": elapsed_seconds,
        "loader_memory": loader_memory,
        "batch_loss": batch_loss,
        "epoch_schedules": epoch_schedules,
        "train_data_wait_seconds": _summary(
            _metric_values(recorder.records, "train/batch/data_wait_time")
        ),
        "train_data_load_seconds": _summary(
            _metric_values(recorder.records, "train/batch/data_load_time")
        ),
        "train_iteration_seconds": _summary(iteration),
        "train_batches_per_second": len(iteration) / sum(iteration)
        if iteration and sum(iteration)
        else None,
        "cpu_percent": _summary(_system_values(sampler.samples, "cpu_percent")),
        "rss_bytes": _summary(_system_values(sampler.samples, "rss_bytes")),
        "peak_rss_bytes": max(
            _system_values(sampler.samples, "peak_rss_bytes"), default=None
        ),
        "gpu_utilization_percent": _summary(
            _system_values(sampler.samples, "gpu_utilization_percent")
        ),
        "torch_device_allocated_bytes": _summary(
            _system_values(sampler.samples, "torch_device_allocated_bytes")
        ),
        "torch_device_reserved_bytes": _summary(
            _system_values(sampler.samples, "torch_device_reserved_bytes")
        ),
        "torch_pinned_allocated_bytes": _summary(
            _system_values(sampler.samples, "torch_pinned_allocated_bytes")
        ),
        "torch_pinned_peak_bytes": max(
            _system_values(sampler.samples, "torch_pinned_peak_bytes"), default=None
        ),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/qualification/rust_loader.yaml")
    )
    parser.add_argument("--loader", choices=("cpu", "rust"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--backend", choices=("cuda", "nccl"), default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--prefetch-batches", type=int, default=2)
    parser.add_argument("--max-concurrent-reads", type=int, default=8)
    parser.add_argument("--system-sample-seconds", type=float, default=0.1)
    args = parser.parse_args()

    cfg = TrainConfig.from_yaml_and_cli(
        [
            str(args.config),
            "--experiment.data_root",
            str(args.data_root),
            "--experiment.base_output_dir",
            str(args.output_root),
            "--experiment.name",
            args.name,
            "--backend",
            args.backend,
        ]
    )
    if args.loader == "cpu":
        cfg.data.loading = CpuDataLoadingConfig(
            num_workers=args.cpu_workers,
            persistent_workers=args.cpu_workers > 0,
        )
    else:
        cfg.data.loading = RustDataLoadingConfig(
            prefetch_batches=args.prefetch_batches,
            max_concurrent_reads=args.max_concurrent_reads,
            prefetch_to_device=True,
        )
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.resume is not None:
        cfg.resume_ckpt_path = str(args.resume)

    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    rank = _rank()
    metrics_path = (
        cfg.experiment.output_dir / f"metrics-{args.run_label}-rank{rank}.jsonl"
    )
    system_path = (
        cfg.experiment.output_dir / f"system-{args.run_label}-rank{rank}.jsonl"
    )
    summary_path = (
        cfg.experiment.output_dir / f"summary-{args.run_label}-rank{rank}.json"
    )

    trainer = Trainer(cfg)
    trainer_ref = weakref.ref(trainer)

    def schedule_evidence() -> dict[str, Any]:
        current_trainer = trainer_ref()
        if current_trainer is None:
            return {}
        return _schedule_evidence(current_trainer)

    recorder = MetricRecorder(
        metrics_path,
        args.run_label,
        schedule_provider=schedule_evidence,
    )
    setattr(trainer.wandb_logger, "log", recorder.record)
    sampler = SystemSampler(system_path, args.system_sample_seconds, trainer.device)
    epoch_schedules: list[dict[str, Any]] = []
    validate_one_epoch_ref = weakref.WeakMethod(trainer.validate_one_epoch)

    def validate_and_capture_schedule(epoch: int) -> dict[str, float]:
        validate_one_epoch = validate_one_epoch_ref()
        if validate_one_epoch is None:
            raise RuntimeError("Trainer was released before validation completed")
        stats = validate_one_epoch(epoch)
        epoch_schedules.append(
            {
                "epoch": epoch,
                "schedule": schedule_evidence(),
            }
        )
        return stats

    setattr(trainer, "validate_one_epoch", validate_and_capture_schedule)

    start = time.perf_counter()
    sampler.start()
    loader_memory: dict[str, Any] = {}
    try:
        trainer.run()
        loader_memory = _loader_memory_evidence(trainer)
    finally:
        sampler.stop()
        recorder.close()
        # Release persistent workers before tearing down the process group.
        del trainer
        destroy_distributed_mode()
    elapsed = time.perf_counter() - start
    summary = _build_summary(
        recorder,
        sampler,
        args.loader,
        args.backend,
        elapsed,
        loader_memory,
        epoch_schedules,
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
