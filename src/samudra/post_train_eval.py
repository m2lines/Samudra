# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import multiprocessing
import queue as queue_module
import time
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from samudra.utils.location import LocalLocation, Location, ResolvedLocation
from samudra.utils.multiton import MultitonScope
from samudra.utils.train import CheckpointPaths

if TYPE_CHECKING:
    from samudra.config import EvalConfig
    from samudra.viz import VizTemplate

logger = logging.getLogger(__name__)
WORKER_TIMEOUT_SECONDS = 6 * 60 * 60


def checkpoint_label(checkpoint_path: Path) -> str:
    """Return the output label for a supported checkpoint filename."""
    if checkpoint_path.name == CheckpointPaths.EMA_CHECKPOINT_NAME:
        return "ema_latest"
    epoch = CheckpointPaths.periodic_checkpoint_epoch(checkpoint_path)
    if epoch is not None:
        return f"epoch_{epoch:04d}"
    raise ValueError(f"Unsupported checkpoint filename: {checkpoint_path.name}")


def discover_checkpoints_from_directory(
    checkpoint_paths: CheckpointPaths,
    last_n_checkpoints: int | None = None,
    checkpoints: list[int] | None = None,
) -> list[Path]:
    if last_n_checkpoints is not None and checkpoints is not None:
        raise ValueError("pass only one of last_n_checkpoints or checkpoints, not both")
    if last_n_checkpoints is not None and last_n_checkpoints < 1:
        raise ValueError(f"last_n_checkpoints must be >= 1, got {last_n_checkpoints}")

    checkpoint_dir = checkpoint_paths.checkpoint_dir
    periodic = checkpoint_paths.periodic_checkpoint_paths()

    if checkpoints is not None:
        # Evaluate exactly the requested epochs; fail loudly if any are missing.
        missing = sorted(set(checkpoints) - periodic.keys())
        if missing:
            raise ValueError(
                f"requested checkpoint epochs not found in {checkpoint_dir}: {missing}"
            )
        targets = [periodic[epoch] for epoch in sorted(set(checkpoints))]
    else:
        targets = [path for _, path in sorted(periodic.items())]
        if last_n_checkpoints is not None:
            targets = targets[-last_n_checkpoints:]

    # The final EMA checkpoint is always included in addition to the selected
    # periodic checkpoints.
    if checkpoint_paths.ema_checkpoint_path.exists():
        targets.append(checkpoint_paths.ema_checkpoint_path)

    return targets


def partition_checkpoint_work(
    entries: list[Path],
    worker_count: int,
) -> list[list[Path]]:
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    return [entries[i::worker_count] for i in range(worker_count)]


def _resolve_workers(
    backend: str,
    num_checkpoints: int,
) -> list[torch.device]:
    wants_gpu = backend in {"auto", "cuda"}
    available_gpus = (
        torch.cuda.device_count() if wants_gpu and torch.cuda.is_available() else 0
    )
    if backend == "cuda" and available_gpus == 0:
        raise RuntimeError("post-train eval requested CUDA but no GPUs are available")
    if available_gpus == 0:
        logger.info("No GPUs available for post-train eval; falling back to serial CPU")
        return [torch.device("cpu")]

    worker_count = max(1, min(num_checkpoints, available_gpus))
    return [torch.device("cuda", index) for index in range(worker_count)]


def _write_summary(results: list[dict[str, object]], sweep_output_dir: Path) -> None:
    summary_path = sweep_output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_eval_config(
    eval_config_path: Path,
    eval_override_args: tuple[str, ...],
) -> "EvalConfig":
    # Imported lazily so config.py can import the built CheckpointSweep type.
    from samudra.config import EvalConfig

    if eval_override_args:
        return EvalConfig.from_yaml_and_cli(
            [str(eval_config_path), *eval_override_args]
        )
    return EvalConfig.from_yaml(eval_config_path)


def _run_single_checkpoint_eval(
    checkpoint_path: Path,
    eval_config_path: Path,
    eval_override_args: tuple[str, ...],
    sweep_output_dir: Path,
    data_root: ResolvedLocation,
    device: torch.device,
) -> dict[str, object]:
    # Eval imports EvalConfig, so keep it out of this module's import path while
    # config.py is defining the CheckpointSweep builder.
    from samudra.eval import Eval

    if device.type == "cuda":
        torch.cuda.set_device(device)

    with MultitonScope():
        cfg = _load_eval_config(eval_config_path, eval_override_args)
        cfg.ckpt_path = str(checkpoint_path)
        cfg.experiment.base_output_dir = str(sweep_output_dir)
        cfg.experiment.name = checkpoint_label(checkpoint_path)
        cfg.experiment.data_root = cast(Location, data_root)

        evaluator = Eval(cfg)
        start = time.perf_counter()
        metrics = evaluator.standalone_inference()
        elapsed_seconds = time.perf_counter() - start
        evaluator.finish()

        # Serialize metrics to JSON-safe primitives, dropping non-scalar tensors.
        serialized: dict[str, float | int | str | bool] = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                serialized[key] = float(value.item())
            elif isinstance(value, (bool, int, float, str)):
                serialized[key] = value

        return {
            "checkpoint_path": str(checkpoint_path),
            "label": checkpoint_label(checkpoint_path),
            "output_dir": str(cfg.experiment.output_dir),
            "elapsed_seconds": elapsed_seconds,
            "metrics": serialized,
        }


def _run_single_checkpoint_viz(
    label: str,
    eval_output_dir: Path,
    template: "VizTemplate",
    steps: list[str],
    viz_dirname: str,
) -> dict[str, object]:
    # Viz config imports TimeConfig, so these need to stay off config.py's
    # module initialization path.
    from samudra.viz.config import run_steps

    prediction_path = eval_output_dir / "predictions.zarr"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Expected saved predictions at {prediction_path} for post-train viz sweep"
        )

    output_path = eval_output_dir / viz_dirname
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    viz = template.instantiate_run(
        output_path,
        label,
        LocalLocation(path=prediction_path.resolve()),
    )
    run_steps(viz, steps)
    elapsed_seconds = time.perf_counter() - start

    return {
        "output_dir": str(output_path),
        "prediction_path": str(prediction_path),
        "elapsed_seconds": elapsed_seconds,
    }


def _worker_main(
    entries: list[Path],
    eval_config_path: Path,
    eval_override_args: tuple[str, ...],
    sweep_output_dir: Path,
    data_root: ResolvedLocation,
    device: torch.device,
    queue: multiprocessing.Queue,
) -> None:
    try:
        for checkpoint_path in entries:
            queue.put(
                _run_single_checkpoint_eval(
                    checkpoint_path=checkpoint_path,
                    eval_config_path=eval_config_path,
                    eval_override_args=eval_override_args,
                    sweep_output_dir=sweep_output_dir,
                    data_root=data_root,
                    device=device,
                )
            )
    except Exception as exc:
        exc.add_note(
            f"checkpoint sweep worker on {device} failed for {checkpoint_path}"
        )
        queue.put(exc)
        raise


def _terminate_workers(processes: Sequence[BaseProcess]) -> None:
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join()


@dataclass(frozen=True)
class CheckpointSweep:
    """Configuration for running a checkpoint evaluation sweep."""

    eval_config_path: Path
    checkpoint_paths: CheckpointPaths
    data_root: ResolvedLocation
    sweep_output_dir: Path
    eval_override_args: tuple[str, ...] = ()
    viz_config_path: Path | None = None
    last_n_checkpoints: int | None = None
    checkpoints: list[int] | None = None
    viz_dirname: str = "viz"

    def __post_init__(self) -> None:
        eval_cfg = _load_eval_config(self.eval_config_path, self.eval_override_args)
        if self.viz_config_path is not None:
            if not eval_cfg.save_zarr:
                raise ValueError(
                    "post-train viz sweep requires eval.save_zarr = true so "
                    "predictions.zarr is written"
                )
            from samudra.viz import VizTemplateConfig

            VizTemplateConfig.from_yaml(self.viz_config_path)

    def run(self) -> list[dict[str, object]]:
        return run_checkpoint_sweep(
            eval_config_path=self.eval_config_path,
            checkpoint_paths=self.checkpoint_paths,
            eval_override_args=self.eval_override_args,
            data_root=self.data_root,
            sweep_output_dir=self.sweep_output_dir,
            viz_config_path=self.viz_config_path,
            last_n_checkpoints=self.last_n_checkpoints,
            checkpoints=self.checkpoints,
            viz_dirname=self.viz_dirname,
        )


def run_checkpoint_sweep(
    eval_config_path: Path,
    checkpoint_paths: CheckpointPaths,
    data_root: ResolvedLocation,
    sweep_output_dir: Path,
    eval_override_args: tuple[str, ...] = (),
    viz_config_path: Path | None = None,
    last_n_checkpoints: int | None = None,
    checkpoints: list[int] | None = None,
    viz_dirname: str = "viz",
) -> list[dict[str, object]]:
    targets = discover_checkpoints_from_directory(
        checkpoint_paths,
        last_n_checkpoints=last_n_checkpoints,
        checkpoints=checkpoints,
    )
    if not targets:
        logger.warning("No checkpoints selected for post-train eval sweep")
        return []

    eval_cfg = _load_eval_config(eval_config_path, eval_override_args)
    sweep_output_dir.mkdir(parents=True, exist_ok=True)

    worker_devices = _resolve_workers(eval_cfg.backend, len(targets))

    logger.info(
        "Running checkpoint sweep for %d checkpoints with %d worker(s)",
        len(targets),
        len(worker_devices),
    )

    ctx = multiprocessing.get_context("spawn")
    queue: multiprocessing.Queue = ctx.Queue()
    processes = []
    shards = partition_checkpoint_work(targets, len(worker_devices))
    for device, shard in zip(worker_devices, shards, strict=True):
        process = ctx.Process(
            target=_worker_main,
            args=(
                shard,
                eval_config_path,
                eval_override_args,
                sweep_output_dir,
                data_root,
                device,
                queue,
            ),
        )
        process.start()
        processes.append(process)

    results: list[dict[str, object]] = []
    deadline = time.monotonic() + WORKER_TIMEOUT_SECONDS
    while len(results) < len(targets):
        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 0:
            _terminate_workers(processes)
            raise TimeoutError(
                f"checkpoint sweep exceeded {WORKER_TIMEOUT_SECONDS} seconds"
            )
        try:
            item = queue.get(timeout=min(30, remaining_seconds))
        except queue_module.Empty:
            for process in processes:
                if process.exitcode not in {None, 0}:
                    _terminate_workers(processes)
                    raise RuntimeError(
                        f"checkpoint sweep worker exited with status {process.exitcode}"
                    )
            continue
        if isinstance(item, Exception):
            _terminate_workers(processes)
            raise item
        results.append(item)

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"checkpoint sweep worker exited with status {process.exitcode}"
            )

    _write_summary(results, sweep_output_dir)

    if viz_config_path is not None:
        from samudra.viz import VizTemplateConfig

        logger.info("Running post-train viz sweep for %d checkpoints", len(results))
        template_cfg = VizTemplateConfig.from_yaml(viz_config_path)
        template = template_cfg.build_template(data_root)
        for result in results:
            result["viz"] = _run_single_checkpoint_viz(
                cast(str, result["label"]),
                Path(cast(str, result["output_dir"])),
                template,
                template_cfg.selected_steps,
                viz_dirname=viz_dirname,
            )
        _write_summary(results, sweep_output_dir)

    return results


def run_standalone_checkpoint_sweep(
    eval_config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    eval_override_args: list[str] | None = None,
    viz_config_path: Path | None = None,
    last_n_checkpoints: int | None = None,
    checkpoints: list[int] | None = None,
) -> list[dict[str, object]]:
    eval_cfg = _load_eval_config(eval_config_path, tuple(eval_override_args or []))
    return CheckpointSweep(
        eval_config_path=eval_config_path,
        checkpoint_paths=CheckpointPaths(checkpoint_dir),
        data_root=eval_cfg.experiment.resolved_data_root,
        sweep_output_dir=output_dir,
        eval_override_args=tuple(eval_override_args or []),
        viz_config_path=viz_config_path,
        last_n_checkpoints=last_n_checkpoints,
        checkpoints=checkpoints,
    ).run()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone inference sweeps for an existing saved_nets directory."
    )
    parser.add_argument(
        "config",
        help="Path to eval config YAML",
    )
    parser.add_argument(
        "--viz_config",
        help="Path to viz config YAML to run after the eval sweep",
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to a saved_nets directory from an old run",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for checkpoint evaluation outputs and summary.json",
    )
    parser.add_argument(
        "--last_n_checkpoints",
        type=int,
        help="Optional limit to only evaluate the last N discovered checkpoints",
    )
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        help="Explicit checkpoint epochs to evaluate; mutually exclusive with "
        "--last_n_checkpoints",
    )
    args, eval_override_args = parser.parse_known_args(argv)

    viz_config_path = Path(args.viz_config) if args.viz_config is not None else None
    run_standalone_checkpoint_sweep(
        eval_config_path=Path(args.config),
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        eval_override_args=eval_override_args,
        viz_config_path=viz_config_path,
        last_n_checkpoints=args.last_n_checkpoints,
        checkpoints=args.checkpoints,
    )


if __name__ == "__main__":
    main()
