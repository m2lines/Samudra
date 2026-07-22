# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import logging
import multiprocessing
import queue as queue_module
import re
import time
from dataclasses import dataclass
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


@dataclass(frozen=True)
class CheckpointEvalTarget:
    epoch: int
    kind: str
    path: str
    for_inference: bool


def checkpoint_label(entry: CheckpointEvalTarget) -> str:
    match entry.kind:
        case "periodic":
            return f"epoch_{entry.epoch:04d}"
        case "ema_latest":
            return "final_ema"
        case _:
            return f"{entry.kind}_epoch_{entry.epoch:04d}"


def _entry_from_file(
    checkpoint_path: Path,
    kind: str,
    *,
    epoch: int | None = None,
    for_inference: bool = False,
) -> CheckpointEvalTarget:
    if epoch is None:
        # Read epoch from the checkpoint file itself.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        epoch = checkpoint.get("epoch")
        if not isinstance(epoch, int):
            raise ValueError(
                f"Checkpoint at {checkpoint_path} is missing integer epoch"
            )
    return CheckpointEvalTarget(
        epoch=epoch,
        kind=kind,
        path=str(checkpoint_path),
        for_inference=for_inference,
    )


def discover_checkpoints_from_directory(
    checkpoint_paths: CheckpointPaths,
    last_n_checkpoints: int | None = None,
    checkpoints: list[int] | None = None,
) -> list[CheckpointEvalTarget]:
    if last_n_checkpoints is not None and checkpoints is not None:
        raise ValueError("pass only one of last_n_checkpoints or checkpoints, not both")
    if last_n_checkpoints is not None and last_n_checkpoints < 1:
        raise ValueError(f"last_n_checkpoints must be >= 1, got {last_n_checkpoints}")

    checkpoint_dir = checkpoint_paths.checkpoint_dir
    periodic: dict[int, Path] = {}
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        if match := re.match(r"^ckpt_(\d+)\.pt$", path.name):
            periodic[int(match.group(1))] = path

    if checkpoints is not None:
        # Evaluate exactly the requested epochs; fail loudly if any are missing.
        missing = sorted(set(checkpoints) - periodic.keys())
        if missing:
            raise ValueError(
                f"requested checkpoint epochs not found in {checkpoint_dir}: {missing}"
            )
        targets = [
            _entry_from_file(periodic[epoch], "periodic", epoch=epoch)
            for epoch in sorted(set(checkpoints))
        ]
    else:
        targets = [
            _entry_from_file(path, "periodic", epoch=epoch)
            for epoch, path in sorted(periodic.items())
        ]
        if last_n_checkpoints is not None:
            targets = targets[-last_n_checkpoints:]

    # The final EMA checkpoint is always included in addition to the selected
    # periodic checkpoints.
    if checkpoint_paths.ema_checkpoint_path.exists():
        targets.append(
            _entry_from_file(
                checkpoint_paths.ema_checkpoint_path,
                "ema_latest",
                for_inference=True,
            )
        )

    return targets


def partition_checkpoint_work(
    entries: list[CheckpointEvalTarget],
    worker_count: int,
) -> list[list[CheckpointEvalTarget]]:
    if worker_count < 1:
        raise ValueError("worker_count must be at least 1")
    return [entries[i::worker_count] for i in range(worker_count)]


def _resolve_worker_count(
    backend: str,
    num_checkpoints: int,
) -> tuple[int, list[int]]:
    wants_gpu = backend in {"auto", "cuda"}
    available_gpus = (
        torch.cuda.device_count() if wants_gpu and torch.cuda.is_available() else 0
    )
    if backend == "cuda" and available_gpus == 0:
        raise RuntimeError("post-train eval requested CUDA but no GPUs are available")
    if available_gpus == 0:
        logger.info("No GPUs available for post-train eval; falling back to serial CPU")
        return 1, []

    worker_count = max(1, min(num_checkpoints, available_gpus))
    return worker_count, list(range(worker_count))


def _write_summary(results: list[dict[str, object]], sweep_root: Path) -> None:
    summary_path = sweep_root / "summary.json"
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
    entry: CheckpointEvalTarget,
    eval_config_path: Path,
    eval_override_args: tuple[str, ...],
    sweep_root: Path,
    data_root: Location | None,
    gpu_index: int | None,
) -> dict[str, object]:
    # Eval imports EvalConfig, so keep it out of this module's import path while
    # config.py is defining the CheckpointSweep builder.
    from samudra.eval import Eval

    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)

    with MultitonScope():
        cfg = _load_eval_config(eval_config_path, eval_override_args)
        cfg.ckpt_path = entry.path
        cfg.experiment.base_output_dir = str(sweep_root)
        cfg.experiment.name = checkpoint_label(entry)
        if data_root is not None:
            cfg.experiment.data_root = data_root
        cfg.experiment.wandb.mode = "disabled"

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
            "checkpoint_kind": entry.kind,
            "checkpoint_epoch": entry.epoch,
            "checkpoint_path": entry.path,
            "for_inference": entry.for_inference,
            "label": checkpoint_label(entry),
            "output_dir": str(cfg.experiment.output_dir),
            "elapsed_seconds": elapsed_seconds,
            "metrics": serialized,
        }


def _run_single_checkpoint_viz(
    result: dict[str, object],
    template: "VizTemplate",
    steps: list[str],
    viz_dirname: str,
) -> dict[str, object]:
    # Viz config imports TimeConfig, so these need to stay off config.py's
    # module initialization path.
    from samudra.viz.config import VizRunConfig, run_steps

    label = cast(str, result["label"])
    eval_output_dir = Path(cast(str, result["output_dir"]))
    prediction_path = eval_output_dir / "predictions.zarr"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Expected saved predictions at {prediction_path} for post-train viz sweep"
        )

    checkpoint_run = VizRunConfig(
        name=label,
        location=LocalLocation(path=prediction_path.resolve()),
        variables=template.variables,
    )
    output_path = eval_output_dir / viz_dirname
    output_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    viz = template.instantiate(
        output_path,
        [checkpoint_run.build(template.data_root)],
    )
    run_steps(viz, steps)
    elapsed_seconds = time.perf_counter() - start

    return {
        "output_dir": str(output_path),
        "prediction_path": str(prediction_path),
        "elapsed_seconds": elapsed_seconds,
    }


def _worker_main(
    entries: list[CheckpointEvalTarget],
    eval_config_path: Path,
    eval_override_args: tuple[str, ...],
    sweep_root: str,
    data_root: Location | None,
    gpu_index: int | None,
    queue: multiprocessing.Queue,
) -> None:
    try:
        for entry in entries:
            queue.put(
                _run_single_checkpoint_eval(
                    entry=entry,
                    eval_config_path=eval_config_path,
                    eval_override_args=eval_override_args,
                    sweep_root=Path(sweep_root),
                    data_root=data_root,
                    gpu_index=gpu_index,
                )
            )
    except Exception as exc:
        queue.put({"error": str(exc), "gpu_index": gpu_index})
        raise


@dataclass(frozen=True)
class CheckpointSweep:
    """Ready-to-run checkpoint sweep built from configuration."""

    eval_config_path: Path
    checkpoint_paths: CheckpointPaths
    eval_override_args: tuple[str, ...] = ()
    data_root: Location | None = None
    sweep_root: Path | None = None
    viz_config_path: Path | None = None
    last_n_checkpoints: int | None = None
    checkpoints: list[int] | None = None
    viz_dirname: str = "viz"

    def run(self) -> list[dict[str, object]]:
        return run_checkpoint_sweep(
            eval_config_path=self.eval_config_path,
            checkpoint_paths=self.checkpoint_paths,
            eval_override_args=self.eval_override_args,
            data_root=self.data_root,
            sweep_root=self.sweep_root,
            viz_config_path=self.viz_config_path,
            last_n_checkpoints=self.last_n_checkpoints,
            checkpoints=self.checkpoints,
            viz_dirname=self.viz_dirname,
        )


def run_checkpoint_sweep(
    eval_config_path: Path,
    checkpoint_paths: CheckpointPaths,
    eval_override_args: tuple[str, ...] = (),
    data_root: Location | None = None,
    sweep_root: Path | None = None,
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
    if sweep_root is None:
        sweep_root = checkpoint_paths.checkpoint_dir.parent / eval_cfg.experiment.name
    sweep_root.mkdir(parents=True, exist_ok=True)

    if viz_config_path is not None and not eval_cfg.save_zarr:
        raise ValueError(
            "post-train viz sweep requires eval.save_zarr = true so predictions.zarr is written"
        )

    worker_count, gpu_indices = _resolve_worker_count(eval_cfg.backend, len(targets))

    logger.info(
        "Running checkpoint sweep for %d checkpoints with %d worker(s)",
        len(targets),
        worker_count,
    )

    results: list[dict[str, object]] = []
    if worker_count == 1:
        gpu_index = gpu_indices[0] if gpu_indices else None
        for entry in targets:
            results.append(
                _run_single_checkpoint_eval(
                    entry=entry,
                    eval_config_path=eval_config_path,
                    eval_override_args=eval_override_args,
                    sweep_root=sweep_root,
                    data_root=data_root,
                    gpu_index=gpu_index,
                )
            )
    else:
        ctx = multiprocessing.get_context("spawn")
        queue: multiprocessing.Queue = ctx.Queue()
        processes = []
        shards = partition_checkpoint_work(targets, worker_count)
        for gpu_index, shard in zip(gpu_indices, shards, strict=True):
            process = ctx.Process(
                target=_worker_main,
                args=(
                    shard,
                    eval_config_path,
                    eval_override_args,
                    str(sweep_root),
                    data_root,
                    gpu_index,
                    queue,
                ),
            )
            process.start()
            processes.append(process)

        while len(results) < len(targets):
            try:
                item = queue.get(timeout=30)
            except queue_module.Empty:
                for process in processes:
                    if process.exitcode not in {None, 0}:
                        raise RuntimeError(
                            f"checkpoint sweep worker exited with status {process.exitcode}"
                        )
                continue
            if "error" in item:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                    process.join()
                raise RuntimeError(
                    f"checkpoint sweep worker failed on gpu {item['gpu_index']}: {item['error']}"
                )
            results.append(item)

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"checkpoint sweep worker exited with status {process.exitcode}"
                )

    _write_summary(results, sweep_root)

    if viz_config_path is not None:
        from samudra.viz import VizTemplateConfig

        logger.info("Running post-train viz sweep for %d checkpoints", len(results))
        template_cfg = VizTemplateConfig.from_yaml(viz_config_path)
        default_data_root: ResolvedLocation = LocalLocation(path=Path.cwd())
        if data_root is not None:
            default_data_root = default_data_root.resolve(data_root)
        template = template_cfg.build_template(default_data_root)
        for result in results:
            result["viz"] = _run_single_checkpoint_viz(
                result,
                template,
                template_cfg.selected_steps,
                viz_dirname=viz_dirname,
            )
        _write_summary(results, sweep_root)

    return results


def run_standalone_checkpoint_sweep(
    eval_config_path: Path,
    checkpoint_dir: Path,
    eval_override_args: list[str] | None = None,
    viz_config_path: Path | None = None,
    last_n_checkpoints: int | None = None,
    checkpoints: list[int] | None = None,
) -> list[dict[str, object]]:
    return CheckpointSweep(
        eval_config_path=eval_config_path,
        checkpoint_paths=CheckpointPaths(checkpoint_dir),
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

    # Expand ~ in user-provided config paths before resolving them to absolute paths.
    viz_config_path = (
        Path(args.viz_config).expanduser().resolve()
        if args.viz_config is not None
        else None
    )
    run_standalone_checkpoint_sweep(
        eval_config_path=Path(args.config).expanduser().resolve(),
        checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
        eval_override_args=eval_override_args,
        viz_config_path=viz_config_path,
        last_n_checkpoints=args.last_n_checkpoints,
        checkpoints=args.checkpoints,
    )


if __name__ == "__main__":
    main()
