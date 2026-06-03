import argparse
import json
import logging
import multiprocessing
import queue as queue_module
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from samudra.config import EvalConfig, PostTrainCheckpointSweepConfig, TrainConfig
from samudra.eval import Eval
from samudra.utils.location import LocalLocation
from samudra.utils.multiton import MultitonScope
from samudra.utils.train import CheckpointPaths
from samudra.viz import VizConfig
from samudra.viz.config import VizRunConfig, run_with_prepared_groundtruth
from samudra.viz.core import PreparedVizGroundtruth

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
) -> list[CheckpointEvalTarget]:
    checkpoint_dir = checkpoint_paths.checkpoint_dir
    periodic: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        if match := re.match(r"^ckpt_(\d+)\.pt$", path.name):
            periodic.append((int(match.group(1)), path))

    targets: list[CheckpointEvalTarget] = [
        _entry_from_file(path, "periodic", epoch=epoch)
        for epoch, path in sorted(periodic)
    ]
    if checkpoint_paths.ema_checkpoint_path.exists():
        targets.append(
            _entry_from_file(
                checkpoint_paths.ema_checkpoint_path,
                "ema_latest",
                for_inference=True,
            )
        )
    if last_n_checkpoints is not None:
        targets = targets[-last_n_checkpoints:]
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


def _run_single_checkpoint_eval(
    entry: CheckpointEvalTarget,
    eval_config_args: tuple[str, ...],
    sweep_root: Path,
    data_root: object | None,
    gpu_index: int | None,
) -> dict[str, object]:
    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)

    with MultitonScope():
        cfg = EvalConfig.from_yaml_and_cli(list(eval_config_args))
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
    template_cfg: VizConfig,
    prepared_groundtruth: PreparedVizGroundtruth,
    viz_dirname: str | None = None,
) -> dict[str, object]:
    label = cast(str, result["label"])
    eval_output_dir = Path(cast(str, result["output_dir"]))
    prediction_path = eval_output_dir / "predictions.zarr"
    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Expected saved predictions at {prediction_path} for post-train viz sweep"
        )

    with MultitonScope():
        # Build a viz config that points its first run at this checkpoint's predictions,
        # inheriting variables (and any extra runs) from the template.
        if viz_dirname is None:
            viz_dirname = template_cfg.name
        checkpoint_run = VizRunConfig(
            name=label,
            location=LocalLocation(path=prediction_path.resolve()),
            variables=template_cfg.runs[0].variables,
        )
        updated = template_cfg.model_dump(mode="python")
        updated["base_output_dir"] = eval_output_dir
        updated["name"] = viz_dirname
        updated["runs"] = [
            checkpoint_run.model_dump(mode="python"),
            *[run.model_dump(mode="python") for run in template_cfg.runs[1:]],
        ]
        cfg = VizConfig.model_validate(updated)

        start = time.perf_counter()
        run_with_prepared_groundtruth(cfg, prepared_groundtruth)
        elapsed_seconds = time.perf_counter() - start

    return {
        "output_dir": str(cfg.output_path),
        "prediction_path": str(prediction_path),
        "elapsed_seconds": elapsed_seconds,
    }


def _worker_main(
    entries: list[CheckpointEvalTarget],
    eval_config_args: tuple[str, ...],
    sweep_root: str,
    data_root: object | None,
    gpu_index: int | None,
    queue: multiprocessing.Queue,
) -> None:
    try:
        for entry in entries:
            queue.put(
                _run_single_checkpoint_eval(
                    entry=entry,
                    eval_config_args=eval_config_args,
                    sweep_root=Path(sweep_root),
                    data_root=data_root,
                    gpu_index=gpu_index,
                )
            )
    except Exception as exc:
        queue.put({"error": str(exc), "gpu_index": gpu_index})
        raise


def run_checkpoint_sweep(
    eval_config_args: tuple[str, ...],
    checkpoint_paths: CheckpointPaths,
    data_root: object | None = None,
    sweep_root: Path | None = None,
    viz_config_args: tuple[str, ...] | None = None,
    last_n_checkpoints: int | None = None,
    viz_dirname: str | None = None,
) -> list[dict[str, object]]:
    targets = discover_checkpoints_from_directory(
        checkpoint_paths,
        last_n_checkpoints=last_n_checkpoints,
    )
    if not targets:
        logger.warning("No checkpoints selected for post-train eval sweep")
        return []

    eval_cfg = EvalConfig.from_yaml_and_cli(list(eval_config_args))
    if sweep_root is None:
        sweep_root = checkpoint_paths.checkpoint_dir.parent / eval_cfg.experiment.name
    sweep_root.mkdir(parents=True, exist_ok=True)

    if viz_config_args is not None and not eval_cfg.save_zarr:
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
                    eval_config_args=eval_config_args,
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
                    eval_config_args,
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

    if viz_config_args is not None:
        logger.info("Running post-train viz sweep for %d checkpoints", len(results))
        with MultitonScope():
            template_cfg = VizConfig.from_yaml_and_cli(list(viz_config_args))
            if not template_cfg.runs:
                raise ValueError(
                    "post-train viz config must define at least one run so variables can be inferred"
                )
            prepared_groundtruth = template_cfg.prepare_groundtruth(
                LocalLocation(path=Path.cwd())
            )
        for result in results:
            result["viz"] = _run_single_checkpoint_viz(
                result,
                template_cfg,
                prepared_groundtruth,
                viz_dirname=viz_dirname,
            )
        _write_summary(results, sweep_root)

    return results


def run_post_train_checkpoint_sweep(
    train_cfg: TrainConfig, checkpoint_paths: CheckpointPaths
) -> list[dict[str, object]]:
    if not train_cfg.post_train_eval.enabled:
        return []

    if train_cfg.post_train_eval.eval_config_path is None:
        raise ValueError(
            "post_train_eval.eval_config_path must be set when post_train_eval.enabled is true"
        )

    eval_config_path = (
        Path(train_cfg.post_train_eval.eval_config_path).expanduser().resolve()
    )
    viz_config_args = None
    if train_cfg.post_train_eval.viz_config_path is not None:
        viz_config_path = (
            Path(train_cfg.post_train_eval.viz_config_path).expanduser().resolve()
        )
        viz_config_args = (str(viz_config_path),)
    sweep_root = None
    if train_cfg.post_train_eval.eval_dirname is not None:
        sweep_root = (
            train_cfg.experiment.output_dir / train_cfg.post_train_eval.eval_dirname
        )
    return run_checkpoint_sweep(
        eval_config_args=(str(eval_config_path),),
        checkpoint_paths=checkpoint_paths,
        data_root=train_cfg.experiment.data_root,
        sweep_root=sweep_root,
        viz_config_args=viz_config_args,
        last_n_checkpoints=train_cfg.post_train_eval.last_n_checkpoints,
        viz_dirname=train_cfg.post_train_eval.viz_dirname,
    )


def run_standalone_checkpoint_sweep(
    eval_config_path: Path,
    checkpoint_dir: Path,
    eval_override_args: list[str] | None = None,
    viz_config_path: Path | None = None,
    last_n_checkpoints: int | None = None,
) -> list[dict[str, object]]:
    eval_config_args = (str(eval_config_path), *(eval_override_args or []))
    viz_config_args = (str(viz_config_path),) if viz_config_path is not None else None

    return run_checkpoint_sweep(
        eval_config_args=eval_config_args,
        checkpoint_paths=CheckpointPaths(checkpoint_dir),
        viz_config_args=viz_config_args,
        last_n_checkpoints=last_n_checkpoints,
    )


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
    args, eval_override_args = parser.parse_known_args(argv)

    # Expand ~ in user-provided config paths before resolving them to absolute paths.
    sweep_kwargs: dict[str, object] = {
        "eval_config_path": Path(args.config).expanduser().resolve(),
        "checkpoint_dir": Path(args.checkpoint_dir).expanduser().resolve(),
        "eval_override_args": eval_override_args,
    }
    if args.viz_config is not None:
        sweep_kwargs["viz_config_path"] = Path(args.viz_config).expanduser().resolve()
    if args.last_n_checkpoints is not None:
        sweep_kwargs["last_n_checkpoints"] = args.last_n_checkpoints

    run_standalone_checkpoint_sweep(**sweep_kwargs)


if __name__ == "__main__":
    main()
