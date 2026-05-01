import argparse
import json
import logging
import multiprocessing
import re
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from samudra.config import EvalConfig, PostTrainCheckpointSweepConfig, TrainConfig
from samudra.eval import Eval
from samudra.utils.location import LocalLocation
from samudra.utils.multiton import MultitonScope
from samudra.utils.train import CheckpointPaths
from samudra.viz import VizConfig
from samudra.viz.config import VizRunConfig, main as run_viz

logger = logging.getLogger(__name__)


def _default_viz_dirname() -> str:
    default = PostTrainCheckpointSweepConfig.model_fields["viz_dirname"].default
    if not isinstance(default, str):
        raise TypeError("PostTrainCheckpointSweepConfig.viz_dirname must default to str")
    return default


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
            raise ValueError(f"Checkpoint at {checkpoint_path} is missing integer epoch")
    return CheckpointEvalTarget(
        epoch=epoch,
        kind=kind,
        path=str(checkpoint_path),
        for_inference=for_inference,
    )


def _load_eval_config(eval_config_args: tuple[str, ...]) -> EvalConfig:
    return EvalConfig.from_yaml_and_cli(list(eval_config_args))


def _write_summary(results: list[dict[str, object]], sweep_root: Path) -> None:
    summary_path = sweep_root / "summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _require_result_str(result: dict[str, object], key: str) -> str:
    value = result.get(key)
    if not isinstance(value, str):
        raise ValueError(f"checkpoint sweep result is missing string field '{key}'")
    return value


def _run_single_checkpoint_eval(
    entry: CheckpointEvalTarget,
    eval_config_args: tuple[str, ...],
    sweep_root: Path,
    data_root: object,
    gpu_index: int | None,
) -> dict[str, object]:
    if gpu_index is not None:
        torch.cuda.set_device(gpu_index)

    with MultitonScope():
        cfg = _load_eval_config(eval_config_args)
        cfg.ckpt_path = entry.path
        cfg.experiment.base_output_dir = str(sweep_root)
        cfg.experiment.name = checkpoint_label(entry)
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


def _worker_main(
    entries: list[CheckpointEvalTarget],
    eval_config_args: tuple[str, ...],
    sweep_root: str,
    data_root: object,
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
    data_root: object,
    sweep_root: Path,
    viz_config_args: tuple[str, ...] | None = None,
    last_n_checkpoints: int | None = None,
    viz_dirname: str | None = None,
) -> list[dict[str, object]]:
    if viz_dirname is None:
        viz_dirname = _default_viz_dirname()

    # Discover checkpoints: periodic ckpt_<N>.pt files, plus the EMA checkpoint if present.
    checkpoint_dir = checkpoint_paths.checkpoint_dir
    periodic: list[tuple[int, Path]] = []
    for path in checkpoint_dir.iterdir():
        if not path.is_file():
            continue
        # Match periodic checkpoints named like ckpt_10.pt and capture the epoch.
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

    if not targets:
        logger.warning("No checkpoints selected for post-train eval sweep")
        return []

    sweep_root.mkdir(parents=True, exist_ok=True)

    eval_cfg = _load_eval_config(eval_config_args)
    if viz_config_args is not None and not eval_cfg.save_zarr:
        raise ValueError(
            "post-train viz sweep requires eval.save_zarr = true so predictions.zarr is written"
        )

    # Resolve worker count: one process per available GPU, capped at the number of checkpoints.
    backend = eval_cfg.backend
    wants_gpu = backend in {"auto", "cuda"}
    available_gpus = (
        torch.cuda.device_count() if wants_gpu and torch.cuda.is_available() else 0
    )
    if backend == "cuda" and available_gpus == 0:
        raise RuntimeError("post-train eval requested CUDA but no GPUs are available")
    if available_gpus == 0:
        logger.info("No GPUs available for post-train eval; falling back to serial CPU")
        worker_count = 1
        gpu_indices: list[int] = []
    else:
        worker_count = max(1, min(len(targets), available_gpus))
        gpu_indices = list(range(worker_count))

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
        # Round-robin shard checkpoints across workers.
        shards = [targets[i::worker_count] for i in range(worker_count)]
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
            item = queue.get()
            if "error" in item:
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
        for result in results:
            label = _require_result_str(result, "label")
            eval_output_dir = Path(_require_result_str(result, "output_dir"))
            prediction_path = eval_output_dir / "predictions.zarr"
            if not prediction_path.exists():
                raise FileNotFoundError(
                    f"Expected saved predictions at {prediction_path} for post-train viz sweep"
                )

            with MultitonScope():
                # Build a viz config that points its first run at this checkpoint's predictions,
                # inheriting variables (and any extra runs) from the template.
                template_cfg = VizConfig.from_yaml_and_cli(list(viz_config_args))
                if not template_cfg.runs:
                    raise ValueError(
                        "post-train viz config must define at least one run so variables can be inferred"
                    )
                checkpoint_run = VizRunConfig(
                    name=label,
                    location=LocalLocation(path=prediction_path),
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
                run_viz(cfg)
                elapsed_seconds = time.perf_counter() - start

            result["viz"] = {
                "output_dir": str(cfg.output_path),
                "prediction_path": str(prediction_path),
                "elapsed_seconds": elapsed_seconds,
            }
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

    eval_config_path = Path(train_cfg.post_train_eval.eval_config_path).expanduser().resolve()
    viz_config_args = None
    if train_cfg.post_train_eval.viz_config_path is not None:
        viz_config_path = (
            Path(train_cfg.post_train_eval.viz_config_path).expanduser().resolve()
        )
        viz_config_args = (str(viz_config_path),)
    return run_checkpoint_sweep(
        eval_config_args=(str(eval_config_path),),
        checkpoint_paths=checkpoint_paths,
        data_root=train_cfg.experiment.data_root,
        sweep_root=train_cfg.experiment.output_dir
        / train_cfg.post_train_eval.eval_dirname,
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
    eval_cfg = _load_eval_config(eval_config_args)
    viz_config_args = (str(viz_config_path),) if viz_config_path is not None else None

    eval_dirname_default = PostTrainCheckpointSweepConfig.model_fields["eval_dirname"].default
    if not isinstance(eval_dirname_default, str):
        raise TypeError("PostTrainCheckpointSweepConfig.eval_dirname must default to str")

    return run_checkpoint_sweep(
        eval_config_args=eval_config_args,
        checkpoint_paths=CheckpointPaths(checkpoint_dir),
        data_root=eval_cfg.experiment.data_root,
        sweep_root=checkpoint_dir.parent / eval_dirname_default,
        viz_config_args=viz_config_args,
        last_n_checkpoints=last_n_checkpoints,
        viz_dirname=_default_viz_dirname(),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run standalone inference sweeps for an existing saved_nets directory."
    )
    parser.add_argument("--config",
        required=True,
        help="Path to eval config YAML"
    )
    parser.add_argument(
        "--viz_config",
        required=True,
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
        "viz_config_path": Path(args.viz_config).expanduser().resolve()
    }
    if args.last_n_checkpoints is not None:
        sweep_kwargs["last_n_checkpoints"] = args.last_n_checkpoints

    run_standalone_checkpoint_sweep(**sweep_kwargs)


if __name__ == "__main__":
    main()
