# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Observation training with Samudra YAML configs, logging and launch conventions."""

import json
import logging
import os
import signal
from pathlib import Path
from types import FrameType

import torch
import wandb

from samudra.backend import init_train_backend
from samudra.observations.config import MIXES, ObservationTrainConfig
from samudra.observations.constants import FORECAST_STEPS
from samudra.observations.model_data import ModelMonths
from samudra.observations.samples import Samples
from samudra.observations.shared_era5 import SharedEra5
from samudra.observations.state import TrainingState, write_json
from samudra.observations.training import Sources, annual_metrics, run_manifest, train
from samudra.utils.distributed import set_seed
from samudra.utils.logging import handle_logging, handle_warnings
from samudra.utils.train import CheckpointPaths
from samudra.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


def require_single_process() -> None:
    """Replay/checkpoint state is process-local until rank-aware ownership is added."""
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    if world_size != 1:
        raise ValueError(
            "Observation training currently requires one process/GPU per run"
        )


def run_training(cfg: ObservationTrainConfig) -> None:
    if not cfg.update_prefetch:
        require_single_process()
    cfg.prepare_output_dirs()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if cfg.gradient_accumulation_steps % world_size:
        raise ValueError("Global example count must be divisible by world size")
    device, distributed = init_train_backend("nccl" if world_size > 1 else cfg.backend)
    try:
        _run_training(cfg, device)
    finally:
        if distributed is not None:
            torch.distributed.destroy_process_group()


def build_sources(
    cfg: ObservationTrainConfig, device: torch.device, *, verify_data=True
) -> Sources:
    def path(name: str) -> Path:
        return cfg.data.local_path(name, cfg.experiment)

    runtime, native_archive = None, None
    if cfg.data.loading is not None:
        from samudra.observations.native import NativeObservationArchive
        from samudra.rust_reader import RustIoRuntime

        runtime = RustIoRuntime(cfg.data.loading.max_concurrent_reads)
        native_archive = NativeObservationArchive(
            path("daily"),
            path("native_observations"),
            runtime,
            cfg.data.native_observations_proof_sha256,
            verify_content=verify_data,
        )
    data = Samples(
        path("observations"),
        device,
        daily=path("daily"),
        normalization=path("normalization"),
        native_archive=native_archive,
    )
    models = {}
    if "om4" in MIXES[cfg.mix]:
        models["om4"] = ModelMonths(
            path("om4") / "OM4.zarr", "om4", data.grid, path("om4"), runtime=runtime
        )
        if cfg.model.forcing_mode == "shared_era5":
            models["om4"].era5 = SharedEra5(
                path("era5_cache"), models["om4"], data, cfg.data.era5_proof_sha256
            )
    if "cm4" in MIXES[cfg.mix]:
        models["cm4"] = ModelMonths(path("cm4"), "cm4", data.grid, path("om4"))
    return Sources(data, models)


def init_tracking(cfg: ObservationTrainConfig, manifest: dict) -> WandBLogger:
    output = cfg.experiment.output_dir
    tracker = WandBLogger.init_instance()
    tracker.configure(
        enabled=cfg.experiment.wandb.mode == "online", is_main_process=True
    )
    # DUPLICATED: setup_run assumes DataBundle and Trainer checkpoint keys.
    # Keep run identity here while sharing the existing logger and WandBConfig.
    id_path = output / "wandb-id.txt"
    if not id_path.exists():
        id_path.write_text(wandb.util.generate_id())
    tracker.init(
        **cfg.experiment.wandb.model_dump(),
        id=id_path.read_text().strip(),
        resume="allow",
        name=cfg.experiment.name,
        dir=output,
        config={"config": cfg.model_dump(mode="json"), "provenance": manifest},
    )
    if cfg.experiment.wandb.mode == "online" and tracker.run is None:
        raise RuntimeError(
            "Online W&B initialization failed; inspect the preceding error"
        )
    return tracker


def _run_training(cfg: ObservationTrainConfig, device: torch.device) -> None:
    set_seed(cfg.experiment.rand_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    output = cfg.experiment.output_dir
    nets = cfg.experiment.nets_dir

    distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if distributed else 0
    sources = build_sources(
        cfg,
        torch.device("cpu") if cfg.update_prefetch else device,
        verify_data=rank == 0,
    )
    data = sources.obs.to(device) if cfg.update_prefetch else sources.obs
    manifest = run_manifest(cfg, sources.obs, sources) if rank == 0 else None
    if distributed:
        shared = [manifest]
        torch.distributed.broadcast_object_list(shared, src=0)
        manifest = shared[0]
    assert manifest is not None
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Output directory belongs to another code/data/recipe manifest"
        )
    model = cfg.model.build(data.grid["names"].tolist()).to(device)
    # DUPLICATED: Trainer currently constructs Adam inline. This entrypoint uses
    # AdamW; a shared optimizer config/factory can replace both constructors.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    state = TrainingState(
        model,
        optimizer,
        sources.names,
        cfg.experiment.rand_seed,
        ema_decay=cfg.ema_decay,
        faster_decay_at_start=cfg.faster_decay_at_start,
        replay_per_month=cfg.replay.per_month,
        replay_max_age=cfg.replay.max_age,
    )
    checkpoints = CheckpointPaths(nets)
    checkpoint = (
        Path(cfg.resume_ckpt_path)
        if cfg.resume_ckpt_path
        else checkpoints.latest_checkpoint_path
    )
    if cfg.resume_ckpt_path or checkpoint.exists():
        state.restore(checkpoint, manifest)
    elif cfg.fork_ckpt_path:
        state.restore(cfg.fork_ckpt_path, manifest, fork=True)
        from samudra.observations.training import digest

        if rank == 0:
            write_json(
                {
                    "parent": cfg.fork_ckpt_path,
                    "sha256": digest(cfg.fork_ckpt_path),
                    "updates": state.updates,
                },
                output / "lineage.json",
            )
    if rank == 0:
        cfg.save_yaml(output / "config.yaml")
        write_json(manifest, manifest_path)
    trajectory = (
        data.trajectory(cfg.validation_origin, steps=FORECAST_STEPS, split="validation")
        if rank == 0
        else None
    )
    tracker = init_tracking(cfg, manifest) if rank == 0 else None
    stopped = False

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopped
        stopped = True

    previous = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGTERM, signal.SIGUSR1)
    }

    def emit(record: dict[str, float | int]) -> None:
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        if tracker is not None:
            tracker.log(record, step=int(record["updates"]))
        logger.info("%s", json.dumps(record, allow_nan=False))

    def validate(current):
        assert trajectory is not None
        return annual_metrics(current, data, trajectory)

    try:
        train(
            cfg,
            state,
            sources,
            nets,
            manifest,
            validate,
            emit,
            stop_after=cfg.stop_after,
            stopping=lambda: stopped,
        )
    finally:
        if tracker is not None:
            tracker.finish()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    cfg = ObservationTrainConfig.from_yaml_and_cli()
    if not cfg.update_prefetch:
        require_single_process()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    try:
        run_training(cfg)
    except Exception:
        logger.exception("Observation training failed")
        raise


if __name__ == "__main__":
    main()
