# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded Rust/prefetch/DDP qualification with matched checkpoint comparisons."""

import gc
import hashlib
import json
import math
import os
import resource
import subprocess
import time
from collections.abc import Sequence

import torch
import torch.distributed as dist

from samudra.backend import init_train_backend
from samudra.observations import training as baseline
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.constants import FORECAST_STEPS
from samudra.observations.pipeline import World, pin_host_tensor
from samudra.observations.qualify import equal_tree
from samudra.observations.state import TrainingState, write_json
from samudra.observations.train import build_sources, init_tracking
from samudra.utils.distributed import set_seed


def input_parity(cfg, native):
    """Compare real fresh and shifted replay windows, including all loss tensors."""
    config = cfg.model_copy(deep=True)
    config.data.loading = None
    python = build_sources(config, torch.device("cpu"))
    checks = []
    origins: dict[str, Sequence[str | int]] = {"obs": ["1993-05", "2001-01"]}
    if "om4" in native.models:
        origins["om4"] = [
            index
            for index in range(len(native.models["om4"].windows))
            if native.layout("om4", index, 1) is not None
        ][:2]
    for source, starts in origins.items():
        for origin in starts:
            for month in (0, 1):
                actual = dict(native.load(source, origin, month))
                expected = dict(python.load(source, origin, month))
                keys = [
                    key for key in expected if isinstance(expected[key], torch.Tensor)
                ]
                for key in keys:
                    torch.testing.assert_close(
                        actual[key], expected[key], rtol=0, atol=0, equal_nan=True
                    )
                checks.append(
                    dict(source=source, origin=origin, month=month, tensors=keys)
                )
    return checks


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        )
    return digest.hexdigest()


def nvidia_smi_uuid(value):
    """NVML needs GPU- before the UUID returned by PyTorch's CUDA properties."""
    value = str(value)
    return value if value.startswith("GPU-") else f"GPU-{value}"


def verify_telemetry_devices(uuids):
    actual = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
            f"--id={','.join(uuids)}",
        ],
        text=True,
        timeout=15,
    )
    if sorted(actual.split()) != sorted(uuids):
        raise RuntimeError(f"GPU telemetry device mismatch: {actual!r}")


def run(cfg):
    if not cfg.update_prefetch or cfg.data.loading is None:
        raise ValueError(
            "Scaling qualification requires update_prefetch and Rust loading"
        )
    if (
        cfg.gradient_accumulation_steps != 8
        or cfg.main_updates < 32
        or cfg.cooldown_updates != 2
    ):
        raise ValueError(
            "Qualify global batch eight, at least 32 main and two cooldown updates"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    size = int(os.environ.get("WORLD_SIZE", "1"))
    device, distributed = init_train_backend("nccl" if size > 1 else "cuda")
    device = torch.device("cuda", torch.cuda.current_device())
    world = World(device)
    output = cfg.experiment.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if world.broadcast(
        (output / "QUALIFIED.json").exists() if world.rank == 0 else None
    ):
        world.close()
        if distributed is not None:
            dist.destroy_process_group()
        return
    set_seed(cfg.experiment.rand_seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Exercise CUDA host staging before expensive archive checks/model setup.
    # OM4 validity broadcasts a spatial mask over the time dimension.
    spatial_mask = torch.arange(12).reshape(1, 1, 3, 4).remainder(2).bool()
    expanded_mask = spatial_mask.expand(1, 26, 3, 4)
    pinned = pin_host_tensor(expanded_mask)
    if not pinned.is_pinned():
        raise AssertionError("Host staging did not allocate pinned memory")
    torch.testing.assert_close(
        pinned.to(device, non_blocking=True).cpu(), expanded_mask, rtol=0, atol=0
    )
    if world.rank == 0:
        print(json.dumps({"event": "expanded_mask_cuda_staging_passed"}), flush=True)
    uuid = nvidia_smi_uuid(torch.cuda.get_device_properties(device).uuid)
    uuids = [uuid]
    if world.size > 1:
        uuids = [""] * world.size
        dist.all_gather_object(uuids, uuid, group=world.group)
    if world.rank == 0:
        verify_telemetry_devices(uuids)
    world.broadcast(True)
    started = time.time()
    sources = build_sources(cfg, torch.device("cpu"), verify_data=world.rank == 0)
    manifest = world.broadcast(
        baseline.run_manifest(cfg, sources.obs, sources) if world.rank == 0 else None
    )
    if world.rank == 0:
        path = output / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError("Scaling output belongs to different code/data/config")
        write_json(manifest, path)
        cfg.save_yaml(output / "config.yaml")
        parity_path = output / "input-parity.json"
        if not parity_path.exists():
            write_json(input_parity(cfg, sources), parity_path)
    world.broadcast(True)
    data = sources.obs.to(device)
    model = cfg.model.build(data.grid["names"].tolist()).to(device)
    state = TrainingState(
        model,
        torch.optim.AdamW(
            model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        ),
        sources.names,
        cfg.experiment.rand_seed,
        ema_decay=cfg.ema_decay,
        faster_decay_at_start=cfg.faster_decay_at_start,
        replay_per_month=cfg.replay.per_month,
        replay_max_age=cfg.replay.max_age,
    )
    tracker = init_tracking(cfg, manifest) if world.rank == 0 else None
    records = []
    stage = "warmup"
    telemetry = telemetry_file = None
    if world.rank == 0:
        telemetry_file = (output / "gpu-utilization.csv").open("a")
        telemetry = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used",
                "--format=csv,noheader,nounits",
                "--loop-ms=1000",
                f"--id={','.join(uuids)}",
            ],
            stdout=telemetry_file,
        )

    def record(value):
        if telemetry is not None and telemetry.poll() is not None:
            raise RuntimeError("GPU utilization telemetry stopped unexpectedly")
        event = {"stage": stage, "timestamp": time.time(), **value}
        records.append(event)
        with (output / "scaling.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event, allow_nan=False), flush=True)
        if tracker is not None:
            tracker.log(event, step=len(records))

    def validate(current):
        trajectory = data.trajectory(
            cfg.validation_origin, steps=FORECAST_STEPS, split="validation"
        )
        metrics = baseline.annual_metrics(current, data, trajectory)
        if metrics["rollout_val/365d/scored_channels"] != 29 or not all(
            math.isfinite(x) for x in metrics.values()
        ):
            raise AssertionError("Invalid annual observation metrics")
        return metrics

    def train_until(name, limit, *, prefetch=True):
        nonlocal stage
        stage = name
        directory = output / name
        directory.mkdir(exist_ok=True)
        config = cfg.model_copy(update={"prefetch_enabled": prefetch})
        baseline.train(
            config,
            state,
            sources,
            directory,
            manifest,
            validate,
            record,
            stop_after=limit,
        )

    try:
        # Each stage repeats from a saved anchor after interruption. No unbounded
        # retraining or checkpoint from another world size is accepted.
        anchor = output / "warmup/ckpt.pt"
        if anchor.exists():
            state.restore(anchor, manifest)
        else:
            train_until("warmup", 4)
        for name, enabled in (("sync", False), ("async", True), ("sync-repeat", False)):
            state.restore(anchor, manifest)
            train_until(name, 20, prefetch=enabled)
        if world.rank == 0:
            expected = torch.load(
                output / "async/ckpt.pt", map_location="cpu", weights_only=False
            )
            for name in ("sync", "sync-repeat"):
                actual = torch.load(
                    output / name / "ckpt.pt", map_location="cpu", weights_only=False
                )
                equal_tree(actual, expected)
                del actual
            del expected
            gc.collect()
            record({"event": "exact_prefetch_and_resume", "updates": 20})
        world.broadcast(True)
        state.restore(output / "async/ckpt.pt", manifest)
        train_until("steady", cfg.main_updates)
        main_path = output / "steady/main.pt"
        parent_sha = baseline.digest(main_path) if world.rank == 0 else None
        fork = cfg.model_copy(
            update={
                "main_updates": 2 * cfg.main_updates,
                "milestone_updates": [2 * cfg.main_updates],
            }
        )
        fork_manifest = {**manifest, "config": fork.scientific_config()}
        state.restore(main_path, fork_manifest, fork=True)
        if state.cooled or state.updates != cfg.main_updates:
            raise AssertionError("Main continuation inherited cooldown")
        state.restore(main_path, manifest)
        train_until("cooldown", cfg.total_updates)
        if not state.cooled or state.ema.num_updates != 2:
            raise AssertionError("Cooldown did not reset EMA exactly once")
        hashes = [model_digest(model)]
        if world.size > 1:
            hashes = [None] * world.size
            dist.all_gather_object(hashes, model_digest(model), group=world.group)
        if len(set(hashes)) != 1:
            raise AssertionError("DDP ranks disagree on final model weights")
        memory = {
            "rank": world.rank,
            "uuid": uuid,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
        memories = [memory]
        if world.size > 1:
            memories = [None] * world.size
            dist.all_gather_object(memories, memory, group=world.group)
        if world.rank == 0:
            if baseline.digest(main_path) != parent_sha:
                raise AssertionError("Cooldown modified uncooled parent checkpoint")
            if telemetry.poll() is not None:
                raise RuntimeError("GPU utilization telemetry stopped unexpectedly")
            report = {
                "purpose": "engineering qualification",
                "world_size": world.size,
                "global_examples": 8,
                "gpu": torch.cuda.get_device_name(device),
                "torch": str(torch.__version__),
                "cuda": torch.version.cuda,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "parameters": sum(p.numel() for p in model.parameters()),
                "exact_prefetch_and_resume": True,
                "isolated_uncooled_fork": True,
                "rank_model_sha256": hashes,
                "memory": memories,
                "exposure": state.exposure,
                "training_records": records,
                "started_at": started,
                "finished_at": time.time(),
                "wandb_url": tracker.run.url if tracker and tracker.run else None,
                "utilization_release_gate": "Assess per-GPU steady telemetry; this receipt does not assert >50%",
            }
            write_json(report, output / "QUALIFIED.json")
            record({"event": "qualified", "updates": state.updates})
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)
            assert telemetry_file is not None
            telemetry_file.close()
        if tracker is not None:
            tracker.finish()
        world.close()
        if distributed is not None:
            dist.destroy_process_group()


if __name__ == "__main__":
    run(ObservationTrainConfig.from_yaml_and_cli())
