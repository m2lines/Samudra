# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched wall-clock training arms using the normal Trainer training loop."""

import argparse
import configparser
import hashlib
import json
import logging
import os
import socket
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from samudra.config import TrainConfig
from samudra.native_reader import PhysicalVariable
from samudra.rust_reader import RustIoRuntime
from samudra.train import Trainer
from samudra.utils.location import LocalLocation, S3Location, UnresolvedLocation

BASE = Path("/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021")
LOCAL = Path("/scratch/jr7309/data/om4_quarterdeg_v2")
REMOTE = S3Location(
    bucket="emulators",
    path="jr7309/data/testing/om4_quarterdeg_v2_zstd_20261021",
    endpoint_url="https://nyu1.osn.mghpcc.org",
)


def credentials():
    config = configparser.ConfigParser(interpolation=None)
    config.read("/opt/osn-rclone.conf")
    remote = config["nyu-osn-emulators"]
    os.environ["AWS_ACCESS_KEY_ID"] = remote["access_key_id"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = remote["secret_access_key"]
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    if remote.get("session_token"):
        os.environ["AWS_SESSION_TOKEN"] = remote["session_token"]


def preflight():
    credentials()
    import samudra_rust_loader

    print("Native extension:", samudra_rust_loader.__file__, flush=True)
    runtime = RustIoRuntime(8)
    metadata = json.loads((LOCAL / "OM4.zarr/.zmetadata").read_text())["metadata"]
    names = sorted(
        k.split("/")[0]
        for k, v in metadata.items()
        if k.endswith("/.zarray") and v["shape"] == [4745, 720, 1440]
    )
    selectors = [PhysicalVariable(n) for n in names]
    local = runtime.open(LocalLocation(path=LOCAL / "OM4.zarr"), selectors)
    remote_location = REMOTE.resolve(UnresolvedLocation(path="OM4.zarr"))
    assert isinstance(remote_location, S3Location)
    remote = runtime.open(remote_location, selectors)
    # Bound memory and compare every physical field, not only easy surface fields.
    for i in [0, 2372, 4744]:
        for selector in selectors:
            a = np.empty((1, 1, 720, 1440), dtype=np.float32)
            b = np.empty_like(a)
            local.read_into([i], [selector], a)
            remote.read_into([i], [selector], b)
            assert a.tobytes() == b.tobytes(), (i, selector)
    for name in ["OM4_means.zarr", "OM4_stds.zarr"]:
        with (
            LocalLocation(path=LOCAL / name).open({}) as norm_local,
            REMOTE.resolve(UnresolvedLocation(path=name)).open({}) as norm_remote,
        ):
            for field in norm_local.data_vars:
                np.testing.assert_array_equal(
                    norm_local[field].values, norm_remote[field].values
                )
    (BASE / "PREFLIGHT_COMPLETE.json").write_text(
        json.dumps(
            {
                "fields": len(selectors),
                "times": [0, 2372, 4744],
                "normalizations_equal": True,
            }
        )
        + "\n"
    )
    print("Real local/S3 parity passed.", flush=True)


class BudgetReached(Exception):
    pass


def main(arm, seconds, warmup):
    credentials()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    cfg = TrainConfig.from_yaml("samudra_om4/train.yaml")
    cfg.backend = "nccl"
    cfg.batch_size = 1
    cfg.gradient_accumulation_steps = 1
    cfg.steps = [4]
    cfg.rollout_validation = None
    cfg.post_train_eval = None
    cfg.experiment.rand_seed = 20261021
    cfg.experiment.name = arm
    cfg.experiment.base_output_dir = str(BASE / ("runs-" + os.environ["SLURM_JOB_ID"]))
    cfg.experiment.data_root = LocalLocation(path=LOCAL) if arm == "local" else REMOTE
    cfg.experiment.wandb.mode = "disabled"
    cfg.data.sources[0].boundary_vars_key = "tau_hfds"
    # Config validation constructs the proper concrete loading config.
    cfg_data = cfg.model_dump(mode="json")
    cfg_data["data"]["loading"] = dict(
        type="rust", max_concurrent_reads=8, prefetch_batches=2, prefetch_to_device=True
    )
    cfg = TrainConfig.model_validate(cfg_data)
    started = time.perf_counter()
    trainer = Trainer(cfg)
    assert dist.get_world_size() == 8
    rank = dist.get_rank()
    output = cfg.experiment.output_dir
    trainer.init_data_loaders(trainer.get_current_step(1))
    trainer.profiler.start()
    digest = hashlib.sha256()
    for name, value in trainer.model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    initial_hash = digest.hexdigest()
    trainer.train_sampler.set_epoch(1)
    sampler_digest = hashlib.sha256(
        repr(list(trainer.train_sampler)).encode()
    ).hexdigest()
    dist.barrier()
    startup_seconds = time.perf_counter() - started
    pending = {}
    stream = (output / f"batches-rank{rank}.jsonl").open("w", buffering=1)
    state: dict[str, float] = {
        "previous": time.perf_counter(),
        "timed_start": 0.0,
        "timed_batches": 0,
        "timed_samples": 0,
        "timed_examples": 0,
    }
    original_log = trainer.wandb_logger.log

    def capture(metrics, step=None, **kwargs):
        for key, value in metrics.items():
            if key.startswith("progress/") or key in [
                "train/batch/loss",
                "train/batch/data_wait_time",
                "train/batch/data_load_time",
            ]:
                pending[key] = (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
        original_log(metrics, step=step, **kwargs)

    setattr(trainer.wandb_logger, "log", capture)
    original_after = trainer.profiler.after_batch

    def after_batch(step):
        original_after(step)
        torch.cuda.synchronize()
        now = time.perf_counter()
        measured = step > warmup
        row = dict(
            step=step,
            rank=rank,
            measured=measured,
            unix_time=time.time(),
            wall_seconds=now - state["previous"],
            **pending,
        )
        if measured:
            state["timed_batches"] += 1
            state["timed_samples"] += pending["progress/batch_sample_windows"]
            state["timed_examples"] += pending["progress/batch_model_examples"]
        if step == warmup:
            state["timed_start"] = now
            state["timed_start_unix"] = row["unix_time"]
        stream.write(json.dumps(row) + "\n")
        state["previous"] = now
        stop = torch.tensor(
            int(
                rank == 0
                and state["timed_start"] > 0
                and now - state["timed_start"] >= seconds
            ),
            device=trainer.device,
        )
        dist.broadcast(stop, src=0)
        if stop.item():
            raise BudgetReached

    setattr(trainer.profiler, "after_batch", after_batch)
    try:
        for epoch in range(1, cfg.epochs + 1):
            trainer.train_sampler.set_epoch(epoch)
            trainer.train_one_epoch(epoch)
        raise RuntimeError("Benchmark exhausted its epoch limit")
    except BudgetReached:
        elapsed = time.perf_counter() - state["timed_start"]
        report = dict(
            arm=arm,
            rank=rank,
            host=socket.gethostname(),
            job_id=os.environ["SLURM_JOB_ID"],
            initial_state_sha256=initial_hash,
            sampler_sha256=sampler_digest,
            startup_seconds=startup_seconds,
            warmup_batches=warmup,
            measured_seconds=elapsed,
            measured_start_unix=state["timed_start_unix"],
            measured_end_unix=time.time(),
            measured_batches=state["timed_batches"],
            global_sample_windows=state["timed_samples"],
            global_model_examples=state["timed_examples"],
            sample_windows_per_second=state["timed_samples"] / elapsed,
            model_examples_per_second=state["timed_examples"] / elapsed,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        )
        (output / f"COMPLETE-rank{rank}.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        if rank == 0:
            print("BENCHMARK_COMPLETE", json.dumps(report), flush=True)
    finally:
        stream.close()
        trainer.train_loader.close()
        trainer.val_loader.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--arm", choices=["local", "remote"])
    p.add_argument("--seconds", type=float, default=3600)
    p.add_argument("--warmup", type=int, default=5)
    a = p.parse_args()
    if a.preflight:
        preflight()
    else:
        main(a.arm, a.seconds, a.warmup)
