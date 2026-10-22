# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Run three independent baseline recipes, then save their selected-checkpoint rollouts.

One full Beta node supplies three disjoint 40-CPU/single-GPU workers. The fourth
GPU is idle: the production recipe is deliberately single-process, not DDP.
Scientific budgets remain the visible constants in observations/baseline.py.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from types import FrameType

from samudra.observations.baseline import COOLDOWN_UPDATES, MAIN_UPDATES, digest
from samudra.observations.constants import STEP_DAYS

MIXES = ("obs", "om4-obs", "om4-cm4-obs")
CPUS_PER_WORKER = 40
PROJECT_ROOT = Path("/projects/ny/lz1955/multiscale/jrusak")
WAND_B_URL = re.compile(r"https://wandb\.ai/[^\s/]+/[^\s/]+/runs/[A-Za-z0-9]+")


@dataclass
class Worker:
    """One independently seeded process and its log cursor."""

    mix: str
    process: subprocess.Popen[str]
    log: Path
    offset: int = 0


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def saved_rollout(output: Path, checkpoint: Path, origin: str, steps: int) -> bool:
    """Reuse only a completed matching export; never overwrite partial archives."""
    if not output.exists():
        return False
    receipt_path = output / "COMPLETE.json"
    if not receipt_path.exists():
        raise RuntimeError(
            f"Partial export at {output}; preserve it and choose a new output"
        )
    receipt = json.loads(receipt_path.read_text())
    start = datetime.fromisoformat(origin)
    expected = {
        "steps": steps,
        "start": start.isoformat(),
        "end_exclusive": (start + timedelta(days=STEP_DAYS * steps)).isoformat(),
        "checkpoint_sha256": digest(checkpoint),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError(
            f"Saved rollout does not match requested checkpoint/interval: {output}"
        )
    if not receipt.get("verification") or not (output / "predictions.zarr").is_dir():
        raise ValueError(f"Missing verified prediction archive at {output}")
    return True


def run_stage(
    stage: str,
    commands: list[list[str]],
    output: Path,
    devices: list[str],
    cpu_sets: list[list[int]],
) -> None:
    """Fail the campaign promptly and ask remaining trainers to save on failure."""
    workers: list[Worker] = []
    interrupted = False

    def stop(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        interrupted = True
        for worker in workers:
            process = worker.process
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)

    previous = {
        sig: signal.signal(sig, stop)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)
    }
    try:
        for mix, command, device, cpus in zip(
            MIXES, commands, devices, cpu_sets, strict=True
        ):
            if interrupted:
                raise RuntimeError(f"{stage} interrupted during worker startup")
            if not command:
                continue  # A matching completed rollout was already verified.
            arm = output / mix
            arm.mkdir(exist_ok=True)
            log_path = arm / f"{stage}.log"
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=device,
                WANDB_MODE="online",
                WANDB_RUN_GROUP=output.name,
                WANDB_DIR=str(arm.resolve()),
            )
            with log_path.open("a") as log:
                process = subprocess.Popen(
                    ["taskset", "--cpu-list", ",".join(map(str, cpus)), *command],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
            workers.append(Worker(mix, process, log_path))
            if interrupted:
                # A signal may arrive while Popen is creating the child, before
                # the handler can find it in workers. Finally terminates/reaps it.
                raise RuntimeError(f"{stage} interrupted while spawning {mix}")
            write_json(
                arm / f"{stage}-process.json",
                {
                    "pid": process.pid,
                    "gpu": device,
                    "cpus": cpus,
                    "command": command,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                },
            )
        failed = False
        while True:
            codes = [worker.process.poll() for worker in workers]
            for worker in workers:
                # Record the real online URL once W&B initialization emits it.
                with worker.log.open() as log:
                    log.seek(worker.offset)
                    text = log.read()
                    worker.offset = log.tell()
                urls = WAND_B_URL.findall(text)
                if urls:
                    (output / worker.mix / "wandb-url.txt").write_text(urls[-1] + "\n")
            incomplete = False
            if stage == "train":
                for worker, code in zip(workers, codes, strict=True):
                    if code == 0:
                        receipt = output / worker.mix / "COMPLETE.json"
                        incomplete |= (
                            not receipt.exists()
                            or json.loads(receipt.read_text()).get("updates")
                            != MAIN_UPDATES + COOLDOWN_UPDATES
                        )
            if not failed and (
                incomplete or any(code not in (None, 0) for code in codes)
            ):
                failed = True
                stop(signal.SIGTERM, None)
            if all(code is not None for code in codes):
                break
            time.sleep(5)
        statuses: dict[str, int | None] = dict.fromkeys(MIXES, 0)
        statuses.update(
            {worker.mix: code for worker, code in zip(workers, codes, strict=True)}
        )
        write_json(output / f"{stage}-status.json", statuses)
        if failed or interrupted:
            raise RuntimeError(f"{stage} stopped: {statuses}; inspect per-arm logs")
    finally:
        for worker in workers:
            process = worker.process
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for worker in workers:
            worker.process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--observations",
        type=Path,
        default=PROJECT_ROOT / "data/obs_full_range/d-observation-pilot/samples",
    )
    parser.add_argument(
        "--daily",
        type=Path,
        default=PROJECT_ROOT / "data/obs_full_range/d-observation-pilot/daily",
    )
    parser.add_argument(
        "--normalization",
        type=Path,
        default=PROJECT_ROOT
        / "runs/observation-baseline-global-20261022/normalization",
    )
    parser.add_argument(
        "--om4",
        type=Path,
        default=PROJECT_ROOT / "data/diffusion-interior-beta/data/om4_onedeg_v3",
    )
    parser.add_argument(
        "--cm4", type=Path, default=PROJECT_ROOT / "data/cm4_onedeg/CM4.zarr"
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=PROJECT_ROOT
        / "data/diffusion-interior-beta/data/om4_onedeg_v3/OM4.zarr",
    )
    parser.add_argument(
        "--smoke-checkpoint",
        type=Path,
        default=PROJECT_ROOT
        / "runs/observation-baseline-global-20261022/qualification/best.pt",
    )
    parser.add_argument("--entity", default="ocean_emulators")
    parser.add_argument("--project", default="observational-transfer")
    parser.add_argument("--origin", default="2015-01-01")
    parser.add_argument("--end-exclusive", default="2023-01-01")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    available_cpus = sorted(os.sched_getaffinity(0))
    if len(available_cpus) < CPUS_PER_WORKER * len(MIXES):
        raise RuntimeError("Campaign requires at least 120 allocated CPUs")
    import torch

    if torch.cuda.device_count() != 4:
        raise RuntimeError("Campaign requires the full four-GPU Beta allocation")
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")[:3]
    if len(devices) != len(MIXES):
        raise RuntimeError("Cannot identify three assigned CUDA devices")
    cpu_sets = [
        available_cpus[i * CPUS_PER_WORKER : (i + 1) * CPUS_PER_WORKER]
        for i in range(3)
    ]
    write_json(
        args.output / "startup.json",
        {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "mixes": MIXES,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(4)],
            "devices_used": devices,
            "cpu_sets": cpu_sets,
            "main_updates": MAIN_UPDATES,
            "cooldown_updates": COOLDOWN_UPDATES,
            "wandb_entity": args.entity,
            "wandb_project": args.project,
        },
    )
    common = [
        "--observations",
        str(args.observations),
        "--daily",
        str(args.daily),
        "--normalization",
        str(args.normalization),
    ]
    exporter = Path(__file__).with_name("rollout_observation_baseline.py")
    if not exporter.is_file():
        raise FileNotFoundError(exporter)
    # Direct script execution places the sibling entrypoint on sys.path.
    from rollout_observation_baseline import (  # type: ignore[import-not-found]
        forecast_intervals,
        preflight_archive,
    )

    rollout_starts, rollout_ends = forecast_intervals(
        args.origin, end_exclusive=args.end_exclusive
    )
    coverage = preflight_archive(args.daily, rollout_starts[0], rollout_ends[-1])
    write_json(args.output / "rollout-input-coverage.json", coverage)
    # Check archive writing on an already-qualified model before starting costly
    # training. This two-step smoke is not an evaluation or a trained result.
    smoke = [
        sys.executable,
        str(exporter),
        *common,
        "--geometry",
        str(args.geometry),
        "--checkpoint",
        str(args.smoke_checkpoint),
        "--output",
        str(args.output / "export-smoke"),
        "--origin",
        "2015-01-01",
        "--steps",
        "2",
    ]
    if not saved_rollout(
        args.output / "export-smoke", args.smoke_checkpoint, "2015-01-01", 2
    ):
        with (args.output / "export-smoke.log").open("a") as log:
            subprocess.run(
                ["taskset", "--cpu-list", ",".join(map(str, cpu_sets[0])), *smoke],
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=devices[0]),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not saved_rollout(
            args.output / "export-smoke", args.smoke_checkpoint, "2015-01-01", 2
        ):
            raise RuntimeError("Exporter smoke returned without a verified archive")
    training = [
        [
            sys.executable,
            "-m",
            "samudra.observations.baseline",
            "train",
            *common,
            "--mix",
            mix,
            "--output",
            str(args.output / mix),
            "--om4",
            str(args.om4),
            "--cm4",
            str(args.cm4),
            "--wandb-mode",
            "online",
            "--entity",
            args.entity,
            "--project",
            args.project,
        ]
        for mix in MIXES
    ]
    run_stage("train", training, args.output, devices, cpu_sets)
    for mix in MIXES:
        receipt = json.loads((args.output / mix / "COMPLETE.json").read_text())
        if receipt["updates"] != MAIN_UPDATES + COOLDOWN_UPDATES:
            raise RuntimeError(f"Incomplete training receipt for {mix}")
        if not (args.output / mix / "wandb-url.txt").exists():
            raise RuntimeError(f"Missing online W&B URL for {mix}")
    rollouts = [
        [
            sys.executable,
            str(exporter),
            *common,
            "--geometry",
            str(args.geometry),
            "--checkpoint",
            str(args.output / mix / "best.pt"),
            "--output",
            str(args.output / mix / "rollout-best"),
            "--origin",
            args.origin,
            "--end-exclusive",
            args.end_exclusive,
        ]
        for mix in MIXES
    ]
    rollout_steps = len(rollout_starts)
    for index, mix in enumerate(MIXES):
        if saved_rollout(
            args.output / mix / "rollout-best",
            args.output / mix / "best.pt",
            args.origin,
            rollout_steps,
        ):
            rollouts[index] = []
    run_stage("rollout", rollouts, args.output, devices, cpu_sets)
    for mix in MIXES:
        if not saved_rollout(
            args.output / mix / "rollout-best",
            args.output / mix / "best.pt",
            args.origin,
            rollout_steps,
        ):
            raise RuntimeError(
                f"Exporter returned without a verified archive for {mix}"
            )
    write_json(
        args.output / "COMPLETE.json",
        {
            "mixes": MIXES,
            "training_updates": MAIN_UPDATES + COOLDOWN_UPDATES,
            "rollout_origin": args.origin,
            "requested_rollout_end_exclusive": args.end_exclusive,
            "actual_rollout_end_exclusive": rollout_ends[-1].isoformat(),
            "omitted_trailing_days": (
                datetime.fromisoformat(args.end_exclusive)
                - rollout_ends[-1].to_pydatetime()
            ).days,
        },
    )


if __name__ == "__main__":
    main()
