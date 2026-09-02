# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Resource-aware execution in the current machine or Slurm allocation."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import SimpleQueue
from typing import Any

import torch

from samudra.search.config import LocalExecutorConfig
from samudra.search.executors.base import Executor
from samudra.utils.multiton import MultitonScope


@dataclass(frozen=True)
class _Task:
    rung: int
    task: int
    anchor: bool


def _visible_devices() -> list[str]:
    """Return CUDA identifiers that can be assigned to child processes."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        if configured.strip() in {"", "-1"}:
            return []
        return [value.strip() for value in configured.split(",") if value.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


def _positive_int_environment(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _slurm_gpu_count() -> int:
    """Best available count for a homogeneous Slurm GPU allocation."""
    total = _positive_int_environment("SLURM_GPUS")
    if total is not None:
        return total
    nodes = _positive_int_environment("SLURM_NNODES") or 1
    per_node = _positive_int_environment("SLURM_GPUS_ON_NODE")
    if per_node is None:
        per_node = len(_visible_devices())
    if per_node < 1:
        raise RuntimeError("The Slurm allocation does not expose any GPUs")
    return nodes * per_node


def _slurm_cpus_per_gpu() -> int:
    cpus = _positive_int_environment("SLURM_CPUS_ON_NODE")
    gpus = _positive_int_environment("SLURM_GPUS_ON_NODE")
    if cpus is None or gpus is None:
        return 1
    return max(1, cpus // gpus)


class LocalExecutor(Executor):
    @property
    def config(self) -> LocalExecutorConfig:
        config = self.search.config.executor
        if not isinstance(config, LocalExecutorConfig):
            raise TypeError("LocalExecutor requires a LocalExecutorConfig")
        return config

    def submit_initial(self, state: dict[str, Any]) -> None:
        """Co-schedule fixed anchors and rung zero across all GPUs."""
        tasks = [
            _Task(len(self.search.rungs) - 1, task, True)
            for task, _ in enumerate(state["anchors"]["candidates"])
        ]
        tasks.extend(
            _Task(0, task, False)
            for task, _ in enumerate(state["rungs"][0]["candidates"])
        )
        state["anchors"]["job_id"] = "local"
        state["rungs"][0]["job_id"] = "local"
        state["status"] = "running"
        self.search.write_state(state)
        if self.config.dry_run:
            return
        self._run_tasks(tasks)
        self.search.advance(0)

    def submit_anchors(self, state: dict[str, Any]) -> None:
        candidates = state["anchors"]["candidates"]
        state["anchors"]["job_id"] = "local"
        self.search.write_state(state)
        if self.config.dry_run:
            return
        self._run_tasks(
            [
                _Task(len(self.search.rungs) - 1, task, True)
                for task, _ in enumerate(candidates)
            ]
        )

    def submit_rung(self, state: dict[str, Any], rung: int) -> None:
        candidates = state["rungs"][rung]["candidates"]
        state["rungs"][rung]["job_id"] = "local"
        state["status"] = "running"
        self.search.write_state(state)
        if self.config.dry_run:
            return
        self._run_tasks([_Task(rung, task, False) for task, _ in enumerate(candidates)])
        self.search.advance(rung)

    def _run_tasks(self, tasks: list[_Task]) -> None:
        if not tasks:
            return
        if "SLURM_JOB_ID" in os.environ:
            step = os.environ.get("SLURM_STEP_ID")
            if step not in {None, "", "batch", "extern"}:
                raise RuntimeError(
                    "Cannot launch local search workers from inside an existing "
                    "Slurm job step. Run the search directly in an sbatch script "
                    "or an salloc shell, rather than with `srun python`."
                )
            concurrency = _slurm_gpu_count()
            runner: Callable[[_Task], None] = self._run_slurm_task
        else:
            devices = _visible_devices()
            if not devices:
                for task in tasks:
                    self._run_in_process(task)
                return
            concurrency = len(devices)
            available_devices: SimpleQueue[str] = SimpleQueue()
            for device in devices:
                available_devices.put(device)

            def run_on_available_device(task: _Task) -> None:
                device = available_devices.get()
                try:
                    self._run_device_task(task, device)
                finally:
                    available_devices.put(device)

            self._run_concurrently(
                tasks,
                min(self._limit(concurrency), len(tasks)),
                run_on_available_device,
            )
            return

        self._run_concurrently(
            tasks,
            min(self._limit(concurrency), len(tasks)),
            runner,
        )

    def _limit(self, available: int) -> int:
        if self.config.max_concurrent is None:
            return available
        return min(available, self.config.max_concurrent)

    @staticmethod
    def _run_concurrently(
        items: list[Any], concurrency: int, runner: Callable[[Any], None]
    ) -> None:
        errors: list[Exception] = []
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [pool.submit(runner, item) for item in items]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]

    def _worker_command(self, task: _Task) -> list[str]:
        return [
            sys.executable,
            "-m",
            "samudra.search.worker",
            "task",
            str(self.search.config_path),
            str(self.search.state_path),
            str(task.rung),
            str(task.task),
            *(["--anchor"] if task.anchor else []),
        ]

    @staticmethod
    def _worker_environment() -> dict[str, str]:
        environment = os.environ.copy()
        environment["SAMUDRA_DISABLE_DISTRIBUTED"] = "1"
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        return environment

    def _run_device_task(self, task: _Task, device: str) -> None:
        environment = self._worker_environment()
        environment["CUDA_VISIBLE_DEVICES"] = device
        subprocess.run(self._worker_command(task), check=True, env=environment)

    def _run_slurm_task(self, task: _Task) -> None:
        subprocess.run(
            [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={_slurm_cpus_per_gpu()}",
                "--gpus-per-task=1",
                "--gpu-bind=single:1",
                *self._worker_command(task),
            ],
            check=True,
            env=self._worker_environment(),
        )

    def _run_in_process(self, task: _Task) -> None:
        """Keep CPU/notebook execution lightweight and backwards compatible."""
        try:
            with MultitonScope():
                self.search.train_task(task.rung, task.task, anchor=task.anchor)
        finally:
            gc.collect()
