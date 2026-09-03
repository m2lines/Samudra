# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run independent search trials inside an existing Slurm allocation."""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING, cast

from samudra.search.executors.pool import PoolExecutor, Task

if TYPE_CHECKING:
    from samudra.search.config import SlurmAllocationExecutorConfig


def _positive_int_environment(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def allocation_gpu_count() -> int:
    """Return the GPU count for a homogeneous Slurm allocation."""
    total = _positive_int_environment("SLURM_GPUS")
    if total is not None:
        return total
    nodes = _positive_int_environment("SLURM_NNODES") or 1
    per_node = _positive_int_environment("SLURM_GPUS_ON_NODE")
    if per_node is None:
        raise RuntimeError(
            "Cannot determine allocation GPU capacity: Slurm did not set "
            "SLURM_GPUS or SLURM_GPUS_ON_NODE"
        )
    return nodes * per_node


def cpus_per_gpu() -> int:
    cpus = _positive_int_environment("SLURM_CPUS_ON_NODE")
    gpus = _positive_int_environment("SLURM_GPUS_ON_NODE")
    if cpus is None or gpus is None:
        return 1
    return max(1, cpus // gpus)


class SlurmAllocationExecutor(PoolExecutor):
    """Use exclusive one-GPU job steps within the current allocation."""

    @property
    def config(self) -> SlurmAllocationExecutorConfig:
        config = self.search.config.executor
        if config.type != "slurm_allocation":
            raise TypeError(
                "SlurmAllocationExecutor requires a SlurmAllocationExecutorConfig"
            )
        return cast("SlurmAllocationExecutorConfig", config)

    @property
    def job_id(self) -> str:
        return os.environ.get("SLURM_JOB_ID", "slurm-allocation")

    def _run_tasks(self, tasks: list[Task]) -> None:
        if not tasks:
            return
        if "SLURM_JOB_ID" not in os.environ:
            raise RuntimeError(
                "The slurm_allocation executor must run inside a Slurm allocation"
            )
        step = os.environ.get("SLURM_STEP_ID")
        if step not in {None, "", "batch", "extern"}:
            raise RuntimeError(
                "Cannot launch allocation workers from inside an existing Slurm "
                "job step. Run the search directly in an sbatch script or an "
                "salloc shell, rather than with `srun python`."
            )
        concurrency = min(self._limit(allocation_gpu_count()), len(tasks))
        self._run_concurrently(tasks, concurrency, self._run_slurm_task)

    def _run_slurm_task(self, task: Task) -> None:
        subprocess.run(
            [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={cpus_per_gpu()}",
                "--gpus-per-task=1",
                "--gpu-bind=single:1",
                *self._worker_command(task),
            ],
            check=True,
            env=self._worker_environment(),
        )
