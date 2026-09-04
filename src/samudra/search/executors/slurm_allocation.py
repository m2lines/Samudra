# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run independent search trials inside an existing Slurm allocation."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING, cast

from samudra.search.executors.pool import PoolExecutor, Task

if TYPE_CHECKING:
    from samudra.search.config import SlurmAllocationExecutorConfig


def _positive_int_environment(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    match = re.fullmatch(r"\s*(\d+)(?:\([^)]*\))?\s*", value)
    if match is None:
        return None
    parsed = int(match.group(1))
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


def step_memory_arguments(*, gpus: int) -> list[str]:
    """Request the task's proportional share of node memory, when known.

    Without an explicit step memory request, Slurm may assign the entire batch
    allocation's memory to the first exclusive step and serialize otherwise
    independent GPU workers.
    """
    memory_per_node = _positive_int_environment("SLURM_MEM_PER_NODE")
    allocated_gpus = _positive_int_environment("SLURM_GPUS_ON_NODE")
    if memory_per_node is None or allocated_gpus is None:
        return []
    memory = max(1, memory_per_node * gpus // allocated_gpus)
    return [f"--mem={memory}M"]


def gpus_per_node() -> int:
    value = _positive_int_environment("SLURM_GPUS_ON_NODE")
    if value is None:
        raise RuntimeError("Slurm did not set SLURM_GPUS_ON_NODE")
    return value


def world_size_placement(world_size: int) -> tuple[int, int] | None:
    """Return ``(nodes, processes_per_node)`` for a uniform placement."""
    per_node = gpus_per_node()
    available_nodes = _positive_int_environment("SLURM_NNODES") or 1
    for nodes in range(1, available_nodes + 1):
        processes_per_node, remainder = divmod(world_size, nodes)
        if remainder == 0 and 0 < processes_per_node <= per_node:
            return nodes, processes_per_node
    return None


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

    @property
    def resource_capacity(self) -> int:
        return allocation_gpu_count()

    @property
    def placeable_world_sizes(self) -> set[int]:
        return {
            size
            for size in range(1, self.resource_capacity + 1)
            if world_size_placement(size) is not None
        }

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
        world_size = getattr(task, "world_size", 1)
        placement = world_size_placement(world_size)
        if placement is None:
            raise ValueError(
                f"world_size={world_size} cannot be placed uniformly across "
                "this Slurm allocation"
            )
        nodes, processes_per_node = placement
        if nodes == 1:
            self._run_single_node_task(task, world_size=world_size)
            return
        subprocess.run(
            [
                "srun",
                "--exclusive",
                "--exact",
                f"--nodes={nodes}",
                f"--ntasks={nodes}",
                "--ntasks-per-node=1",
                f"--cpus-per-task={cpus_per_gpu() * processes_per_node}",
                f"--gpus-per-task={processes_per_node}",
                *step_memory_arguments(gpus=processes_per_node),
                "--gpu-bind=none",
                sys.executable,
                "-m",
                "samudra.search.node_launcher",
                f"--nnodes={nodes}",
                f"--nproc-per-node={processes_per_node}",
                f"--master-port={self._master_port(task)}",
                *self._worker_arguments(task),
            ],
            check=True,
            env=self._worker_environment(distributed=True),
        )

    def _run_single_node_task(self, task: Task, *, world_size: int) -> None:
        distributed = world_size > 1
        command = (
            self._distributed_worker_command(task)
            if distributed
            else self._worker_command(task)
        )
        subprocess.run(
            [
                "srun",
                "--exclusive",
                "--exact",
                "--nodes=1",
                "--ntasks=1",
                f"--cpus-per-task={cpus_per_gpu() * world_size}",
                f"--gpus-per-task={world_size}",
                *step_memory_arguments(gpus=world_size),
                "--gpu-bind=none",
                *command,
            ],
            check=True,
            env=self._worker_environment(distributed=distributed),
        )

    def _master_port(self, task: Task) -> int:
        job_digits = "".join(
            character for character in self.job_id if character.isdigit()
        )
        job_id = int(job_digits or 0)
        identity = task.rung * 10_000 + task.task * 2 + int(task.anchor)
        return 15_000 + (job_id + identity) % 40_000
