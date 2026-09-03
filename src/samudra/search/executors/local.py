# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run independent search trials on one local machine."""

from __future__ import annotations

import os
import subprocess
import threading
from typing import TYPE_CHECKING, cast

import torch

from samudra.search.executors.pool import PoolExecutor, Task

if TYPE_CHECKING:
    from samudra.search.config import LocalExecutorConfig


def visible_devices() -> list[str]:
    """Return CUDA identifiers that can be assigned to child processes."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        if configured.strip() in {"", "-1"}:
            return []
        return [value.strip() for value in configured.split(",") if value.strip()]
    return [str(index) for index in range(torch.cuda.device_count())]


class LocalExecutor(PoolExecutor):
    """Use each GPU visible to the current machine as a worker slot."""

    @property
    def config(self) -> LocalExecutorConfig:
        config = self.search.config.executor
        if config.type != "local":
            raise TypeError("LocalExecutor requires a LocalExecutorConfig")
        return cast("LocalExecutorConfig", config)

    @property
    def job_id(self) -> str:
        return "local"

    @property
    def resource_capacity(self) -> int:
        return max(1, len(visible_devices()))

    def _run_tasks(self, tasks: list[Task]) -> None:
        if not tasks:
            return
        devices = visible_devices()
        if not devices:
            for task in tasks:
                self._run_in_process(task)
            return

        pool = _DevicePool(devices)

        def run_on_available_device(task: Task) -> None:
            world_size = getattr(task, "world_size", 1)
            assigned = pool.acquire(world_size)
            try:
                environment = self._worker_environment(distributed=world_size > 1)
                environment["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
                command = (
                    self._distributed_worker_command(task)
                    if world_size > 1
                    else self._worker_command(task)
                )
                subprocess.run(command, check=True, env=environment)
            finally:
                pool.release(assigned)

        self._run_concurrently(
            tasks,
            min(self._limit(len(devices)), len(tasks)),
            run_on_available_device,
        )


class _DevicePool:
    """Atomically reserve groups of CUDA identifiers for concurrent trials."""

    def __init__(self, devices: list[str]) -> None:
        self._available = list(devices)
        self._condition = threading.Condition()

    def acquire(self, count: int) -> list[str]:
        with self._condition:
            self._condition.wait_for(lambda: len(self._available) >= count)
            assigned = self._available[:count]
            del self._available[:count]
            return assigned

    def release(self, devices: list[str]) -> None:
        with self._condition:
            self._available.extend(devices)
            self._condition.notify_all()
