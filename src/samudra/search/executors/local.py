# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run independent search trials on one local machine."""

from __future__ import annotations

import os
import subprocess
from queue import SimpleQueue
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

    def _run_tasks(self, tasks: list[Task]) -> None:
        if not tasks:
            return
        devices = visible_devices()
        if not devices:
            for task in tasks:
                self._run_in_process(task)
            return

        available_devices: SimpleQueue[str] = SimpleQueue()
        for device in devices:
            available_devices.put(device)

        def run_on_available_device(task: Task) -> None:
            device = available_devices.get()
            try:
                environment = self._worker_environment()
                environment["CUDA_VISIBLE_DEVICES"] = device
                subprocess.run(self._worker_command(task), check=True, env=environment)
            finally:
                available_devices.put(device)

        self._run_concurrently(
            tasks,
            min(self._limit(len(devices)), len(tasks)),
            run_on_available_device,
        )
