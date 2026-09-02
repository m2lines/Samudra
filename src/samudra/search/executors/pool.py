# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Shared orchestration for executors that pool independent workers."""

from __future__ import annotations

import gc
import os
import sys
from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol

from samudra.search.executors.base import Executor
from samudra.utils.multiton import MultitonScope


class PoolConfig(Protocol):
    max_concurrent: int | None
    dry_run: bool


@dataclass(frozen=True)
class Task:
    rung: int
    task: int
    anchor: bool


class PoolExecutor(Executor):
    """Run independent trials using slots supplied by a concrete executor."""

    @property
    @abstractmethod
    def config(self) -> PoolConfig: ...

    @property
    @abstractmethod
    def job_id(self) -> str: ...

    def submit_initial(self, state: dict[str, Any]) -> None:
        """Co-schedule fixed anchors and rung zero across available slots."""
        tasks = [
            Task(len(self.search.rungs) - 1, task, True)
            for task, _ in enumerate(state["anchors"]["candidates"])
        ]
        tasks.extend(
            Task(0, task, False)
            for task, _ in enumerate(state["rungs"][0]["candidates"])
        )
        state["anchors"]["job_id"] = self.job_id
        state["rungs"][0]["job_id"] = self.job_id
        state["status"] = "running"
        self.search.write_state(state)
        if self.config.dry_run:
            return
        self._run_tasks(tasks)
        self.search.advance(0)

    def submit_anchors(self, state: dict[str, Any]) -> None:
        candidates = state["anchors"]["candidates"]
        state["anchors"]["job_id"] = self.job_id
        self.search.write_state(state)
        if self.config.dry_run:
            return
        self._run_tasks(
            [
                Task(len(self.search.rungs) - 1, task, True)
                for task, _ in enumerate(candidates)
            ]
        )

    def submit_rung(self, state: dict[str, Any], rung: int) -> None:
        candidates = state["rungs"][rung]["candidates"]
        state["rungs"][rung]["job_id"] = self.job_id
        state["status"] = "running"
        self.search.write_state(state)
        if self.config.dry_run:
            return
        self._run_tasks([Task(rung, task, False) for task, _ in enumerate(candidates)])
        self.search.advance(rung)

    @abstractmethod
    def _run_tasks(self, tasks: list[Task]) -> None: ...

    def _limit(self, available: int) -> int:
        if self.config.max_concurrent is None:
            return available
        return min(available, self.config.max_concurrent)

    @staticmethod
    def _run_concurrently(
        items: list[Task], concurrency: int, runner: Callable[[Task], None]
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

    def _worker_command(self, task: Task) -> list[str]:
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

    def _run_in_process(self, task: Task) -> None:
        """Run a CPU task with isolated process-global training state."""
        try:
            with MultitonScope():
                self.search.train_task(task.rung, task.task, anchor=task.anchor)
        finally:
            gc.collect()
