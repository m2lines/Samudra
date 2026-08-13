# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Synchronous local search execution."""

from typing import Any

from samudra.search.executors.base import Executor


class LocalExecutor(Executor):
    def submit_anchors(self, state: dict[str, Any]) -> None:
        candidates = state["anchors"]["candidates"]
        state["anchors"]["job_id"] = "local"
        self.search.write_state(state)
        if self.search.config.executor.dry_run:
            return
        for task, _ in enumerate(candidates):
            self.search.train_task(len(self.search.rungs) - 1, task, anchor=True)

    def submit_rung(self, state: dict[str, Any], rung: int) -> None:
        candidates = state["rungs"][rung]["candidates"]
        state["rungs"][rung]["job_id"] = "local"
        state["status"] = "running"
        self.search.write_state(state)
        if self.search.config.executor.dry_run:
            return
        for task, _ in enumerate(candidates):
            self.search.train_task(rung, task, anchor=False)
        self.search.advance(rung)
