# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Synchronous local search execution."""

import gc

import torch

from samudra.search.executors.base import Executor
from samudra.search.state import SearchState, SearchStatus
from samudra.utils.multiton import MultitonScope


class LocalExecutor(Executor):
    def submit_anchors(self, state: SearchState) -> None:
        candidates = state.anchors.candidates
        state.anchors.job_id = "local"
        self.search.write_state(state)
        if self.search.config.executor.dry_run:
            return
        for task, _ in enumerate(candidates):
            self._run_task(len(self.search.rungs) - 1, task, anchor=True)

    def submit_rung(self, state: SearchState, rung: int) -> None:
        candidates = state.rungs[rung].candidates
        state.rungs[rung].job_id = "local"
        state.transition(SearchStatus.RUNNING)
        self.search.write_state(state)
        if self.search.config.executor.dry_run:
            return
        for task, _ in enumerate(candidates):
            self._run_task(rung, task, anchor=False)
        self.search.advance(rung)

    def _run_task(self, rung: int, task: int, *, anchor: bool) -> None:
        """Run one candidate with isolated process-global training state."""
        try:
            with MultitonScope():
                self.search.train_task(rung, task, anchor=anchor)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
