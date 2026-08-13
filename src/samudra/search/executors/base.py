# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compute executor interface used by search algorithms."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from samudra.search.config import SearchConfig


class SearchController(Protocol):
    """Algorithm surface needed by an executor."""

    config: SearchConfig
    config_path: Path
    state_path: Path
    search_dir: Path
    slug: str
    run_id: str
    rungs: list[int]

    def write_state(self, state: dict[str, Any]) -> None: ...
    def read_state(self) -> dict[str, Any]: ...
    def train_task(self, rung: int, task: int, *, anchor: bool) -> None: ...
    def advance(self, rung: int) -> None: ...


class Executor(ABC):
    """Run search tasks and notify the algorithm when a rung is complete."""

    def __init__(self, search: SearchController) -> None:
        self.search = search

    @abstractmethod
    def submit_anchors(self, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def submit_rung(self, state: dict[str, Any], rung: int) -> None: ...
