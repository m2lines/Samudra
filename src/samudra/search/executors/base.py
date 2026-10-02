# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compute executor interface used by search algorithms."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from samudra.search.successive_halving import SuccessiveHalving


class Executor(ABC):
    """Run search tasks and notify the algorithm when a rung is complete."""

    def __init__(self, search: "SuccessiveHalving") -> None:
        self.search = search

    def submit_initial(self, state: dict[str, Any]) -> None:
        """Submit fixed anchors and the first successive-halving rung."""
        self.submit_anchors(state)
        self.submit_rung(self.search.read_state(), 0)

    @abstractmethod
    def submit_anchors(self, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def submit_rung(self, state: dict[str, Any], rung: int) -> None: ...
