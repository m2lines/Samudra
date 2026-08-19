# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Compute executor interface used by search algorithms."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from samudra.search.state import SearchState

if TYPE_CHECKING:
    from samudra.search.successive_halving import SuccessiveHalving


class Executor(ABC):
    """Run search tasks and notify the algorithm when a rung is complete."""

    def __init__(self, search: "SuccessiveHalving") -> None:
        self.search = search

    @abstractmethod
    def submit_anchors(self, state: SearchState) -> None: ...

    @abstractmethod
    def submit_rung(self, state: SearchState, rung: int) -> None: ...
