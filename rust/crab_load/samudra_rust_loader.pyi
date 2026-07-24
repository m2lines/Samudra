# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from os import PathLike

from numpy import float32
from numpy.typing import NDArray

type CompactVariableSelector = tuple[str, int | None]

class FlatOm4ReadPool:
    def __init__(self, max_concurrent_reads: int) -> None: ...

class FlatOm4Reader:
    def __init__(
        self,
        path: str | PathLike[str],
        variables: list[str],
        read_pool: FlatOm4ReadPool,
    ) -> None: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    def read_into(
        self,
        indexes: list[int],
        variables: list[str],
        target: NDArray[float32],
    ) -> None: ...

class CompactOm4Reader:
    def __init__(
        self,
        path: str | PathLike[str],
        variable_selectors: list[CompactVariableSelector],
        read_pool: FlatOm4ReadPool,
    ) -> None: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    def read_into(
        self,
        indexes: list[int],
        variable_selectors: list[CompactVariableSelector],
        target: NDArray[float32],
    ) -> None: ...
