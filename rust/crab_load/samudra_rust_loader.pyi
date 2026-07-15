# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from os import PathLike

from numpy import float32
from numpy.typing import NDArray

class FlatOm4ReadPool:
    def __init__(self, max_concurrent_reads: int) -> None: ...
    @property
    def max_concurrent_reads(self) -> int: ...

class FlatOm4Reader:
    def __init__(
        self,
        path: str | PathLike[str],
        variables: list[str],
        read_pool: FlatOm4ReadPool,
    ) -> None: ...
    @property
    def shape(self) -> tuple[int, int, int]: ...
    def read(self, indexes: list[int], variables: list[str]) -> NDArray[float32]: ...
    def read_into(
        self,
        indexes: list[int],
        variables: list[str],
        target: NDArray[float32],
    ) -> None: ...
