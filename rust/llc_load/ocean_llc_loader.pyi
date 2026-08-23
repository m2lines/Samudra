"""Type stubs for the `ocean_llc_loader` extension (rust/llc_load/src/lib.rs)."""

from os import PathLike
from typing import Optional

from numpy import float32
from numpy.typing import NDArray

#: `(array name, level)`. `("Theta", 7)` is `Theta_7`; `("Eta", None)` is a
#: surface variable; `("boundary", 2)` is a packed cache's third channel.
ChannelSelector = tuple[str, Optional[int]]

class LlcReadPool:
    """Rayon pool shared by every reader in one process."""

    def __init__(self, max_concurrent_reads: int) -> None: ...

class LlcPatchReader:
    """Persistent reader for one tile of a raw LLC store or a packed cache."""

    def __init__(
        self,
        path: str | PathLike[str],
        channels: list[ChannelSelector],
        face: Optional[int],
        j_start: int,
        j_stop: int,
        i_start: int,
        i_stop: int,
        read_pool: LlcReadPool,
    ) -> None: ...
    @property
    def shape(self) -> tuple[int, int, int]:
        """`(store time length, tile height, tile width)`."""

    @property
    def full_row_reads(self) -> bool: ...
    def read_into(
        self,
        indexes: list[int],
        channel_indexes: list[int],
        target: NDArray[float32],
    ) -> None:
        """Fill a C-contiguous `[time, channel, j, i]` float32 array."""

def read_static(
    path: str | PathLike[str],
    name: str,
    face: Optional[int],
    j_start: int,
    j_stop: int,
    i_start: int,
    i_stop: int,
    level: Optional[int] = None,
) -> NDArray[float32]:
    """Read a static `[j, i]` grid field (`XC`, `YC`, `rA`, ...) for the tile."""
