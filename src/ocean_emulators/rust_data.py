"""Optional native (Rust) reader for LLC4320 Zarr stores.

Ported from the OM4 Rust data loader in m2lines/Samudra#800, retargeted at the
raw LLC store rather than a preprocessed 3D patch cache. It is opt-in
(``data.loader_backend: rust``) and additive: nothing here runs unless that knob
is set, and the whole feature is `rust/llc_load/` plus this module plus a handful
of guarded call sites.

The split of responsibility is the PR's: Python owns every canonical semantic --
which channels exist and in what order, which timestamps a sample reads,
normalisation, masking, DDP schedules -- and Rust owns only persistent Zarr
handles and chunk reads. In particular :func:`channel_selectors` reproduces the
channel order that ``DataSource.filter`` plus ``_dataset_to_numpy`` produce on
the xarray path, so a Rust batch is element-for-element what the CPU loader
would have built.

Why this exists, in one paragraph. Caching 3D prognostics for the globe costs
~500 TB, which is not available. Reading them straight out of the raw store
costs nothing extra to store, and for a **chunk-aligned 720x720 tile** it reads
exactly one chunk per variable per timestamp -- no amplification at all. What
does not work is reading the 2D boundary fields the same way: they are chunked
one-globe-per-timestamp, so four of them cost ~1.5 GiB per sample to deliver
8 MiB. Hence :attr:`NativeStoreSpec.packed_prefix` and
``data.boundary_data_location``: prognostics from the raw store, boundaries from
a small packed cache. Measured on the production store, one 720x720 tile:

===========================================  ============
CPU loader, full packed 3D cache             0.9 s
Rust, raw-store prognostics + cached bounds  2.7 s
Rust, everything from the raw store          8.2 s
CPU loader, everything from the raw store    13.4 s
===========================================  ============
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

_LEVEL_SUFFIX = re.compile(r"^(?P<base>.+)_(?P<level>\d+)$")
#: Vertical axis names in a raw LLC store. `W` lives on `k_p1`.
_LEVEL_DIMS = frozenset({"k", "k_p1", "k_l", "k_u", "lev", "level"})


class RustLoaderUnavailable(RuntimeError):
    """The `ocean_llc_loader` extension is not importable."""


def _extension():
    try:
        import ocean_llc_loader
    except ImportError as error:  # pragma: no cover - depends on the build
        raise RustLoaderUnavailable(
            "data.loader_backend='rust' needs the `ocean_llc_loader` extension. "
            "Build it with `scripts/build_rust_loader.sh`, or set "
            "data.loader_backend='cpu' to use the xarray loader."
        ) from error
    return ocean_llc_loader


def is_available() -> bool:
    try:
        _extension()
    except RustLoaderUnavailable:
        return False
    return True


@lru_cache(maxsize=None)
def _array_dims(store: str, name: str) -> tuple[str, ...]:
    """Axis names of one array, from Zarr V3 metadata or xarray's V2 attribute."""
    root = Path(store) / name
    for meta_name, key in (
        ("zarr.json", "dimension_names"),
        (".zattrs", "_ARRAY_DIMENSIONS"),
    ):
        path = root / meta_name
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text())
        names = metadata.get(key)
        if names:
            return tuple(str(name) for name in names)
    raise KeyError(
        f"Could not read dimension names for {name!r} in {store}; expected "
        f"{root / 'zarr.json'} or {root / '.zattrs'}"
    )


def _has_level_axis(store: str, name: str) -> bool:
    return any(dim in _LEVEL_DIMS for dim in _array_dims(store, name))


@lru_cache(maxsize=None)
def _packed_channel_names(store: str, prefix: str) -> tuple[str, ...]:
    """Channel names of a packed cache, from its root `<prefix>_channel_names_json`."""
    for meta_name in ("zarr.json", ".zattrs"):
        path = Path(store) / meta_name
        if not path.is_file():
            continue
        attributes = json.loads(path.read_text())
        attributes = attributes.get("attributes", attributes)
        raw = attributes.get(f"{prefix}_channel_names_json")
        if raw is not None:
            return tuple(json.loads(raw))
    raise KeyError(
        f"{store} has no {prefix}_channel_names_json attribute; it does not look "
        "like a packed cache"
    )


def channel_selectors(
    store: str, var_names: list[str], packed_prefix: str | None = None
) -> list[tuple[str, int | None]]:
    """The `(array, level)` selectors behind a list of logical channel names.

    For a packed cache (`packed_prefix`, e.g. `"boundary"`) the channels are one
    array indexed along its `<prefix>_channel` axis, so a selector is
    `("boundary", position)` and the order is exactly ``var_names`` -- the same
    order ``DataSource.filter``'s packed branch produces.

    For a raw LLC store this mirrors ``DataSource.filter``'s compact branch: it
    collects base names and levels in first-seen order, then emits every level of
    every levelled variable. That ordering, not the order of ``var_names``, is
    what the xarray path feeds the model, and the two must agree channel for
    channel.
    """
    if packed_prefix is not None:
        available = _packed_channel_names(store, packed_prefix)
        position = {name: index for index, name in enumerate(available)}
        missing = [name for name in var_names if name not in position]
        if missing:
            raise KeyError(
                f"Packed {packed_prefix} cache {store} is missing requested "
                f"channels: {missing}"
            )
        return [(packed_prefix, position[name]) for name in var_names]

    bases: list[str] = []
    levels: list[int] = []
    for name in var_names:
        match = _LEVEL_SUFFIX.match(name)
        if match is None:
            if name not in bases:
                bases.append(name)
            continue
        base = match.group("base")
        level = int(match.group("level"))
        if base not in bases:
            bases.append(base)
        if level not in levels:
            levels.append(level)

    selectors: list[tuple[str, int | None]] = []
    for base in bases:
        if levels and _has_level_axis(store, base):
            selectors.extend((base, level) for level in levels)
        else:
            selectors.append((base, None))

    if len(selectors) != len(var_names):
        raise ValueError(
            f"Rust loader derived {len(selectors)} channels from {len(var_names)} "
            f"names for {store}; the xarray loader would build "
            f"{len(var_names)}. Channels: {selectors[:8]}..."
        )
    return selectors


@dataclasses.dataclass(frozen=True)
class NativeStoreSpec:
    """Everything Rust needs to serve one tile of one store.

    Carried on :class:`~ocean_emulators.utils.data.DataSource` so that
    ``filter``/``slice``/``map`` propagate it for free through
    ``dataclasses.replace``.
    """

    path: str
    face: int | None
    j_start: int
    j_stop: int
    i_start: int
    i_stop: int
    read_threads: int = 0
    #: Set for a packed cache whose channels live on one `<prefix>_channel` axis
    #: (the boundary-only cache). None for a raw LLC store.
    packed_prefix: str | None = None

    @property
    def grid_shape(self) -> tuple[int, int]:
        return (self.j_stop - self.j_start, self.i_stop - self.i_start)

    def is_chunk_aligned(self, chunk: int = 720) -> bool:
        """Does this window sit on the source's 720x720 chunk grid?

        Alignment is the difference between one chunk per variable and four, and
        it is worth ~1.7x on a 205-channel read. Only meaningful for a raw store.
        """
        if self.packed_prefix is not None:
            return True
        return all(
            value % chunk == 0
            for value in (self.j_start, self.j_stop, self.i_start, self.i_stop)
        )

    def threads(self) -> int:
        if self.read_threads > 0:
            return self.read_threads
        configured = int(os.environ.get("OCEAN_RUST_LOADER_THREADS", "0"))
        if configured > 0:
            return configured
        # One DataLoader worker per rank already runs several of these, so stay
        # modest by default rather than claiming every core.
        return max(1, min(8, len(os.sched_getaffinity(0))))

    def read_static(self, name: str, level: int | None = None) -> np.ndarray:
        """Read a static `[j, i]` grid field (`XC`, `YC`, `rA`, ...) for the tile."""
        return _extension().read_static(
            self.path,
            name,
            self.face,
            self.j_start,
            self.j_stop,
            self.i_start,
            self.i_stop,
            level,
        )

    def spatial_features(self) -> torch.Tensor | None:
        """`[sphere_x, sphere_y, sphere_z, log_rA]` for this tile.

        The packed caches carry these as stored arrays and
        ``_packed_spatial_features`` builds the tensor from them. The raw store
        has the same XC/YC/rA, so building it here keeps a raw-store run's input
        channel count equal to a cache run's -- which is what makes their step
        times comparable.
        """
        try:
            lon = np.deg2rad(self.read_static("XC"))
            lat = np.deg2rad(self.read_static("YC"))
            area = self.read_static("rA")
        except Exception as error:
            logger.warning(
                "Rust loader could not read XC/YC/rA from %s (%s); training "
                "without spatial feature channels.",
                self.path,
                error,
            )
            return None
        if not (
            np.isfinite(lon).all() and np.isfinite(lat).all() and np.isfinite(area).all()
        ):
            raise ValueError("LLC spatial fields XC, YC, and rA must be finite")
        if np.any(area <= 0):
            raise ValueError("LLC cell-area field rA must be strictly positive")
        # Same reference area as the packed path, so the channel means the same
        # thing in both: absolute scale, not per-patch normalised.
        features = np.stack(
            [
                np.cos(lat) * np.cos(lon),
                np.cos(lat) * np.sin(lon),
                np.sin(lat),
                np.log(area / 1_000_000.0),
            ],
            axis=0,
        )
        return torch.from_numpy(features.astype(np.float32, copy=False))


#: One Rayon pool per process, shared by every reader in it.
_READ_POOL: dict[tuple[int, int], object] = {}


def _read_pool(threads: int):
    pid = os.getpid()
    key = (pid, threads)
    pool = _READ_POOL.get(key)
    if pool is None:
        # Drop pools inherited across a fork: their Rayon threads did not come
        # with us, so they would never run anything.
        for stale in [stale for stale in _READ_POOL if stale[0] != pid]:
            del _READ_POOL[stale]
        pool = _extension().LlcReadPool(threads)
        _READ_POOL[key] = pool
    return pool


class NativeLlcReader:
    """Reads one or more named channel groups out of one store.

    Groups that share a store share a reader, so an array behind two groups --
    `Eta` is both a prognostic and a boundary channel of a raw LLC store -- is
    opened once and read once per timestamp. When prognostics and boundaries come
    from *different* stores (raw store + boundary-only cache) each gets its own
    reader, and that is the whole difference.

    The Rust handle is built lazily and rebuilt after a fork: a Rayon pool does
    not survive `fork`, and DataLoader workers are forked whenever the sources
    support it.
    """

    def __init__(self, spec: NativeStoreSpec, groups: dict[str, list[str]]) -> None:
        if not groups:
            raise ValueError("NativeLlcReader needs at least one channel group")
        self.spec = spec

        channels: list[tuple[str, int | None]] = []
        index_of: dict[tuple[str, int | None], int] = {}
        self.group_indices: dict[str, list[int]] = {}
        for name, var_names in groups.items():
            selectors = channel_selectors(spec.path, var_names, spec.packed_prefix)
            positions = []
            for selector in selectors:
                if selector not in index_of:
                    index_of[selector] = len(channels)
                    channels.append(selector)
                positions.append(index_of[selector])
            self.group_indices[name] = positions

        self.channels = channels
        self._reader = None
        self._reader_pid: int | None = None

        if not spec.is_chunk_aligned():
            logger.warning(
                "Rust loader tile j=[%d:%d) i=[%d:%d) is NOT aligned to the "
                "store's 720x720 chunk grid, so every 3D read spans 4 chunks "
                "instead of 1 (~1.7x slower). Align the tile to multiples of 720 "
                "if you can.",
                spec.j_start,
                spec.j_stop,
                spec.i_start,
                spec.i_stop,
            )

    def __getstate__(self) -> dict:
        # The native handle is process-local; `spawn` workers rebuild it.
        state = dict(self.__dict__)
        state["_reader"] = None
        state["_reader_pid"] = None
        return state

    @property
    def reader(self):
        pid = os.getpid()
        if self._reader is None or self._reader_pid != pid:
            spec = self.spec
            self._reader = _extension().LlcPatchReader(
                spec.path,
                self.channels,
                spec.face,
                spec.j_start,
                spec.j_stop,
                spec.i_start,
                spec.i_stop,
                _read_pool(spec.threads()),
            )
            self._reader_pid = pid
        return self._reader

    @property
    def time_len(self) -> int:
        return int(self.reader.shape[0])

    def read(self, group: str, indices) -> torch.Tensor:
        """`[time, channel, j, i]` float32 for one channel group."""
        channel_indices = self.group_indices[group]
        indices = [int(index) for index in np.asarray(indices).reshape(-1)]
        height, width = self.spec.grid_shape
        out = np.empty(
            (len(indices), len(channel_indices), height, width), dtype=np.float32
        )
        self.reader.read_into(indices, channel_indices, out)
        return torch.from_numpy(out)
