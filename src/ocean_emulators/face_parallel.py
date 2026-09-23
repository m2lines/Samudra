"""Advancing one face as a single replay row, split across the ranks.

Replay already groups tiles: a row holds `[T, C, H, W]`, the tiles share one
cursor, and their overlaps are reconciled before the state is written back.
What it assumes is that one rank holds the whole group -- true for a 2x2
cluster, impossible for a 36-tile face, since one H200 fits about four 752^2
tiles with their activations.

This module is the piece that removes that assumption, and nothing else. A row
still addresses a whole face; each rank simply stores, advances and scores the
slice of it that it owns. Three things follow, and they are the whole content
here:

* **Who owns what.** Contiguous blocks, chosen for store-chunk locality -- see
  `chunk_reader`. Every rank owns the same number of tiles, so every rank runs
  the same number of forward/backward chunks, which is what keeps DDP from
  deadlocking on a rank that finishes early.
* **Agreement.** The ranks must stay on the same row at the same lead step. The
  planner is already deterministic given a seed, so sharing the seed is enough
  for the common path; the exception is divergence, which is data-dependent and
  would otherwise let one rank reseed a row the others keep advancing.
* **Normalization.** A per-rank loss denominator would give a rank holding the
  face's two all-land tiles the same say as one holding open ocean, weighting
  its few wet cells up to a hundredfold. A face-wide denominator removes that
  exactly and costs one collective at startup.

`DomainParallelContext` is the sibling to compare against: it shards one
sample's *pixels* across ranks and needs the model rewritten to match. This
shards *tiles*, so the model, the loss and the blender are untouched -- which
is also what lets one trained model serve a single tile later.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence

import pydantic
import torch

from ocean_emulators.utils.loss import GradientZNorms

from ocean_emulators.tiling import (
    DistributedTileBlender,
    ReplayGroup,
    TileGroupLayout,
    WindowKind,
    contiguous_tile_blocks,
    ownership_masks,
)

logger = logging.getLogger(__name__)


class FaceParallelConfig(pydantic.BaseModel):
    """Knobs for advancing a whole face per replay row."""

    model_config = pydantic.ConfigDict(extra="forbid")

    enabled: bool = False
    read_threads: int = pydantic.Field(
        default=8,
        ge=1,
        description=(
            "Chunk decodes in flight per rank while reading a face frame. "
            "Sized against the STORE, not the core count: 4 ranks x 8 threads "
            "read a face in 2.91 s where 4 x 15 took 7.85 s on the same work "
            "(scripts/_bench/face_read.py). Aim for about 32 across all ranks."
        ),
    )
    tiles_per_chunk: int = pydantic.Field(
        default=3,
        ge=1,
        description=(
            "Tiles a rank puts through one forward/backward. Bounded by "
            "activation memory: 4 tiles of 752^2 peaked at ~110 GB of an "
            "H200's 180 GB, so 3 leaves room. Gradients accumulate across a "
            "rank's chunks and DDP syncs on the last one."
        ),
    )


@dataclasses.dataclass(frozen=True)
class TileAssignment:
    """Which rank owns which tile, and where a tile sits in its owner's list."""

    tile_ranks: tuple[int, ...]
    local_tiles: tuple[int, ...]
    rank: int

    @property
    def tiles_per_rank(self) -> int:
        return len(self.local_tiles)

    def position(self, tile: int) -> int:
        """Index of a global tile within this rank's own ordering."""
        return self.local_tiles.index(tile)

    def owns(self, tile: int) -> bool:
        return self.tile_ranks[tile] == self.rank


def assign_tiles(num_tiles: int, world_size: int, rank: int) -> TileAssignment:
    """Contiguous, equal-sized blocks of the tile grid, one per rank.

    Equal-sized is a hard requirement rather than a preference: a rank with
    fewer tiles would run fewer forward/backward chunks and the others would
    block forever in the DDP all-reduce it never reaches. The error names the
    world sizes that do work rather than silently padding.
    """
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside a world of {world_size}")
    if num_tiles % world_size:
        usable = [size for size in range(1, num_tiles + 1) if num_tiles % size == 0]
        raise ValueError(
            f"{num_tiles} tiles do not divide over {world_size} ranks; every "
            "rank must hold the same number or DDP will deadlock on the rank "
            f"that runs out first. World sizes that divide it: {usable}."
        )
    blocks = contiguous_tile_blocks(num_tiles, world_size)
    tile_ranks = [0] * num_tiles
    for owner, block in enumerate(blocks):
        for tile in block:
            tile_ranks[tile] = owner
    return TileAssignment(
        tile_ranks=tuple(tile_ranks),
        local_tiles=tuple(sorted(blocks[rank])),
        rank=rank,
    )


def split_into_chunks(count: int, tiles_per_chunk: int) -> list[tuple[int, ...]]:
    """Local tile positions, grouped into forward/backward chunks.

    Chunks are as even as possible rather than greedy. Greedy packing of 9
    tiles at 4 per chunk gives 4/4/1, whose last chunk wastes most of the GPU
    and whose peak is set by the 4; 3/3/3 has the same chunk count and a lower
    peak.
    """
    if count < 1:
        raise ValueError(f"a rank must own at least one tile, got {count}")
    if tiles_per_chunk < 1:
        raise ValueError(f"tiles_per_chunk must be >= 1, got {tiles_per_chunk}")
    num_chunks = -(-count // tiles_per_chunk)
    chunks: list[tuple[int, ...]] = []
    start = 0
    for index in range(num_chunks):
        remaining = num_chunks - index
        # Round up, so any short chunk lands at the END. A short first chunk
        # would set the same peak memory while leaving the GPU idle last.
        size = -(-(count - start) // remaining)
        chunks.append(tuple(range(start, start + size)))
        start += size
    return chunks


class FaceParallelContext:
    """The rank's view of a face-sized replay group."""

    def __init__(
        self,
        layout: TileGroupLayout,
        *,
        world_size: int,
        rank: int,
        tiles_per_chunk: int = 3,
        window: WindowKind = "quintic",
        ramp_width: int | None = None,
        dtype: torch.dtype = torch.float32,
        process_group: object | None = None,
    ) -> None:
        self.layout = layout
        self.world_size = world_size
        self.rank = rank
        self.process_group = process_group
        self.assignment = assign_tiles(layout.num_tiles, world_size, rank)
        self.local_tiles = self.assignment.local_tiles
        self.chunks = split_into_chunks(len(self.local_tiles), tiles_per_chunk)
        self.blender = DistributedTileBlender(
            layout,
            tile_ranks=self.assignment.tile_ranks,
            rank=rank,
            window=window,
            ramp_width=ramp_width,
            dtype=dtype,
            process_group=process_group,
        )
        logger.info(
            "Face-parallel rank %d/%d: tiles %s, %d chunk(s) of %s, %d halo "
            "message(s) per blend",
            rank,
            world_size,
            list(self.local_tiles),
            len(self.chunks),
            [len(chunk) for chunk in self.chunks],
            self.blender.exchanges,
        )

    @property
    def num_chunks(self) -> int:
        """Identical on every rank, which is what DDP requires."""
        return len(self.chunks)

    def local_ownership_masks(self) -> torch.Tensor:
        """``[Tlocal, 1, H, W]`` marking the cells this rank's tiles alone own.

        Tiles overlap, so scoring each in full would count the shared bands
        twice and pull every metric toward the seams.
        """
        return ownership_masks(self.layout)[list(self.local_tiles)]

    def global_wet_denominator(self, local_wet: torch.Tensor) -> torch.Tensor:
        """Per-channel wet-cell count over the WHOLE face, divided by world size.

        ``local_wet`` is ``[Tlocal, C, H, W]``, already multiplied by this
        rank's ownership masks so no cell is counted twice.

        The division by ``world_size`` is what makes DDP's mean over ranks come
        out as the face-wide weighted mean: with ``L_r = N_r / D`` on each rank,
        ``(1/R) sum_r L_r`` equals ``sum_r N_r / (R * D)``, so ``D`` has to be
        the face total over ``R``.

        One collective, at setup. After this the loss needs no communication
        and the answer no longer depends on which tiles landed where.

        NOT YET WIRED INTO THE LOSS. The base metric and `gradient_h` share
        `_weighted_channel_mean` and would take this directly, but
        `gradient_z_l1_loss` normalizes by valid *level pairs* rather than wet
        cells, so handing it a cell count would silently reweight it against
        the other terms -- worse than the distortion being fixed. It needs its
        own face-wide pair count, derived from the same masks. Until all three
        terms are converted, the trainer keeps the per-rank denominator; with
        contiguous blocks on face 1 that is a ~3.4x reweighting between the
        land-heavy and the open-ocean rank.
        """
        if local_wet.ndim != 4:
            raise ValueError(
                f"local_wet must be [Tlocal, C, H, W], got {tuple(local_wet.shape)}"
            )
        if local_wet.shape[0] != len(self.local_tiles):
            raise ValueError(
                f"local_wet holds {local_wet.shape[0]} tiles but rank "
                f"{self.rank} owns {len(self.local_tiles)}"
            )
        total = local_wet.to(dtype=torch.float64).sum(dim=(0, 2, 3))
        total = self._all_reduce(total, op="sum")
        if bool((total <= 0).any()):
            dry = int((total <= 0).sum())
            logger.warning(
                "%d prognostic channel(s) have no wet cell anywhere on the "
                "face; they will contribute nothing to the loss.",
                dry,
            )
        return (total / self.world_size).to(dtype=torch.float32)

    def global_loss_norms(
        self, wet: torch.Tensor, local_sample_weight: torch.Tensor | None, *, tiles: int
    ) -> tuple[torch.Tensor, "GradientZNorms"]:
        """Denominators that make a rank's score a share of the face's.

        Every term is normalized by a constant derived from the masks alone,
        so a rank's chunks sum to the rank's share and DDP's mean over ranks
        comes out as the face-wide weighted mean -- independent of which tiles
        landed where, which is what stops the two all-land tiles reweighting
        whoever holds them.

        The ``/ world_size`` lands on a different factor for the two terms.
        The base metric is one ratio, ``sum_r N_r / D``, so DDP's ``1/R``
        has to be absorbed into ``D``. `gradient_z` is an average OF ratios,
        ``(sum_p N_p / DEN_p) / COUNT``, so ``DEN`` must stay the true
        domain count and the ``1/R`` goes into ``COUNT`` instead.
        """
        from ocean_emulators.utils.loss import (
            gradient_z_norms,
            weighted_channel_denominator,
        )

        local = weighted_channel_denominator(
            wet=wet, batch=tiles, extra_weight=local_sample_weight
        )
        denominator = self._all_reduce(
            local.detach().to(dtype=torch.float64).clone(), op="sum"
        )
        norms = gradient_z_norms(
            wet=wet,
            batch=tiles,
            sample_weight=local_sample_weight,
            reduce=lambda cells: self._all_reduce(cells.clone(), op="sum"),
        )
        return (
            (denominator / self.world_size).to(dtype=torch.float32),
            GradientZNorms(
                valid_cells=norms.valid_cells,
                count_by_time=norms.count_by_time / self.world_size,
            ),
        )

    def agree(self, flag: bool) -> bool:
        """True if ANY rank raises the flag.

        Divergence is the one decision the ranks cannot each make for
        themselves. It is detected from the data, so one rank can see it while
        the others do not -- and if that rank reseeds a row the others keep
        advancing, the buffers silently part company and the tiles of a "face"
        stop being one timestamp.
        """
        return bool(self._all_reduce(torch.tensor([float(flag)]), op="max").item())

    def _all_reduce(self, tensor: torch.Tensor, *, op: str) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "Face-parallel training spans "
                f"{self.world_size} ranks but torch.distributed is not "
                "initialized."
            )
        reduce_op = {"sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op]
        dist.all_reduce(tensor, op=reduce_op, group=self.process_group)
        return tensor

    def __repr__(self) -> str:
        return (
            f"FaceParallelContext(rank={self.rank}/{self.world_size}, "
            f"tiles={len(self.local_tiles)}/{self.layout.num_tiles}, "
            f"chunks={[len(c) for c in self.chunks]})"
        )


def face_group_is_shardable(
    layout: TileGroupLayout, world_size: int
) -> tuple[bool, str]:
    """Whether this group can be split, and why not when it cannot."""
    # Geometry first: a shape or face problem is the more fundamental one, and
    # reporting "not a square grid" for a group of mismatched tiles would send
    # the reader after the wrong thing.
    shapes = {tile.shape for tile in layout.tiles}
    if len(shapes) != 1:
        return False, (
            f"tiles have shapes {sorted(shapes)}; the blender and a batched "
            "forward both need one shape."
        )
    faces = {tile.face for tile in layout.tiles}
    if len(faces) != 1:
        return False, (
            f"group spans faces {sorted(faces)}; cross-face halos need a "
            "rotation operator that does not exist yet."
        )
    try:
        assign_tiles(layout.num_tiles, world_size, 0)
    except ValueError as error:
        return False, str(error)
    return True, ""


def local_tile_windows(
    windows: Sequence[tuple[int, int, int, int, int]],
    assignment: TileAssignment,
) -> list[tuple[int, int, int, int, int]]:
    """The subset of a face's windows this rank reads, in its own order."""
    return [windows[tile] for tile in assignment.local_tiles]


def build_face_replay_groups(
    layout: TileGroupLayout,
    context: "FaceParallelContext",
    *,
    num_strides: int,
    dataset_index_of: Sequence[int] | None = None,
) -> list["ReplayGroup"]:
    """Replay groups whose tile list is this rank's share of the face.

    The trick that keeps the change small: a group's ``dataset_indices`` names
    only the tiles this rank owns, while its ``layout`` still describes all 36.
    Everything downstream that iterates a group -- `datasets_for`,
    `_entry_states`, `_seed_transitions_for_slot`, `_slot_input_span`,
    `_batch_wet_weight` -- keys off ``dataset_indices`` and so becomes
    rank-local for free, while the blender, which needs the whole face's
    geometry, reads the layout.

    A replay row therefore still *addresses* a whole face; each rank simply
    stores and advances its slice of it.
    """
    if num_strides < 1:
        raise ValueError(f"num_strides must be >= 1, got {num_strides}")
    if num_strides != 1:
        raise ValueError(
            "Face-parallel replay advances one face on one cursor, so it takes "
            f"a single temporal stride; got {num_strides}. Use data_stride=[1]."
        )
    if dataset_index_of is None:
        dataset_index_of = list(range(layout.num_tiles))
    if len(dataset_index_of) != layout.num_tiles:
        raise ValueError(
            f"dataset_index_of has {len(dataset_index_of)} entries for "
            f"{layout.num_tiles} tiles"
        )
    local = tuple(int(dataset_index_of[tile]) for tile in context.local_tiles)
    return [
        ReplayGroup(
            group_id=0,
            dataset_indices=local,
            layout=layout,
            blender=context.blender,
        )
    ]
