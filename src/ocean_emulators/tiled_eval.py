"""Autoregressive inference over a group of overlapping tiles.

A sibling of `Eval`, not surgery on it: one shared model steps every tile of a
group at the same time, the predicted residuals are reconciled in their overlaps
by `tiling.TileBlender`, and the consensus is scattered back before the next
step. That reproduces STRATA's arrangement -- a model trained on independent
tiles, blended only at inference -- so it measures what blending buys before any
training code changes.

The rollout loop is written out here rather than reusing `Stepper.inference`
because that owns its own loop; the blend has to happen *between* steps, which
is the one operation this file adds.

Three products, each its own zarr so an optional diagnostic can never change the
schema of the main one:

* ``predictions.zarr``  -- the canonical stitched state, always written.
* ``preblend.zarr``     -- pre-blend tile-to-tile disagreement (opt-in).
* ``perturbation.zarr`` -- far-field perturbation response (opt-in).

Holding a whole rollout would be hopeless: one canonical frame of a whole face
is 205 x 4320 x 4320 float32 = 15 GB. So ``predictions.zarr`` is laid out for
every frame up front and each rank writes the cells its own tiles own, on a
background thread, every step -- nothing is gathered. The diagnostic stores
append along ``time`` every ``num_model_steps_forward`` steps.

Frames are read from a packed cache with the training `GroupFrameReader`
(float16, each chunk decoded once, one step ahead of the GPU); other stores
fall back to per-tile `InferenceDataset` reads.
"""

import dataclasses
import datetime
import logging
import math
import os
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import dask.array
import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
import zarr

from ocean_emulators.backend import init_eval_backend
from ocean_emulators.chunk_reader import GroupFrameReader
from ocean_emulators.config import EvalConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS, TensorMap
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.face_parallel import assign_tiles
from ocean_emulators.face_validation import ReadAhead
from ocean_emulators.tile_diagnostics import response_by_distance
from ocean_emulators.tiling import (
    DistributedTileBlender,
    TileBlender,
    build_group_layout,
    build_tile_catalog,
    ownership_boxes,
    tile_catalog_from_windows,
    validate_tile_group,
)
from ocean_emulators.utils.data import (
    SPATIAL_FEATURE_CHANNELS,
    Normalize,
    get_inference_steps,
    spherical_area_weights,
)
from ocean_emulators.utils.device import using_gpu
from ocean_emulators.utils.distributed import (
    get_rank,
    get_world_size,
    init_distributed_mode,
    set_seed,
)
from ocean_emulators.utils.logging import (
    get_model_summary,
    handle_logging,
    handle_warnings,
)

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _SeamPair:
    """One shared band between two tiles, in each tile's local coordinates."""

    name: str
    left_tile: int
    right_tile: int
    axis: str  # "i" for a vertical seam, "j" for a horizontal one
    left_slice: tuple[slice, slice]
    right_slice: tuple[slice, slice]
    width: int


@dataclasses.dataclass
class _RolloutBuffers:
    """One write-chunk's worth of diagnostics, held only until the next flush.

    Every list is keyed by the same time axis, so `times` alone measures how many
    steps are pending.
    """

    times: list[xr.DataArray] = dataclasses.field(default_factory=list)
    preblend: list[np.ndarray] = dataclasses.field(default_factory=list)
    preblend_full: list[dict[str, np.ndarray]] = dataclasses.field(default_factory=list)
    response_curves: list[np.ndarray] = dataclasses.field(default_factory=list)
    response_maps: list[np.ndarray] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)

    def clear(self) -> None:
        self.times.clear()
        self.preblend.clear()
        self.preblend_full.clear()
        self.response_curves.clear()
        self.response_maps.clear()


def resolve_steps_per_write(num_model_steps_forward: int, total_steps: int) -> int:
    """How many steps to buffer before appending to zarr.

    Matches `get_rollout_step_chunks`: a non-positive value (the -1 sentinel)
    means one chunk covering the whole rollout, which for a tiled run is only
    ever viable for a short window.
    """
    if num_model_steps_forward <= 0:
        return max(total_steps, 1)
    return min(num_model_steps_forward, max(total_steps, 1))


def _seam_pairs(layout) -> list[_SeamPair]:
    """Enumerate the adjacent tile pairs and the band each shares."""
    pairs: list[_SeamPair] = []
    tiles = list(layout.tiles)
    for a_index, tile in enumerate(tiles):
        for other in tiles[a_index + 1 :]:
            j0 = max(tile.j_start, other.j_start)
            j1 = min(tile.j_end, other.j_end)
            i0 = max(tile.i_start, other.i_start)
            i1 = min(tile.i_end, other.i_end)
            if j1 <= j0 or i1 <= i0:
                continue
            height, width = j1 - j0, i1 - i0
            # A four-tile corner is shared by the diagonal pair too, but it is
            # already covered by the two edge pairs; skip the small square.
            if height < tile.shape[0] and width < tile.shape[1]:
                continue
            axis = "i" if width < height else "j"
            pairs.append(
                _SeamPair(
                    name=f"t{tile.tile_id}-t{other.tile_id}-{axis}",
                    left_tile=a_index,
                    right_tile=tiles.index(other),
                    axis=axis,
                    left_slice=(
                        slice(j0 - tile.j_start, j1 - tile.j_start),
                        slice(i0 - tile.i_start, i1 - tile.i_start),
                    ),
                    right_slice=(
                        slice(j0 - other.j_start, j1 - other.j_start),
                        slice(i0 - other.i_start, i1 - other.i_start),
                    ),
                    width=width if axis == "i" else height,
                )
            )
    return pairs


def preblend_disagreement(
    residuals: torch.Tensor,
    seam_pairs: list[_SeamPair],
    *,
    include_bands: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """RMS ``|delta_A - delta_B|`` across each seam, before any blending.

    This is the only honest measure of how much two tiles disagree: after the
    blend they agree by construction, so a post-blend number is identically zero
    and says nothing.

    The summary reduces along each seam and keeps the across-seam offset, so it
    stacks uniformly over seams of either orientation. The optional raw bands do
    not -- a vertical seam's band is transposed relative to a horizontal one --
    so they come back keyed by seam name for the caller to group.
    """
    summaries: list[np.ndarray] = []
    bands: dict[str, np.ndarray] = {}
    for pair in seam_pairs:
        left = residuals[pair.left_tile][(slice(None), *pair.left_slice)]
        right = residuals[pair.right_tile][(slice(None), *pair.right_slice)]
        difference = left - right
        reduce_dim = 1 if pair.axis == "i" else 2
        summaries.append(difference.pow(2).mean(dim=reduce_dim).sqrt().cpu().numpy())
        if include_bands:
            bands[pair.name] = difference.cpu().numpy()
    return np.stack(summaries), bands


def apply_perturbation(
    inputs: torch.Tensor,
    *,
    num_out: int,
    centre: tuple[int, int],
    box: int,
    amplitude: float,
) -> torch.Tensor:
    """Copy ``inputs`` with a box of the prognostic channels shifted."""
    half = box // 2
    j0, j1 = max(0, centre[0] - half), centre[0] + half
    i0, i1 = max(0, centre[1] - half), centre[1] + half
    perturbed = inputs.clone()
    perturbed[:, :num_out, j0:j1, i0:i1] += amplitude
    return perturbed


def advance_state(
    inputs: torch.Tensor,
    residuals: torch.Tensor,
    *,
    blender: TileBlender,
    tile_wet: list[torch.Tensor],
    num_out: int,
    blend: bool,
) -> torch.Tensor:
    """One autoregressive step: reconcile overlaps, then remask per tile.

    Remasking uses each tile's *own* wet mask rather than a shared one, because
    the tiles genuinely differ -- the live group's fourth tile has real land the
    other three do not, and masking it with a neighbour's mask would zero live
    cells before they become the next step's context.
    """
    reconciled = blender.blend(residuals) if blend else residuals
    state = inputs[:, :num_out] + reconciled
    for index, wet in enumerate(tile_wet):
        state[index] = torch.where(wet, state[index], 0.0)
    return state


def owned_slices(
    layout, tiles: tuple[int, ...], *, origin: tuple[int, int] = (0, 0)
) -> list[tuple[tuple[slice, slice], tuple[slice, slice]]]:
    """``(tile-local, canvas)`` slices of the cells each of ``tiles`` owns, on a
    canvas whose corner sits at canonical ``origin``."""
    boxes = ownership_boxes(layout)
    pieces = []
    for index in tiles:
        j0, j1, i0, i1 = boxes[index]
        cj, _, ci, _ = layout.canonical_bounds(layout.tiles[index])
        cj, ci = cj - origin[0], ci - origin[1]
        pieces.append(
            (
                (slice(j0, j1), slice(i0, i1)),
                (slice(cj + j0, cj + j1), slice(ci + i0, ci + i1)),
            )
        )
    return pieces


def rank_region(layout, tiles: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Canonical ``(j0, j1, i0, i1)`` covered by the cells ``tiles`` own.

    The ownership boxes tile the canvas exactly, so a contiguous block of tiles
    -- which `assign_tiles` hands every rank -- owns a rectangle.
    """
    pieces = [canonical for _, canonical in owned_slices(layout, tiles)]
    j0 = min(rows.start for rows, _ in pieces)
    j1 = max(rows.stop for rows, _ in pieces)
    i0 = min(cols.start for _, cols in pieces)
    i1 = max(cols.stop for _, cols in pieces)
    area = sum((rows.stop - rows.start) * (cols.stop - cols.start) for rows, cols in pieces)
    if area != (j1 - j0) * (i1 - i0):
        raise ValueError(f"Tiles {tiles} do not own a rectangle; ranks cannot write it as one region.")
    return j0, j1, i0, i1


def output_chunk(regions, *, cap: int = 1080) -> int:
    """Spatial chunk edge of ``predictions.zarr``.

    Ranks write concurrently, so no chunk may straddle two ranks' regions: the
    edge must divide every region boundary. The largest such edge up to ``cap``
    keeps chunks at a few MB.
    """
    edge = math.gcd(*(bound for region in regions for bound in region))
    return max(size for size in range(1, min(edge, cap) + 1) if edge % size == 0)


def create_prediction_store(
    path: Path,
    *,
    names: list[str],
    times: xr.DataArray,
    lat: np.ndarray,
    lon: np.ndarray,
    chunk: int,
    attrs: dict[str, str],
) -> None:
    """Lay out every frame of the rollout up front; ranks fill their regions.

    Unwritten chunks read back as NaN, the fill value, so a cut-short run is
    padded rather than truncated; its progress markers say how far it got.
    """
    lazy = dask.array.full(
        (times.size, lat.size, lon.size),
        np.nan,
        dtype=np.float32,
        chunks=(1, chunk, chunk),
    )
    dataset = xr.Dataset(
        {name: (["time", "lat", "lon"], lazy) for name in names},
        coords={"time": times, "lat": lat, "lon": lon},
        attrs=attrs,
    )
    # compute=False writes metadata and coords but no data chunk.
    dataset.to_zarr(
        path, mode="w-", compute=False, encoding={"time": {"dtype": "float64"}}
    )
    logger.info("Created %s (%d frames, %d-cell chunks)", path, times.size, chunk)


def _progress_path(path: Path, rank: int) -> Path:
    return path.with_suffix(".progress") / f"rank{rank}"


def frames_on_disk(path: Path) -> int:
    """Frames every writer rank finished, from their progress markers."""
    ranks = zarr.open_group(str(path), mode="r").attrs.get("writer_ranks")
    if ranks is None:
        raise ValueError(
            f"{path} was written by the old rank-0 appending writer and cannot "
            "be resumed by this one; start a fresh experiment."
        )
    done = []
    for rank in range(int(ranks)):
        marker = _progress_path(path, rank)
        done.append(int(marker.read_text()) if marker.exists() else 0)
    return min(done)


class PredictionWriter:
    """This rank's region of ``predictions.zarr``, written behind the rollout.

    A frame goes GPU -> pinned host -> zarr on a background thread, so the GPU
    keeps stepping while the last frame compresses and lands. Two host buffers:
    a third frame waits for the first to land, which bounds memory when the
    store is slower than the model.
    """

    def __init__(
        self,
        path: Path,
        *,
        names: list[str],
        region: tuple[int, int, int, int],
        rank: int,
        threads: int = 8,
    ) -> None:
        group = zarr.open_group(str(path), mode="r+")
        self.arrays = [group[name] for name in names]
        self.region = region
        j0, j1, i0, i1 = region
        rows, cols = self.arrays[0].chunks[1:]
        if j0 % rows or j1 % rows or i0 % cols or i1 % cols:
            raise ValueError(
                f"Region {region} straddles {path}'s {rows}x{cols} chunks, so two "
                "ranks would race on a chunk; resume with the original world size."
            )
        self.progress = _progress_path(path, rank)
        self.progress.parent.mkdir(exist_ok=True)
        pin = torch.cuda.is_available()
        self._host = [
            torch.empty((len(names), j1 - j0, i1 - i0), pin_memory=pin)
            for _ in range(2)
        ]
        self._pending: list[Future | None] = [None, None]
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pred_write")
        self._pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="pred_chunk")
        self.wait_seconds = 0.0

    def submit(self, canvas: torch.Tensor, frame: int) -> None:
        """Queue ``canvas`` ``[C, H, W]`` (any device) as time index ``frame``."""
        slot = frame % 2
        if self._pending[slot] is not None:
            waited = time.perf_counter()
            self._pending[slot].result()
            self.wait_seconds += time.perf_counter() - waited
        self._host[slot].copy_(canvas, non_blocking=True)
        ready = None
        if canvas.is_cuda:
            ready = torch.cuda.Event()
            ready.record()
        self._pending[slot] = self._writer.submit(self._write, slot, frame, ready)

    def _write(self, slot: int, frame: int, ready) -> None:
        if ready is not None:
            ready.synchronize()
        host = self._host[slot].numpy()
        j0, j1, i0, i1 = self.region

        def put(index: int) -> None:
            self.arrays[index][frame, j0:j1, i0:i1] = host[index]

        list(self._pool.map(put, range(len(self.arrays))))
        # One writer thread lands frames in order, so this count only grows.
        self.progress.write_text(str(frame + 1))

    def close(self) -> None:
        for future in self._pending:
            if future is not None:
                future.result()
        self._writer.shutdown()
        self._pool.shutdown()


class TiledEval:
    def __init__(self, cfg: EvalConfig) -> None:
        cfg.prepare_output_dirs()
        self.cfg = cfg
        self.device = init_eval_backend(cfg.backend)
        set_seed(cfg.experiment.rand_seed)

        if not using_gpu():
            cfg.data.num_workers = 0

        self.prognostic_var_names = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
        self.boundary_var_names = BOUNDARY_VARS[cfg.experiment.boundary_vars_key]
        self.N_prog = len(self.prognostic_var_names)
        self.N_bound = len(self.boundary_var_names)
        self.hist = cfg.data.hist
        self.tensor_map = TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        # ---------------- data ----------------
        logger.info("Loading tile caches")
        self.data_container = cfg.data.build(
            cfg.experiment.resolved_data_root,
            self.prognostic_var_names,
            self.boundary_var_names,
        )
        sources = self.data_container.replay_sources or [self.data_container.source]
        if len(sources) < 2:
            logger.warning(
                "Only %d cache resolved; a tiled rollout with one tile is a "
                "plain rollout with no blending to do.",
                len(sources),
            )
        self.sources = sources

        windows = self.data_container.replay_windows
        if windows:
            # Tiles cut out of ONE store (`data.llc_tiles`), which is how a
            # whole face is addressed. Their geometry is the configured window;
            # reopening the store per tile would hand every one of them the
            # store's full extent and quietly collapse the catalog to 36
            # identical tiles.
            catalog = tile_catalog_from_windows(windows)
            self.layout = build_group_layout(catalog)
        else:
            # The catalog and its overlap gate need the cache's own grid arrays
            # and attrs. A loaded DataSource keeps only prognostic/boundary
            # under renamed lat/lon dims and drops XC/YC/rA, so validating
            # against it would compare nothing and pass vacuously. Reopen the
            # raw stores.
            raw = self._open_raw_caches()
            catalog = build_tile_catalog(
                raw, names=[str(name) for name in self._raw_names]
            )
            self.layout = build_group_layout(catalog)
            report = validate_tile_group(
                self.layout, raw, probe_times=(0,), probe_vars=("prognostic",)
            )
            if not report.is_clean:
                logger.warning(
                    "Tile group reported non-finite probe times %s; if the "
                    "rollout starts on one of these the state is poisoned "
                    "from step 0.",
                    report.nonfinite_times,
                )
        logger.info(
            "Tile group: %d tiles, canonical %s at origin %s, overlaps %s",
            self.layout.num_tiles,
            self.layout.canonical_shape,
            self.layout.canonical_origin,
            sorted(set(self.layout.overlaps.values())),
        )

        # Under torchrun each rank steps a contiguous block of the tiles, trades
        # only the seams with the others, and writes the cells its tiles own.
        # One rank owns every tile.
        self.world_size, self.rank = get_world_size(), get_rank()
        self.tile_ranks = (0,) * self.layout.num_tiles
        if self.world_size > 1:
            self.tile_ranks = assign_tiles(
                self.layout.num_tiles, self.world_size, self.rank
            ).tile_ranks
        self.local_tiles = tuple(
            tile for tile, owner in enumerate(self.tile_ranks) if owner == self.rank
        )
        if self.world_size > 1 and (
            cfg.tiling.preblend_mode != "none" or cfg.tiling.perturbation
        ):
            raise ValueError(
                "Multi-GPU tiled eval does not support the preblend or "
                "perturbation diagnostics; set --tiling.preblend_mode none "
                "--tiling.perturbation false, or run on one GPU."
            )

        self.normalize = Normalize.init_instance(
            sources[0],
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
        )

        # ---------------- model ----------------
        spatial = all(source.spatial_features is not None for source in sources)
        if cfg.spatial_features is not None:
            spatial = cfg.spatial_features
        self.spatial_features = spatial
        # The valid-mask channel, like the spatial ones, is baked into a
        # checkpoint's input count, so eval has to reproduce whatever training
        # used -- `data.valid_mask`, which defaults to True. Tiled eval used to
        # ignore the field entirely and always build without the channel, so a
        # checkpoint trained with it failed to load with an off-by-one num_in.
        self.valid_mask = bool(cfg.data.valid_mask)
        self.num_in = (
            int((self.hist + 1) * (self.N_prog + self.N_bound))
            + (SPATIAL_FEATURE_CHANNELS if spatial else 0)
            + (1 if self.valid_mask else 0)
        )
        self.num_out = int((self.hist + 1) * self.N_prog)
        logger.info("Number of inputs: %d, outputs: %d", self.num_in, self.num_out)

        # Per-tile masks differ (tile 3 of the live group has real land), so the
        # model gets their union and each tile is remasked with its own after
        # the step. Masking with one tile's mask would zero live cells in another.
        self.tile_wet = [
            source.masks.prognostic_with_hist(self.hist).to(self.device)
            for source in sources
        ]
        union_wet = self.tile_wet[0].clone()
        for wet in self.tile_wet[1:]:
            union_wet |= wet

        self.area_weights = spherical_area_weights(sources[0].data).to(self.device)
        self.model = cfg.model.build(
            in_channels=self.num_in,
            out_channels=self.num_out,
            hist=self.hist,
            wet=union_wet,
            area_weights=self.area_weights,
            static_data=self.data_container.static_data,
            lat=torch.from_numpy(sources[0].data.lat.to_numpy()),
            lon=torch.from_numpy(sources[0].data.lon.to_numpy()),
        ).to(self.device)
        get_model_summary(self.model, None, cfg.debug)

        if cfg.ckpt_path is None:
            raise ValueError("ckpt_path must be set; try --ckpt_path=path/to/checkpoint")
        self._load_checkpoint(cfg.ckpt_path)
        self.model.eval()

        # ---------------- blender ----------------
        self.blender = TileBlender(
            self.layout,
            window=cfg.tiling.window,
            kbd_beta=cfg.tiling.kbd_beta,
            ramp_width=cfg.tiling.ramp_width,
            dtype=torch.float32,
        ).to(self.device)
        self.step_blender = self.blender
        if self.world_size > 1:
            self.step_blender = DistributedTileBlender(
                self.layout,
                tile_ranks=self.tile_ranks,
                rank=self.rank,
                window=cfg.tiling.window,
                kbd_beta=cfg.tiling.kbd_beta,
                ramp_width=cfg.tiling.ramp_width,
                dtype=torch.float32,
            ).to(self.device)
        self.local_wet = [self.tile_wet[index] for index in self.local_tiles]
        self.seam_pairs = _seam_pairs(self.layout)

        # The output canvas. Blended tiles agree in their overlaps, so the
        # stitched face is just every cell taken from the tile that owns it, and
        # each rank can write its own share. Land is NaN by each tile's OWN mask
        # (`Normalize`'s is only sources[0]'s); the masks agree in overlaps.
        self.region = rank_region(self.layout, self.local_tiles)
        self.owned = owned_slices(self.layout, self.local_tiles, origin=self.region[::2])
        self.output_chunk = output_chunk(
            [
                rank_region(
                    self.layout,
                    tuple(t for t, r in enumerate(self.tile_ranks) if r == owner),
                )
                for owner in range(self.world_size)
            ]
        )
        logger.info(
            "Rank %d writes canonical region j[%d:%d) i[%d:%d)", self.rank, *self.region
        )
        logger.info(
            "Blend %s with window=%r; seams: %s",
            "enabled" if cfg.tiling.blend else "DISABLED (hard-crop control)",
            cfg.tiling.window,
            [pair.name for pair in self.seam_pairs],
        )

        # ---------------- datasets ----------------
        self.datasets: list[InferenceDataset] = []
        for index in self.local_tiles:
            sliced = sources[index].slice(cfg.inference_time)
            self.datasets.append(
                InferenceDataset(
                    src=sliced,
                    prognostic_var_names=self.prognostic_var_names,
                    boundary_var_names=self.boundary_var_names,
                    hist=self.hist,
                    normalize_before_mask=cfg.data.normalize_before_mask,
                    masked_fill_value=cfg.data.masked_fill_value,
                    long_rollout=True,
                    inference_stride=cfg.inference_stride,
                    append_spatial_features_to_inputs=spatial,
                    append_valid_mask=self.valid_mask,
                )
            )
        self.num_steps = get_inference_steps(
            sources[0].slice(cfg.inference_time),
            hist=self.hist,
            inference_stride=cfg.inference_stride,
        )
        lengths = {len(dataset) for dataset in self.datasets}
        if len(lengths) != 1:
            raise ValueError(f"Tiles disagree on rollout length: {sorted(lengths)}")

        self.frame_reader = self._open_frame_reader()

        self.output_dir = Path(cfg.experiment.output_dir)
        self.tile_shape = self.layout.tiles[0].shape
        self._created_stores: set[str] = set()
        self._frames_written = 0

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

    def _open_frame_reader(self) -> GroupFrameReader | None:
        """Training's chunk-streaming reader over this rank's tiles, if it applies.

        It reads the packed cache's float16 chunks once each, rather than every
        tile reopening a window that straddles nine chunks through xarray and
        upcasting on the host. Anything else reads per tile.
        """
        windows = self.data_container.replay_windows
        locations = self.data_container.replay_locations or [None]
        path = getattr(locations[0], "path", None)
        packed = False
        if windows and path is not None and self.cfg.data.boundary_data_location is None:
            try:
                group = zarr.open_group(str(path), mode="r")
                packed = all(group[name].ndim == 4 for name in ("prognostic", "boundary"))
            except (KeyError, ValueError, zarr.errors.GroupNotFoundError):
                packed = False
        if not packed or self.hist != 0:
            logger.info("Reading frames per tile through InferenceDataset")
            return None
        return GroupFrameReader(
            str(path),
            windows=[windows[index] for index in self.local_tiles],
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            threads=self.cfg.tiling.read_threads,
        )

    def _pinned(self, frames: list[torch.Tensor]) -> list[torch.Tensor]:
        """Pin a read on its reader thread, so the H2D copy can be async."""
        if self.device.type != "cuda":
            return frames
        return [frame.pin_memory() for frame in frames]

    def _read_boundary(self, step: int) -> list[torch.Tensor]:
        """This rank's forcing for ``step``, per tile, on the host; thread-safe."""
        if self.frame_reader is None:
            return [dataset._get_boundary(step) for dataset in self.datasets]
        input_time = self.datasets[0].window_times(step)[0]
        return self._pinned(self.frame_reader.read_boundary(input_time))

    def _initial_state(self) -> list[torch.Tensor]:
        if self.frame_reader is None:
            return [dataset.initial_prognostic.to(self.device) for dataset in self.datasets]
        input_time = self.datasets[0].window_times(0)[0]
        frames = self.frame_reader.read_prognostic(input_time)
        return [
            dataset.prepare_frame(frame.to(self.device))
            for dataset, frame in zip(self.datasets, frames, strict=True)
        ]

    def _open_raw_caches(self) -> list[xr.Dataset]:
        """Open the packed stores behind each replay source, in the same order."""
        locations = self.data_container.replay_locations
        if not locations:
            raise ValueError(
                "The data container did not record replay cache locations, so "
                "the raw stores cannot be reopened for tile validation."
            )
        self._raw_names = [getattr(loc, "path", loc) for loc in locations]
        return [location.open() for location in locations]

    def _load_checkpoint(self, ckpt_path: str) -> None:
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        state_dict = OrderedDict(
            (key.removeprefix("module."), value)
            for key, value in checkpoint["model"].items()
        )
        for name, expected in self.model.state_dict().items():
            saved = state_dict.get(name)
            if saved is not None and tuple(saved.shape) != tuple(expected.shape):
                raise ValueError(
                    f"Checkpoint parameter '{name}' has shape {tuple(saved.shape)} "
                    f"but this model was built with {tuple(expected.shape)}. Tiled "
                    f"eval is using num_in={self.num_in} "
                    f"(spatial_features={self.spatial_features}, "
                    f"valid_mask={self.valid_mask}). Both are baked into a "
                    "checkpoint's input count; set --data.valid_mask / "
                    "--spatial_features to match how it was trained."
                )
        self.model.load_state_dict(state_dict)
        logger.info("Loaded checkpoint %s", ckpt_path)

    def _perturbation_centre(self) -> tuple[int, int]:
        configured = self.cfg.tiling.perturbation_centre
        if configured is not None:
            return (int(configured[0]), int(configured[1]))
        height, width = self.tile_shape
        return (height // 2, width // 2)

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------

    def _predict(
        self, state: list[torch.Tensor], step: int, boundaries: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One model call per tile. Returns ``(residuals, inputs)``, both stacked."""
        inputs, residuals = [], []
        for index, dataset in enumerate(self.datasets):
            boundary = boundaries[index].to(self.device, non_blocking=True)
            if self.frame_reader is not None:
                boundary = dataset.prepare_frame(boundary, boundary=True)
            model_input = dataset.merge_prognostic_and_boundary(
                prognostic=state[index], step=step, boundary=boundary
            ).to(self.device)
            prediction = self.model.predict_step(model_input)
            inputs.append(model_input)
            residuals.append(prediction - model_input[:, : self.num_out])
        return torch.cat(residuals, dim=0), torch.cat(inputs, dim=0)

    def _owned_canvas(self, tiles: torch.Tensor, canvas: torch.Tensor) -> torch.Tensor:
        """Fill ``canvas`` with this rank's region: unnormalized, land NaN, on GPU."""
        for (local, owned), tile, wet in zip(self.owned, tiles, self.local_wet, strict=True):
            canvas[(slice(None), *owned)] = self.normalize.unnormalize_tensor_prognostic(
                tile[(slice(None), *local)],
                fill_value=float("nan"),
                wet=wet[(slice(None), *local)],
            )
        return canvas

    def _resume_state(self) -> tuple[list[torch.Tensor], int]:
        """This rank's tiles cut from the last saved frame, and the next step.

        The saved frame is the stitched state, and blended tiles agree in their
        overlaps, so cutting it back up recovers the state the rollout held --
        to float32 round-off through unnormalize/normalize.
        """
        if self.hist != 0:
            raise ValueError("resume needs hist=0; one saved frame is not a history")
        path = self.output_dir / "predictions.zarr"
        start = frames_on_disk(path)
        if start == 0:
            raise ValueError(f"{path} holds no complete frame; rerun without resume")
        saved = xr.open_zarr(path, consolidated=False)
        frame = saved[list(self.tensor_map.prognostic_var_names)].isel(time=start - 1)
        state = []
        for index in self.local_tiles:
            j0, j1, i0, i1 = self.layout.canonical_bounds(self.layout.tiles[index])
            raw = torch.from_numpy(
                frame.isel(lat=slice(j0, j1), lon=slice(i0, i1))
                .to_array()
                .values
            )
            tile_state = self.normalize.normalize_tensor_prognostic(raw)
            wet = self.tile_wet[index].cpu()
            state.append(torch.where(wet, tile_state, 0.0).unsqueeze(0).to(self.device))
        self._frames_written = start
        self._created_stores.add("predictions.zarr")
        logger.info("Resuming from frame %d of %s", start - 1, path)
        return state, start

    @torch.no_grad()
    def run(self) -> None:
        started = time.perf_counter()
        cfg = self.cfg

        start = 0
        if cfg.tiling.resume:
            state, start = self._resume_state()
        else:
            state = self._initial_state()
        # The store holds the whole window; max_steps only stops early.
        total_steps = min(self.num_steps, len(self.datasets[0]))
        num_steps = min(total_steps, cfg.tiling.max_steps or total_steps)
        steps_per_write = resolve_steps_per_write(cfg.num_model_steps_forward, num_steps)
        logger.info(
            "Rolling out %d steps over %d tiles on this rank, writing every step",
            num_steps,
            len(state),
        )
        predictions = self.output_dir / "predictions.zarr"
        if not cfg.tiling.resume:
            self._check_stores_are_writable()
            if self.world_size > 1:
                # Every rank checks before rank 0 creates the store.
                dist.barrier()
            if self.rank == 0:
                coords = self._canonical_coords()
                create_prediction_store(
                    predictions,
                    names=list(self.tensor_map.prognostic_var_names),
                    times=self.datasets[0].get_target_time(0, total_steps),
                    lat=coords["lat"],
                    lon=coords["lon"],
                    chunk=self.output_chunk,
                    attrs={
                        "model_path": str(cfg.ckpt_path),
                        "blend": str(cfg.tiling.blend),
                        "window": cfg.tiling.window,
                        "ramp_width": str(cfg.tiling.ramp_width),
                        "writer_ranks": str(self.world_size),
                    },
                )
            if self.world_size > 1:
                dist.barrier()
        writer = PredictionWriter(
            predictions,
            names=list(self.tensor_map.prognostic_var_names),
            region=self.region,
            rank=self.rank,
            threads=cfg.tiling.write_threads,
        )
        j0, j1, i0, i1 = self.region
        canvas = torch.empty((self.num_out, j1 - j0, i1 - i0), device=self.device)

        buffers = _RolloutBuffers()
        perturb_centre = self._perturbation_centre()
        diagnostics = cfg.tiling.preblend_mode != "none" or cfg.tiling.perturbation
        reads = ReadAhead(
            [lambda step=step: self._read_boundary(step) for step in range(start, num_steps)],
            depth=2,
        )
        frames = iter(reads)
        read_wait = 0.0
        window_start, window_first = time.perf_counter(), start
        loop_start = None

        for step in range(start, num_steps):
            if step == start + 1:
                loop_start = time.perf_counter()
            if step > start and step % 10 == 0:
                now = time.perf_counter()
                logger.info(
                    "Rollout step %d of %d: %.2f s/step (read wait %.1f s, write "
                    "wait %.1f s since step %d)",
                    step,
                    num_steps - 1,
                    (now - window_start) / (step - window_first),
                    read_wait,
                    writer.wait_seconds,
                    window_first,
                )
                window_start, window_first = now, step
                read_wait, writer.wait_seconds = 0.0, 0.0

            waited = time.perf_counter()
            boundaries = next(frames)
            read_wait += time.perf_counter() - waited
            residuals, inputs = self._predict(state, step, boundaries)

            if cfg.tiling.preblend_mode != "none":
                summary, bands = preblend_disagreement(
                    residuals,
                    self.seam_pairs,
                    include_bands=cfg.tiling.preblend_mode == "full",
                )
                buffers.preblend.append(summary)
                if cfg.tiling.preblend_mode == "full":
                    buffers.preblend_full.append(bands)

            if cfg.tiling.perturbation:
                curve, response_map = self._perturbation_response(
                    inputs, residuals, centre=perturb_centre
                )
                buffers.response_curves.append(curve)
                buffers.response_maps.append(response_map)

            next_state = advance_state(
                inputs,
                residuals,
                blender=self.step_blender,
                tile_wet=self.local_wet,
                num_out=self.num_out,
                blend=cfg.tiling.blend,
            )
            state = [next_state[index : index + 1] for index in range(len(state))]

            # Blended tiles already agree in their overlaps. Unblended (the
            # hard-crop control) ones do not, and the saved field has always
            # been their windowed mean, so blend a copy for output only.
            output = next_state if cfg.tiling.blend else self.step_blender.blend(next_state)
            writer.submit(self._owned_canvas(output, canvas), step)

            if diagnostics:
                buffers.times.append(self.datasets[0].get_target_time(step, 1))
                if len(buffers) >= steps_per_write:
                    self._flush(buffers, centre=perturb_centre)

        frames.close()
        # The rollout length need not divide the write interval; the tail is a
        # short chunk rather than a lost one.
        self._flush(buffers, centre=perturb_centre)
        writer.close()
        if self.world_size > 1:
            # NCCL calls only enqueue, so a rank that finishes first reaches
            # here with its last step still in flight; leaving would strand the rest.
            dist.barrier()

        finished = time.perf_counter()
        elapsed = str(datetime.timedelta(seconds=int(finished - started)))
        logger.info(
            "Tiled inference finished in %s; %d frames written",
            elapsed,
            num_steps - start,
        )
        if loop_start is not None and num_steps - start > 1:
            # Excludes the first step, which pays for cuDNN autotuning and warm-up.
            logger.info(
                "Steady state: %.2f s/step over %d steps",
                (finished - loop_start) / (num_steps - start - 1),
                num_steps - start - 1,
            )

    # ------------------------------------------------------------------
    # flushing
    # ------------------------------------------------------------------

    def _flush(self, buffers: _RolloutBuffers, *, centre: tuple[int, int]) -> None:
        """Append the buffered frames to every store, then drop them."""
        if not buffers.times:
            return
        first = self._frames_written
        logger.info("Writing diagnostics: frames %d-%d", first, first + len(buffers) - 1)
        if buffers.preblend:
            self._write_preblend(buffers.preblend, buffers.preblend_full, buffers.times)
        if buffers.response_curves:
            self._write_perturbation(
                buffers.response_curves,
                buffers.response_maps,
                buffers.times,
                centre=centre,
            )
        self._frames_written += len(buffers)
        buffers.clear()

    def _store_names(self) -> list[str]:
        """The stores this configuration will write, in the order they appear."""
        names = ["predictions.zarr"]
        if self.cfg.tiling.preblend_mode != "none":
            names.append("preblend.zarr")
        if self.cfg.tiling.perturbation:
            names.append("perturbation.zarr")
        return names

    def _check_stores_are_writable(self) -> None:
        """Fail before the rollout rather than after the first chunk of it."""
        for name in self._store_names():
            path = self.output_dir / name
            if path.exists():
                raise FileExistsError(
                    f"{path} already exists. Choose a unique experiment name or "
                    "delete it first."
                )

    # ------------------------------------------------------------------
    # diagnostics
    # ------------------------------------------------------------------

    def _perturbation_response(
        self,
        inputs: torch.Tensor,
        control_residuals: torch.Tensor,
        *,
        centre: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Re-predict from a state perturbed far from the seam; return the response.

        Both branches start from the *same* control state each step rather than
        running two diverging rollouts, so this measures the model's one-step
        coupling rather than accumulated trajectory drift.
        """
        cfg = self.cfg.tiling
        perturbed_input = apply_perturbation(
            inputs,
            num_out=self.num_out,
            centre=centre,
            box=cfg.perturbation_box,
            amplitude=cfg.perturbation_amplitude,
        )
        perturbed = self.model.predict_step(perturbed_input)
        response = (
            perturbed - perturbed_input[:, : self.num_out]
        ) - control_residuals

        magnitude = response.abs().cpu().numpy()
        _, curve = response_by_distance(
            magnitude, centre=centre, num_bins=cfg.response_bins
        )
        return curve, magnitude[:, cfg.perturbation_channel]

    # ------------------------------------------------------------------
    # writers
    # ------------------------------------------------------------------

    def _canonical_coords(self) -> dict[str, np.ndarray]:
        origin_j, origin_i = self.layout.canonical_origin
        height, width = self.layout.canonical_shape
        return {
            "lat": np.arange(origin_j, origin_j + height),
            "lon": np.arange(origin_i, origin_i + width),
        }

    def _write_preblend(
        self,
        summaries: list[np.ndarray],
        full: list[dict[str, np.ndarray]],
        times: list[xr.DataArray],
    ) -> None:
        stacked = np.stack(summaries)  # [time, seam, channel, offset]
        dataset = xr.Dataset(
            {"disagreement": (["time", "seam", "channel", "offset"], stacked)},
            coords={
                "time": xr.concat(times, dim="time"),
                "seam": [pair.name for pair in self.seam_pairs],
                "channel": list(self.tensor_map.prognostic_var_names),
                "offset": np.arange(stacked.shape[-1]),
            },
        )
        dataset.attrs["note"] = (
            "Pre-blend RMS |delta_A - delta_B| across each seam, reduced along "
            "the seam. Values are normalized units. Post-blend disagreement is "
            "zero by construction and is not recorded."
        )
        if full:
            # Vertical and horizontal bands are transposed relative to each
            # other, so they cannot share one variable; group by orientation.
            for axis in ("i", "j"):
                names = [pair.name for pair in self.seam_pairs if pair.axis == axis]
                if not names:
                    continue
                band = np.stack(
                    [np.stack([frame[name] for name in names]) for frame in full]
                )
                dataset[f"band_difference_{axis}"] = (
                    ["time", f"seam_{axis}", "channel", f"band_j_{axis}", f"band_i_{axis}"],
                    band,
                )
                dataset = dataset.assign_coords({f"seam_{axis}": names})
        self._to_zarr(dataset, "preblend.zarr")

    def _write_perturbation(
        self,
        curves: list[np.ndarray],
        maps: list[np.ndarray],
        times: list[xr.DataArray],
        *,
        centre: tuple[int, int],
    ) -> None:
        stacked = np.stack(curves)  # [time, tile, channel, bin]
        bin_centres, _ = response_by_distance(
            np.zeros((1, *self.tile_shape)),
            centre=centre,
            num_bins=self.cfg.tiling.response_bins,
        )
        dataset = xr.Dataset(
            {
                "response_by_distance": (
                    ["time", "tile", "channel", "distance"],
                    stacked,
                ),
                "response_map": (["time", "tile", "lat", "lon"], np.stack(maps)),
            },
            coords={
                "time": xr.concat(times, dim="time"),
                "tile": [tile.tile_id for tile in self.layout.tiles],
                "channel": list(self.tensor_map.prognostic_var_names),
                "distance": bin_centres,
                "lat": np.arange(self.tile_shape[0]),
                "lon": np.arange(self.tile_shape[1]),
            },
        )
        dataset.attrs.update(
            note=(
                "One-step response to a perturbation in a box far from the seam. "
                "Receptive-field coupling decays with distance; GroupNorm couples "
                "every cell to the whole tile and appears as a flat floor."
            ),
            perturbation_centre=str(centre),
            perturbation_box=str(self.cfg.tiling.perturbation_box),
            perturbation_amplitude=str(self.cfg.tiling.perturbation_amplitude),
            response_map_channel=self.tensor_map.prognostic_var_names[
                self.cfg.tiling.perturbation_channel
            ],
        )
        self._to_zarr(dataset, "perturbation.zarr")

    def _to_zarr(self, dataset: xr.Dataset, name: str) -> None:
        """Create the store on the first flush, append along time on the rest.

        The chunk shape is taken from the first flush, so the on-disk time chunk
        is the write interval; a short tail chunk partially fills the last one.
        """
        path = self.output_dir / name
        if name in self._created_stores:
            dataset.to_zarr(path, mode="a", append_dim="time")
            return
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Choose a unique experiment name or "
                "delete it first."
            )
        # Float time: from one frame xarray infers integer "days since <t0>",
        # which the next hourly append cannot be cast back into.
        dataset.to_zarr(path, mode="w", encoding={"time": {"dtype": "float64"}})
        self._created_stores.add(name)
        logger.info("Created %s", path)


def main() -> None:
    cfg = EvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        init_distributed_mode()

    evaluator = TiledEval(cfg)
    try:
        evaluator.run()
    except Exception:
        logger.exception("Tiled evaluation failed")
        raise


if __name__ == "__main__":
    main()
