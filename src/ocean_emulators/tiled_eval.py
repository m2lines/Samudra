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

Every store is written incrementally, appending along ``time`` once every
``num_model_steps_forward`` steps, exactly as `Stepper.inference` does for the
single-tile path. Holding a whole rollout would be hopeless: one canonical frame
of the full prognostic stack is 205 x 720 x 720 float32 = 425 MB, so a month at
hourly stride is ~178 GB.
"""

import dataclasses
import datetime
import logging
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from ocean_emulators.backend import init_eval_backend
from ocean_emulators.config import EvalConfig
from ocean_emulators.constants import BOUNDARY_VARS, PROGNOSTIC_VARS, TensorMap
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.tile_diagnostics import response_by_distance
from ocean_emulators.tiling import (
    TileBlender,
    build_group_layout,
    build_tile_catalog,
    validate_tile_group,
)
from ocean_emulators.utils.data import (
    SPATIAL_FEATURE_CHANNELS,
    Normalize,
    get_inference_steps,
    spherical_area_weights,
)
from ocean_emulators.utils.device import using_gpu
from ocean_emulators.utils.distributed import set_seed
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
    """One write-chunk's worth of frames, held only until the next flush.

    Every list is keyed by the same time axis, so `times` alone measures how many
    steps are pending.
    """

    canonical: list[np.ndarray] = dataclasses.field(default_factory=list)
    times: list[xr.DataArray] = dataclasses.field(default_factory=list)
    preblend: list[np.ndarray] = dataclasses.field(default_factory=list)
    preblend_full: list[dict[str, np.ndarray]] = dataclasses.field(default_factory=list)
    response_curves: list[np.ndarray] = dataclasses.field(default_factory=list)
    response_maps: list[np.ndarray] = dataclasses.field(default_factory=list)

    def __len__(self) -> int:
        return len(self.times)

    def clear(self) -> None:
        self.canonical.clear()
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

        # The catalog and its overlap gate need the cache's own grid arrays and
        # attrs. A loaded DataSource keeps only prognostic/boundary under
        # renamed lat/lon dims and drops XC/YC/rA, so validating against it
        # would compare nothing and pass vacuously. Reopen the raw stores.
        raw = self._open_raw_caches()
        catalog = build_tile_catalog(raw, names=[str(name) for name in self._raw_names])
        self.layout = build_group_layout(catalog)
        report = validate_tile_group(
            self.layout, raw, probe_times=(0,), probe_vars=("prognostic",)
        )
        if not report.is_clean:
            logger.warning(
                "Tile group reported non-finite probe times %s; if the rollout "
                "starts on one of these the state is poisoned from step 0.",
                report.nonfinite_times,
            )
        logger.info(
            "Tile group: %d tiles, canonical %s at origin %s, overlaps %s",
            self.layout.num_tiles,
            self.layout.canonical_shape,
            self.layout.canonical_origin,
            sorted(set(self.layout.overlaps.values())),
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
        self.num_in = int((self.hist + 1) * (self.N_prog + self.N_bound)) + (
            SPATIAL_FEATURE_CHANNELS if spatial else 0
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
        self.seam_pairs = _seam_pairs(self.layout)
        logger.info(
            "Blend %s with window=%r; seams: %s",
            "enabled" if cfg.tiling.blend else "DISABLED (hard-crop control)",
            cfg.tiling.window,
            [pair.name for pair in self.seam_pairs],
        )

        # ---------------- datasets ----------------
        self.datasets: list[InferenceDataset] = []
        for source in sources:
            sliced = source.slice(cfg.inference_time)
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

        self.output_dir = Path(cfg.experiment.output_dir)
        self.tile_shape = self.layout.tiles[0].shape
        self._created_stores: set[str] = set()
        self._frames_written = 0

    # ------------------------------------------------------------------
    # setup helpers
    # ------------------------------------------------------------------

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
                    f"(spatial_features={self.spatial_features})."
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

    def _predict(self, state: torch.Tensor, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One model call per tile. Returns ``(residuals, inputs)``, both stacked."""
        inputs, residuals = [], []
        for index, dataset in enumerate(self.datasets):
            model_input = dataset.merge_prognostic_and_boundary(
                prognostic=state[index], step=step
            ).to(self.device)
            prediction = self.model.predict_step(model_input)
            inputs.append(model_input)
            residuals.append(prediction - model_input[:, : self.num_out])
        return torch.cat(residuals, dim=0), torch.cat(inputs, dim=0)

    @torch.no_grad()
    def run(self) -> None:
        started = time.perf_counter()
        cfg = self.cfg

        state = [
            dataset.initial_prognostic.to(self.device) for dataset in self.datasets
        ]
        num_steps = min(self.num_steps, len(self.datasets[0]))
        steps_per_write = resolve_steps_per_write(cfg.num_model_steps_forward, num_steps)
        logger.info(
            "Rolling out %d steps over %d tiles, appending to zarr every %d steps",
            num_steps,
            len(state),
            steps_per_write,
        )
        self._check_stores_are_writable()

        buffers = _RolloutBuffers()
        perturb_centre = self._perturbation_centre()

        for step in range(num_steps):
            if step % 10 == 0:
                logger.info("Rollout step %d of %d", step, num_steps - 1)

            residuals, inputs = self._predict(state, step)

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
                blender=self.blender,
                tile_wet=self.tile_wet,
                num_out=self.num_out,
                blend=cfg.tiling.blend,
            )
            state = [next_state[index : index + 1] for index in range(len(state))]

            # Unnormalize per tile, then stitch. Normalize's wet mask is a single
            # tile's, so it cannot be applied to a canonical frame; and because
            # unnormalization is affine per channel while the blend is a weighted
            # mean with weights summing to one, the two commute. (The tile masks
            # were verified to agree exactly in every overlap, so a land cell is
            # never averaged against a live one.)
            unnormalized = self.normalize.unnormalize_tensor_prognostic(
                next_state.cpu(), fill_value=0.0
            )
            buffers.canonical.append(
                self.blender.to_canonical(unnormalized.unsqueeze(0))[0].numpy()
            )
            buffers.times.append(self.datasets[0].get_target_time(step, 1))

            if len(buffers) >= steps_per_write:
                self._flush(buffers, centre=perturb_centre)

        # The rollout length need not divide the write interval; the tail is a
        # short chunk rather than a lost one.
        self._flush(buffers, centre=perturb_centre)

        elapsed = str(datetime.timedelta(seconds=int(time.perf_counter() - started)))
        logger.info(
            "Tiled inference finished in %s; %d frames written",
            elapsed,
            self._frames_written,
        )

    # ------------------------------------------------------------------
    # flushing
    # ------------------------------------------------------------------

    def _flush(self, buffers: _RolloutBuffers, *, centre: tuple[int, int]) -> None:
        """Append the buffered frames to every store, then drop them."""
        if not buffers.times:
            return
        first = self._frames_written
        logger.info("Writing to zarr: frames %d-%d", first, first + len(buffers) - 1)
        self._write_canonical(buffers.canonical, buffers.times)
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

    def _write_canonical(
        self, frames: list[np.ndarray], times: list[xr.DataArray]
    ) -> None:
        stacked = np.stack(frames)  # already unnormalized
        coords = self._canonical_coords()
        dataset = xr.Dataset(
            {
                name: (["time", "lat", "lon"], stacked[:, index])
                for index, name in enumerate(self.tensor_map.prognostic_var_names)
            },
            coords={
                "time": xr.concat(times, dim="time"),
                "lat": coords["lat"],
                "lon": coords["lon"],
            },
        )
        dataset.attrs.update(
            model_path=str(self.cfg.ckpt_path),
            blend=str(self.cfg.tiling.blend),
            window=self.cfg.tiling.window,
            ramp_width=str(self.cfg.tiling.ramp_width),
        )
        self._to_zarr(dataset, "predictions.zarr")

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
        dataset.to_zarr(path, mode="w")
        self._created_stores.add(name)
        logger.info("Created %s", path)


def main() -> None:
    cfg = EvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    evaluator = TiledEval(cfg)
    try:
        evaluator.run()
    except Exception:
        logger.exception("Tiled evaluation failed")
        raise


if __name__ == "__main__":
    main()
