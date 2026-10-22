#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export one continuous, observation-initialized physical-state rollout.

The default covers 584 complete five-day means from 2015-01-01 through
2022-12-30 (exclusive). The final two days of 2022 are deliberately not a partial
model step. Only the initial 95 days use observed ocean fields; subsequent input
is ERA5 forcing. No target ocean observations or metrics are loaded/computed.

Use the validation-selected best.pt. Its EMA weights, rather than the raw
optimizer weights stored alongside them, generated its selection score.
"""

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
import zarr  # type: ignore[import-untyped]

import samudra
from samudra.constants import build_om4_layout
from samudra.models.surface_initialized import SurfaceInitializedModel
from samudra.observations.archive import ObservationArchive, context_planes
from samudra.observations.baseline import archive_identity, digest
from samudra.observations.constants import (
    HISTORY_DAYS,
    HISTORY_STEPS,
    LAST_HISTORY_STEP,
    MODEL_DEPTHS,
    STATE_CHANNELS,
    STEP_DAYS,
    SURFACE_CHANNELS,
)
from samudra.observations.samples import Samples
from samudra.observations.state import write_json
from samudra.utils.data import stack_levels
from samudra.utils.device import autocast
from samudra.utils.ema import EMATracker


def forecast_intervals(
    origin: str, *, end_exclusive: str | None = None, steps: int | None = None
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    start = pd.Timestamp(origin)
    if start != start.normalize() or start.tzinfo is not None:
        raise ValueError("Origin must be an unzoned calendar date at midnight")
    if (steps is None) == (end_exclusive is None):
        raise ValueError("Specify either steps or an exclusive end date")
    if end_exclusive is not None:
        end = pd.Timestamp(end_exclusive)
        if end != end.normalize() or end.tzinfo is not None:
            raise ValueError("End must be an unzoned calendar date at midnight")
        steps = (end - start).days // STEP_DAYS
    if steps is None or steps < 1:
        raise ValueError("A rollout needs at least one complete five-day interval")
    starts = pd.date_range(start, periods=steps, freq=f"{STEP_DAYS}D")
    return starts, starts + pd.Timedelta(days=STEP_DAYS)


def preflight_archive(
    daily: Path, origin: pd.Timestamp, final_end: pd.Timestamp
) -> dict[str, int]:
    """Check every required shard before spending compute; never require future SST/ADT."""
    first = origin - pd.Timedelta(days=HISTORY_DAYS)
    counts: dict[str, int] = {}
    for product in ("sst", "adt", "era5"):
        end = final_end if product == "era5" else origin
        count = 0
        for stamp in pd.date_range(first, end - pd.Timedelta(days=1), freq="D"):
            path = daily / product / str(stamp.year) / f"{stamp:%Y-%m-%d}.npz"
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required {product} input shard is missing: {path}"
                )
            count += 1
        counts[product] = count
    return counts


def analysis_coords(
    grid: Mapping[str, Any], geometry: xr.Dataset
) -> dict[str, xr.DataArray]:
    """Preserve actual source-grid physical geometry, without reading model fields."""
    layout = build_om4_layout()
    latitude, longitude = np.asarray(grid["lat"]), np.asarray(grid["lon"])
    ny, nx = len(latitude), len(longitude)
    for name, expected in (("lat", latitude), ("lon", longitude)):
        actual = np.asarray(geometry[name])
        axis = 0 if name == "lat" else 1
        if actual.ndim == 2:
            expected = np.broadcast_to(
                expected[:, None] if axis == 0 else expected[None], (ny, nx)
            )
        np.testing.assert_allclose(
            actual, expected, rtol=0, atol=1e-6, err_msg=f"Geometry {name}"
        )
    np.testing.assert_allclose(geometry.lev, grid["depth"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(geometry.lev, layout.depth_levels, rtol=0, atol=1e-6)
    np.testing.assert_allclose(geometry.dz, layout.depth_thickness, rtol=0, atol=1e-6)
    area = np.asarray(geometry["areacello"])
    if area.shape != (ny, nx) or not np.isfinite(area).all() or not (area > 0).all():
        raise ValueError(
            "Geometry must provide positive physical cell areas on the exact grid"
        )
    mask = np.asarray(grid["mask"], dtype=bool)
    if mask.shape != (STATE_CHANNELS, ny, nx):
        raise ValueError("Expected canonical 77-channel wet masks")
    coords = {
        "y": xr.DataArray(latitude, dims="y"),
        "x": xr.DataArray(longitude, dims="x"),
        "lat": xr.DataArray(
            np.broadcast_to(latitude[:, None], (ny, nx)), dims=("y", "x")
        ),
        "lon": xr.DataArray(
            np.broadcast_to(longitude[None], (ny, nx)), dims=("y", "x")
        ),
        "lev": xr.DataArray(np.asarray(geometry.lev), dims="lev"),
        "dz": xr.DataArray(np.asarray(geometry.dz), dims="lev"),
        "areacello": xr.DataArray(area, dims=("y", "x")),
    }
    for index, name in enumerate(("uo", "vo", "thetao", "so")):
        coords[f"wetmask_{name}"] = xr.DataArray(
            mask[index * MODEL_DEPTHS : (index + 1) * MODEL_DEPTHS],
            dims=("lev", "y", "x"),
        )
    coords["wetmask_zos"] = xr.DataArray(mask[-1], dims=("y", "x"))
    # The common OM4 geometry uses the same depth mask for each physical field.
    # Preserve per-variable masks regardless; a generic wetmask is valid only
    # when all four depth masks actually agree.
    if all(
        np.array_equal(mask[:MODEL_DEPTHS], mask[k : k + MODEL_DEPTHS])
        for k in range(MODEL_DEPTHS, 4 * MODEL_DEPTHS, MODEL_DEPTHS)
    ):
        coords["wetmask"] = coords["wetmask_thetao"]
    if "ocean_fraction" in geometry:
        fraction = np.asarray(geometry["ocean_fraction"])
        if (
            fraction.shape != (MODEL_DEPTHS, ny, nx)
            or not np.isfinite(fraction).all()
            or ((fraction < 0) | (fraction > 1)).any()
        ):
            raise ValueError("Invalid source ocean_fraction geometry")
        coords["ocean_fraction"] = xr.DataArray(fraction, dims=("lev", "y", "x"))
    for name in ("lat_b", "lon_b"):
        if name in geometry:
            value = geometry[name].load()
            coords[name] = xr.DataArray(value.values, dims=value.dims)
    return coords


def normalized_forcing(data: Samples, raw: np.ndarray) -> torch.Tensor:
    values = (raw - data.stats["atmosphere_mean"][:, None, None]) / data.stats[
        "atmosphere_std"
    ][:, None, None]
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite ERA5 forcing input")
    return data.tensor(values)[None, None]


def initial_inputs(
    data: Samples, archive: ObservationArchive, origin: pd.Timestamp
) -> dict[str, torch.Tensor]:
    starts = pd.date_range(
        origin - pd.Timedelta(days=HISTORY_DAYS),
        periods=HISTORY_STEPS,
        freq=f"{STEP_DAYS}D",
    )
    surface, forcing = [], []
    for start in starts:
        end = start + pd.Timedelta(days=STEP_DAYS)
        surface.append(
            np.concatenate(
                (
                    archive.interval("sst", start, end),
                    archive.interval("adt", start, end)[:1],
                )
            )
        )
        forcing.append(normalized_forcing(data, archive.interval("era5", start, end)))
    raw = np.stack(surface)
    wet = np.asarray(data.grid["mask"])[SURFACE_CHANNELS].astype(bool)
    valid = np.isfinite(raw) & wet
    normalized = (
        np.where(valid, raw, 0) - data.grid["mean"][SURFACE_CHANNELS, None, None]
    ) / data.grid["std"][SURFACE_CHANNELS, None, None]
    normalized = np.where(valid, normalized, 0)
    return {
        "surface": data.tensor(normalized)[None],
        "validity": data.tensor(valid)[None].bool(),
        "atmosphere": torch.cat(forcing, dim=1),
        "contexts": data.tensor(
            context_planes(
                data.grid["lat"],
                data.grid["lon"],
                starts + pd.Timedelta(days=STEP_DAYS / 2),
            )
        )[None],
    }


def write_physical_prediction(
    path: Path,
    physical: torch.Tensor,
    time: pd.Timestamp,
    coords: dict[str, xr.DataArray],
) -> None:
    """Write a physical frame using the standard analysis channel reassembly.

    ZarrWriter owns normalization through BatchPreprocessor. These predictions
    are already in physical units, so reuse its stack_levels operation directly
    instead of manufacturing a training-data preprocessor just for output.
    """
    layout = build_om4_layout()
    fields = physical.detach().cpu().numpy()
    data = xr.Dataset(
        {
            name: (("time", "y", "x"), fields[:, channel])
            for channel, name in enumerate(layout.prognostic_var_names)
        },
        coords={
            **coords,
            "time": xr.DataArray(np.array([time], dtype="datetime64[ns]"), dims="time"),
        },
    )
    data = stack_levels(data, layout).transpose("time", "lev", "y", "x", ...)
    data = data.chunk({"time": 1})
    if path.exists():
        data.to_zarr(path, mode="a", append_dim="time")
    else:
        data.to_zarr(
            path,
            mode="w",
            encoding={name: {"compressor": None} for name in data.data_vars},
        )


@torch.no_grad()
def export_rollout(
    model: SurfaceInitializedModel,
    data: Samples,
    archive: ObservationArchive,
    starts: pd.DatetimeIndex,
    ends: pd.DatetimeIndex,
    coords: dict[str, xr.DataArray],
    output: Path,
    checkpoint: Path,
    provenance: dict[str, Any],
) -> None:
    """Stream detached recurrent states and one physical five-day prediction at a time."""
    if len(starts) != len(ends) or not len(starts):
        raise ValueError("Need matching nonempty forecast intervals")
    if not np.all(ends - starts == pd.Timedelta(days=STEP_DAYS)) or not np.array_equal(
        starts[1:], ends[:-1]
    ):
        raise ValueError("Forecast intervals must be contiguous five-day means")
    if list(data.grid["names"]) != list(build_om4_layout().prognostic_var_names):
        raise ValueError(
            "Checkpoint fields do not use the canonical physical-channel order"
        )
    output.mkdir(parents=True, exist_ok=True)
    pred_path = output / "predictions.zarr"
    if pred_path.exists():
        raise FileExistsError(f"Predictions already exist: {pred_path}")
    write_json(provenance, output / "rollout.json")
    model.eval()
    inputs = initial_inputs(data, archive, starts[0])
    wet = data.mask.bool()
    with autocast(
        torch.device(data.device).type == "cuda", torch.bfloat16, device=data.device
    ):
        states = model.initialize(
            inputs["surface"],
            inputs["atmosphere"],
            inputs["contexts"][:, LAST_HISTORY_STEP],
            data.mask,
            inputs["validity"],
        )
        del inputs
        for step, (start, end) in enumerate(zip(starts, ends, strict=True)):
            future = normalized_forcing(data, archive.interval("era5", start, end))
            context = data.tensor(
                context_planes(
                    data.grid["lat"],
                    data.grid["lon"],
                    [start + pd.Timedelta(days=STEP_DAYS / 2)],
                )
            )[None]
            prediction, states = model.continue_forecast(
                states, future, context, data.mask, start_lead=step + 1
            )
            states = states.detach()
            physical = data.physical(prediction)[0]
            if not torch.isfinite(physical[:, wet]).all():
                raise FloatingPointError(
                    f"Nonfinite wet prediction at forecast step {step + 1}"
                )
            write_physical_prediction(
                pred_path,
                physical.masked_fill(~wet, float("nan")),
                start + pd.Timedelta(days=STEP_DAYS / 2),
                coords,
            )
            print(
                json.dumps(
                    {
                        "step": step + 1,
                        "steps": len(starts),
                        "interval_end": end.isoformat(),
                    }
                ),
                flush=True,
            )
    midpoint = starts + pd.Timedelta(days=STEP_DAYS / 2)
    metadata = xr.Dataset(
        {
            "time_bounds": (
                ("time", "bounds"),
                np.stack((starts.values, ends.values), axis=1),
            )
        },
        coords={
            "bounds": [0, 1],
            "lead_days": ("time", np.arange(1, len(starts) + 1) * STEP_DAYS),
        },
    )
    group = zarr.open_group(pred_path, mode="a")
    # Add only new arrays, preserving the already-written time coordinate.
    # Bounds share its CF units/calendar; fractional days need float encoding.
    time_encoding = {
        "units": group["time"].attrs["units"],
        "calendar": group["time"].attrs["calendar"],
        "dtype": "float64",
    }
    metadata.to_zarr(pred_path, mode="a", encoding={"time_bounds": time_encoding})
    group.attrs.update(
        {
            "model_path": str(checkpoint),
            "forecast_origin": starts[0].isoformat(),
            "forecast_end_exclusive": ends[-1].isoformat(),
            "temporal_operator": "five-day interval mean; time is interval midpoint",
            "initialization": "one 95-day observed surface history; no subsequent ocean reset",
            "forcing": "realized ERA5 only after initialization",
            "weights": "validation-selected checkpoint EMA",
            "rollout_provenance": provenance,
        }
    )
    group["time"].attrs["bounds"] = "time_bounds"
    group["lead_days"].attrs["units"] = "days"
    group["areacello"].attrs["units"] = "m2"
    group["dz"].attrs["units"] = "m"
    group["lev"].attrs.update({"units": "m", "positive": "down"})
    for name, units in (
        ("thetao", "degC"),
        ("so", "1e-3"),
        ("uo", "m s-1"),
        ("vo", "m s-1"),
        ("zos", "m"),
    ):
        group[name].attrs.update({"units": units, "cell_methods": "time: mean"})
    zarr.consolidate_metadata(pred_path)
    with xr.open_zarr(pred_path, decode_timedelta=False) as saved:
        np.testing.assert_array_equal(saved.time.values, midpoint.values)
        np.testing.assert_array_equal(
            saved.time_bounds.values, np.stack((starts.values, ends.values), axis=1)
        )
        np.testing.assert_array_equal(
            saved.areacello.values, coords["areacello"].values
        )
        np.testing.assert_array_equal(saved.dz.values, coords["dz"].values)
        for name in ("uo", "vo", "thetao", "so"):
            if saved[name].shape != (len(starts), MODEL_DEPTHS, *data.mask.shape[-2:]):
                raise AssertionError(f"Incomplete exported {name} dimensions")
    write_json(
        {
            "steps": len(starts),
            "start": starts[0].isoformat(),
            "end_exclusive": ends[-1].isoformat(),
            "checkpoint_sha256": provenance.get("checkpoint_sha256"),
            "verification": "time axis, bounds, physical geometry and full field dimensions read back",
        },
        output / "COMPLETE.json",
    )


def load_selected_model(
    checkpoint: Path, data: Samples
) -> tuple[SurfaceInitializedModel, dict[str, Any]]:
    # mmap avoids materializing replay-pool tensors that inference never uses.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    manifest = saved["manifest"]
    identity = archive_identity(data.daily)
    if identity != manifest["daily_identity"]:
        raise ValueError(
            "Daily observation/forcing archive differs from the checkpoint inventory"
        )
    assert samudra.__file__ is not None
    package = Path(samudra.__file__).parent
    for relative, expected in manifest["code_sha256"].items():
        if digest(package / relative) != expected:
            raise ValueError(
                f"Training source changed since checkpoint creation: {relative}"
            )
    for path in (
        data.root / "grid.npz",
        *(
            data.normalization / name
            for name in ("normalization.npz", "statistics.npz", "normalization.json")
        ),
    ):
        expected = {
            value
            for key, value in manifest["data_sha256"].items()
            if Path(key).name == path.name
        }
        if expected != {digest(path)}:
            raise ValueError(f"Checkpoint normalization/grid does not match: {path}")
    if "ema_params" not in saved["ema"]:
        raise ValueError("Selected checkpoint contains no EMA weights")
    model = SurfaceInitializedModel(data.grid["names"].tolist()).to(data.device)
    model.load_state_dict(saved["model"])
    # EMATracker operates on named_parameters of any nn.Module; its legacy
    # copy_to annotation still names only BaseModel/DDP.
    EMATracker.from_state(saved["ema"], model).copy_to(model)  # type: ignore[arg-type]
    return model.eval(), {
        "training_manifest": manifest,
        "checkpoint_updates": saved["updates"],
        "validation_selection_score": saved["best_score"],
        "weights": "ema",
        "daily_identity": identity,
        "normalization_artifacts": {
            str((data.normalization / name).resolve()): digest(
                data.normalization / name
            )
            for name in ("normalization.npz", "statistics.npz", "normalization.json")
        },
        "grid_sha256": digest(data.root / "grid.npz"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "observations",
        "normalization",
        "daily",
        "geometry",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--origin", default="2015-01-01")
    interval = parser.add_mutually_exclusive_group()
    interval.add_argument("--steps", type=int)
    interval.add_argument("--end-exclusive")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    requested_end = args.end_exclusive or (
        None if args.steps is not None else "2023-01-01"
    )
    starts, ends = forecast_intervals(
        args.origin, end_exclusive=requested_end, steps=args.steps
    )
    coverage = preflight_archive(args.daily, starts[0], ends[-1])
    data = Samples(
        args.observations,
        args.device,
        daily=args.daily,
        normalization=args.normalization,
    )
    archive = ObservationArchive(args.daily)
    for name in ("lat", "lon", "mask"):
        np.testing.assert_array_equal(archive.grid[name], data.grid[name])
    with xr.open_zarr(args.geometry, consolidated=True) as geometry:
        coords = analysis_coords(data.grid, geometry)
    model, provenance = load_selected_model(args.checkpoint, data)
    provenance.update(
        {
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": digest(args.checkpoint),
            "script_sha256": digest(__file__),
            "geometry": str(args.geometry.resolve()),
            "geometry_metadata_sha256": digest(args.geometry / ".zmetadata"),
            "daily": str(args.daily.resolve()),
            "input_shards": coverage,
            "origin": args.origin,
            "requested_end_exclusive": requested_end,
            "actual_end_exclusive": ends[-1].isoformat(),
            "omitted_trailing_days": None
            if requested_end is None
            else (pd.Timestamp(requested_end) - ends[-1]).days,
            "steps": len(starts),
        }
    )
    export_rollout(
        model,
        data,
        archive,
        starts,
        ends,
        coords,
        args.output,
        args.checkpoint,
        provenance,
    )


if __name__ == "__main__":
    main()
