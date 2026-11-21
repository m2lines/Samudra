# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Adversarial timestamp/support checks for the snapshot runtime adapter."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from samudra.config import BlockConfig, UNetBackboneConfig
from samudra.models.surface_initialized import SurfaceInitializedConfig
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.snapshot import LLC_STATE_NAMES, SnapshotTaskConfig
from samudra.observations.snapshot_data import SnapshotSources


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def snapshot_sources(tmp_path, monkeypatch):
    """A storage oracle with values encoding timestamp and physical channel."""
    prepared, llc = tmp_path / "prepared", tmp_path / "llc.zarr"
    prepared.mkdir()
    times = pd.date_range("2012-01-01", "2012-03-01", freq="6h", inclusive="left")
    scalar = np.arange(len(times), dtype=np.float32)[:, None, None, None]
    depths = np.arange(51, dtype=np.float32)[None, :, None, None]
    data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    for i, name in enumerate(("U", "V", "Theta", "Salt")):
        values = np.broadcast_to(scalar + depths + i * 1000, (len(times), 51, 2, 3))
        data[name] = (("time", "k", "y", "x"), values)
    data["Eta"] = (
        ("time", "y", "x"),
        np.broadcast_to(scalar[:, 0] + 4000, (len(times), 2, 3)),
    )
    xr.Dataset(
        data,
        coords={
            "time": times,
            "lat": ("y", [-30.0, 30.0]),
            "lon": ("x", [0.0, 120.0, 240.0]),
        },
    ).to_zarr(llc, consolidated=True)
    np.savez(
        prepared / "grid.npz",
        names=np.array(LLC_STATE_NAMES),
        wet_area=np.ones((205, 2, 3)),
        lat=[-30.0, 30.0],
        lon=[0.0, 120.0, 240.0],
    )
    np.savez(
        prepared / "statistics.npz",
        state_mean=np.zeros(205),
        state_std=np.ones(205),
        atmosphere_mean=np.zeros(8),
        atmosphere_std=np.ones(8),
    )
    proof = dict(
        version="llc-obs-six-hour-v1",
        sha256={name: sha(prepared / name) for name in ("grid.npz", "statistics.npz")},
        llc_metadata_sha256=sha(llc / ".zmetadata"),
        daily_root=str(tmp_path / "daily"),
        time_start="2012-01-01",
        time_end="2012-03-01",
        splits={
            source: {
                "train": ["2012-01-01", "2012-02-01"],
                "validation": ["2012-02-01", "2012-03-01"],
            }
            for source in ("obs", "llc")
        },
        excluded=[["2012-01-15", "2012-01-16"]],
    )
    (prepared / "COMPLETE.json").write_text(json.dumps(proof))
    model = SurfaceInitializedConfig(
        backbone=UNetBackboneConfig(
            ch_width=[4], dilation=[1], n_layers=[1], core_block=BlockConfig()
        ),
        history_steps=29,
        tasks=("llc", "observation"),
        forcing_mode="shared_era5",
    )
    cfg = ObservationTrainConfig.model_validate(
        dict(
            experiment={
                "name": "test-snapshot-runtime",
                "data_root": str(tmp_path),
                "base_output_dir": str(tmp_path),
                "wandb": {"mode": "disabled", "project": "snapshot-runtime-test"},
            },
            model=model,
            data={"llc": str(llc), "snapshots": str(prepared)},
            snapshot=SnapshotTaskConfig(),
            pretrain_mix="llc-obs",
            fresh_examples=8,
            post_training_updates=0,
            update_prefetch=True,
            loss={"monthly_ts": 0},
        )
    )
    sources = SnapshotSources(cfg, torch.device("cpu"))

    def shard(relative, product):
        day = pd.Timestamp(Path(relative).stem)
        # Each forcing slot encodes its interval END, independent of loader arithmetic.
        endpoints = pd.date_range(day + pd.Timedelta(hours=6), periods=4, freq="6h")
        hours = np.array(
            (endpoints - times[0]).total_seconds() / 3600, dtype=np.float32
        )
        if product == "era5":
            return np.broadcast_to(hours[:, None, None, None], (4, 8, 2, 3)).copy()
        value = (day - times[0]).total_seconds() / 3600
        return np.full(
            (1, 2, 3), value + (0 if product == "sst" else 10000), dtype=np.float32
        )

    monkeypatch.setattr(sources, "shard", shard)
    return sources


def test_exclusion_applies_to_preceding_forcing_support(snapshot_sources):
    sources = snapshot_sources
    start, end = np.datetime64("2012-01-15"), np.datetime64("2012-01-16")
    for source in sources.names:
        for index in sources.origins[source, "train"]:
            frame_times = sources.times[index] + sources.task.offsets()
            # The first forcing frame begins six hours before the first ocean frame.
            support_start = frame_times[0] - np.timedelta64(6, "h")
            assert not (support_start < end and frame_times[-1] >= start), (
                source,
                sources.times[index],
                support_start,
                frame_times[-1],
            )


def test_native_state_reads_preserve_all_depths_and_recurrent_times(snapshot_sources):
    sources = snapshot_sources
    index = int(np.flatnonzero(sources.times == np.datetime64("2012-01-10"))[0])
    sample = sources.load("llc", index)
    assert sample["initial_fields"].shape == (1, 2, 205, 2, 3)
    assert sample["target_fields"].shape == (1, 12, 205, 2, 3)
    for base, channel in ((0, 0), (1000, 51), (2000, 102), (3000, 153)):
        torch.testing.assert_close(
            sample["initial_fields"][0, :, channel, 0, 0],
            torch.tensor([index - 1 + base, index + base], dtype=torch.float32),
        )
        torch.testing.assert_close(
            sample["target_fields"][0, :, channel + 50, 0, 0],
            torch.arange(
                index + 1 + base + 50, index + 13 + base + 50, dtype=torch.float32
            ),
        )
    expected_hours = torch.tensor(
        (sources.times[index] + sources.task.offsets() - np.datetime64("2012-01-01"))
        / np.timedelta64(1, "h"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(sample["atmosphere"][0, :, 0, 0, 0], expected_hours)


def test_daily_obs_slots_are_sparse_and_forcing_retains_six_hour_phase(
    snapshot_sources,
):
    sources = snapshot_sources
    index = int(np.flatnonzero(sources.times == np.datetime64("2012-01-10"))[0])
    sample = sources.load("obs", index)
    availability = sample["validity"][0, :, 0, 0, 0]
    assert torch.nonzero(availability).flatten().tolist() == list(range(0, 41, 4))
    assert (sample["surface"][0, ~availability] == 0).all()
    expected_hours = torch.tensor(
        (sources.times[index] + sources.task.offsets() - np.datetime64("2012-01-01"))
        / np.timedelta64(1, "h"),
        dtype=torch.float32,
    )
    torch.testing.assert_close(sample["atmosphere"][0, :, 0, 0, 0], expected_hours)
    torch.testing.assert_close(
        sample["surface"][0, availability, 0, 0, 0], expected_hours[availability]
    )


def test_era5_six_hour_bins_use_distinct_state_and_accumulation_hours(
    tmp_path, monkeypatch
):
    """Identity remapping isolates temporal arithmetic from the ESMF dependency."""
    import importlib.util
    import sys
    from types import ModuleType, SimpleNamespace

    script = (
        Path(__file__).resolve().parents[1]
        / "data/prepare_llc_observation_snapshots.py"
    )
    spec = importlib.util.spec_from_file_location("snapshot_prep_review", script)
    assert spec is not None and spec.loader is not None
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    module = ModuleType("ocean_preprocessing.regridding")

    class IdentityRemap:
        def __init__(self, *args):
            pass

        def __call__(self, values):
            return values, np.ones_like(values)

    monkeypatch.setattr(module, "FiniteCoverageRemap", IdentityRemap, raising=False)
    monkeypatch.setattr(module, "rectilinear_grid", lambda *args: None, raising=False)
    monkeypatch.setitem(sys.modules, "ocean_preprocessing.regridding", module)
    output, source = tmp_path / "out", tmp_path / "source"
    output.mkdir()
    raw_root = source / "era5-surface.zarr"
    raw_root.mkdir(parents=True)
    (raw_root / ".zmetadata").write_text("{}")
    coordinates = dict(
        lat=[-30.0, 30.0],
        lon=[0.0, 120.0, 240.0],
        lat_b=[-60.0, 0.0, 60.0],
        lon_b=[-60.0, 60.0, 180.0, 300.0],
    )
    np.savez(
        output / "grid.npz",
        lat=coordinates["lat"],
        lon=coordinates["lon"],
        lat_b=coordinates["lat_b"],
        lon_b=coordinates["lon_b"],
    )
    times = pd.date_range("2013-01-01", periods=25, freq="h")
    hours = np.arange(25, dtype=np.float32)
    ds = xr.Dataset(
        {
            name: (
                ("time", "lat", "lon"),
                np.broadcast_to(hours[:, None, None], (25, 2, 3)),
            )
            for name in prep.FORCING
        },
        coords={**coordinates, "time": times},
    )
    monkeypatch.setattr(prep.xr, "open_zarr", lambda *args, **kwargs: ds)
    prep.era5(
        SimpleNamespace(
            output=output, source=source, source_grid=None, year=2013, limit=1
        )
    )
    with np.load(output / "era5/2013/2013-01-01.npz") as shard:
        values = shard["values"][:, :, 0, 0]
        assert values.shape == (4, 8)
        for slot in range(4):
            state_mean = slot * 6 + 2.5
            accumulated_mean = slot * 6 + 3.5
            np.testing.assert_allclose(values[slot, :5], state_mean)
            np.testing.assert_allclose(values[slot, 5:7], accumulated_mean / 3600)
            np.testing.assert_allclose(values[slot, 7], accumulated_mean * 1000 / 3600)
        np.testing.assert_array_equal(
            shard["interval_end"] - shard["interval_start"],
            np.full(4, np.timedelta64(6, "h")),
        )
