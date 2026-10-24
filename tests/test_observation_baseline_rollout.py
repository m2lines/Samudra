# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Continuous rollout export uses only initial ocean history and preserves geometry."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

from samudra.constants import build_om4_layout
from samudra.utils.ema import EMATracker


@pytest.fixture
def exporter():
    path = Path(__file__).parents[1] / "scripts/rollout_observation_baseline.py"
    spec = importlib.util.spec_from_file_location("rollout_observation_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def grid_and_geometry():
    layout = build_om4_layout()
    latitude, longitude = np.array([-45.0, 45.0]), np.array([45.0, 135.0, 225.0, 315.0])
    mask = np.ones((77, 2, 4), dtype=bool)
    mask[:, 0, 0] = False
    mask[1, 1, 1] = False  # Preserve genuinely different channel masks.
    grid = dict(
        lat=latitude,
        lon=longitude,
        depth=np.array(layout.depth_levels),
        names=np.array(layout.prognostic_var_names),
        mask=mask,
        mean=np.arange(77, dtype=np.float32),
        std=np.full(77, 2, dtype=np.float32),
    )
    lat2d, lon2d = np.meshgrid(latitude, longitude, indexing="ij")
    lat_bounds, lon_bounds = np.meshgrid(
        np.array([-90, 0, 90]), np.arange(0, 361, 90), indexing="ij"
    )
    geometry = xr.Dataset(
        coords={
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
            "lev": ("lev", np.array(layout.depth_levels)),
            "dz": ("lev", np.array(layout.depth_thickness)),
            "areacello": (("y", "x"), np.full((2, 4), 6.4e13)),
            "lat_b": (("y_b", "x_b"), lat_bounds),
            "lon_b": (("y_b", "x_b"), lon_bounds),
            "ocean_fraction": (("lev", "y", "x"), mask[38:57].astype(float)),
        }
    )
    return grid, geometry


def fake_data(grid):
    mean = torch.tensor(grid["mean"])[None, None, :, None, None]
    std = torch.tensor(grid["std"])[None, None, :, None, None]
    return SimpleNamespace(
        grid=grid,
        mask=torch.tensor(grid["mask"]),
        device="cpu",
        stats={"atmosphere_mean": np.zeros(8), "atmosphere_std": np.ones(8)},
        tensor=lambda values: torch.as_tensor(values, dtype=torch.float32),
        physical=lambda values: values * std + mean,
    )


class FakeArchive:
    def __init__(self):
        self.calls = []
        self.origin = pd.Timestamp("2015-01-01")

    def interval(self, product, start, end):
        self.calls.append((product, start, end))
        assert end - start == pd.Timedelta(days=5)
        if product != "era5":
            assert end <= self.origin, "Future ocean data must never be requested"
        channels = {"sst": 1, "adt": 3, "era5": 8}[product]
        return np.ones((channels, 2, 4), dtype=np.float32)


class ContinuousModel(torch.nn.Module):
    def __init__(self, invalid_step=None):
        super().__init__()
        self.initializations, self.steps = 0, 0
        self.invalid_step = invalid_step

    def initialize(self, surface, atmosphere, context, mask, validity):
        self.initializations += 1
        assert surface.shape == (1, 19, 2, 2, 4)
        assert atmosphere.shape[1] == 19
        return torch.zeros(1, 2, 77, 2, 4)

    def continue_forecast(self, states, forcing, contexts, mask, start_lead):
        assert not states.requires_grad
        assert start_lead == self.steps + 1
        assert torch.all(states[:, -1] == self.steps)
        assert forcing.shape == (1, 1, 8, 2, 4)
        self.steps += 1
        prediction = states[:, -1:] + 1
        if self.steps == self.invalid_step:
            prediction[..., 1, 2] = float("nan")
        return prediction, torch.cat((states[:, -1:], prediction), dim=1)


def test_full_heldout_period_has_no_fabricated_partial_interval(exporter):
    starts, ends = exporter.forecast_intervals("2015-01-01", end_exclusive="2023-01-01")
    assert len(starts) == 584
    assert starts[0] == pd.Timestamp("2015-01-01")
    assert ends[-1] == pd.Timestamp("2022-12-30")
    assert (pd.Timestamp("2023-01-01") - ends[-1]).days == 2
    assert ((ends - starts) == pd.Timedelta(days=5)).all()
    np.testing.assert_array_equal(starts[1:], ends[:-1])


def test_archive_preflight_needs_only_past_ocean_and_all_forcing(exporter, tmp_path):
    origin, end = pd.Timestamp("2015-01-01"), pd.Timestamp("2015-01-06")
    for product in ("sst", "adt", "era5"):
        last = end if product == "era5" else origin
        for stamp in pd.date_range(
            origin - pd.Timedelta(days=95), last - pd.Timedelta(days=1)
        ):
            path = tmp_path / product / str(stamp.year) / f"{stamp:%Y-%m-%d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    assert exporter.preflight_archive(tmp_path, origin, end) == {
        "sst": 95,
        "adt": 95,
        "era5": 100,
    }
    (tmp_path / "era5" / "2015" / "2015-01-05.npz").unlink()
    with pytest.raises(FileNotFoundError, match="era5"):
        exporter.preflight_archive(tmp_path, origin, end)


def test_stream_export_retains_physical_fields_masks_bounds_and_geometry(
    exporter, tmp_path
):
    grid, geometry = grid_and_geometry()
    data, model, archive = fake_data(grid), ContinuousModel(), FakeArchive()
    starts, ends = exporter.forecast_intervals("2015-01-01", steps=3)
    coords = exporter.analysis_coords(grid, geometry)
    exporter.export_rollout(
        model,
        data,
        archive,
        starts,
        ends,
        coords,
        tmp_path,
        Path("best.pt"),
        {"checkpoint_sha256": "test-checkpoint"},
    )
    assert model.initializations == 1 and model.steps == 3
    assert sum(product == "era5" for product, _, _ in archive.calls) == 22
    assert sum(product == "sst" for product, _, _ in archive.calls) == 19
    with xr.open_zarr(tmp_path / "predictions.zarr", decode_timedelta=False) as output:
        assert output.thetao.dims == ("time", "lev", "y", "x")
        assert output.zos.dims == ("time", "y", "x")
        assert output.thetao.shape == (3, 19, 2, 4)
        np.testing.assert_array_equal(
            output.time.values, (starts + pd.Timedelta(days=2.5)).values
        )
        np.testing.assert_array_equal(output.time_bounds[:, 1].values, ends.values)
        np.testing.assert_array_equal(output.areacello, geometry.areacello)
        np.testing.assert_array_equal(output.dz, geometry.dz)
        np.testing.assert_array_equal(output.lat_b, geometry.lat_b)
        np.testing.assert_array_equal(output.wetmask_uo, grid["mask"][:19])
        assert np.isnan(output.uo.values[:, 1, 1, 1]).all()
        assert np.isnan(output.thetao.values[:, :, 0, 0]).all()
        assert output.thetao.values[0, 0, 1, 2] == 40
        assert output.thetao.values[2, 0, 1, 2] == 44
        assert output.time.attrs["bounds"] == "time_bounds"
        np.testing.assert_array_equal(output.lead_days, [5, 10, 15])
        assert output.attrs["weights"] == "validation-selected checkpoint EMA"
        assert (
            output.attrs["rollout_provenance"]["checkpoint_sha256"] == "test-checkpoint"
        )
    assert (tmp_path / "COMPLETE.json").exists()


def test_export_fails_loudly_on_nonfinite_wet_prediction(exporter, tmp_path):
    grid, geometry = grid_and_geometry()
    starts, ends = exporter.forecast_intervals("2015-01-01", steps=3)
    with pytest.raises(FloatingPointError, match="forecast step 2"):
        exporter.export_rollout(
            ContinuousModel(invalid_step=2),
            fake_data(grid),
            FakeArchive(),
            starts,
            ends,
            exporter.analysis_coords(grid, geometry),
            tmp_path,
            Path("best.pt"),
            {},
        )
    assert not (tmp_path / "COMPLETE.json").exists()


def test_geometry_mismatch_is_not_silently_reconstructed(exporter):
    grid, geometry = grid_and_geometry()
    geometry = geometry.assign_coords(lat=geometry.lat + 0.01)
    with pytest.raises(AssertionError, match="Geometry lat"):
        exporter.analysis_coords(grid, geometry)


def test_ocean_fraction_preserves_conservative_remapping_roundoff(exporter):
    grid, geometry = grid_and_geometry()
    fraction = geometry.ocean_fraction.values
    fraction[0, 1, 2] = 1.0000000000000844  # Published OM4 maximum.
    fraction[0, 0, 0] = -1e-14
    coords = exporter.analysis_coords(grid, geometry)
    np.testing.assert_array_equal(coords["ocean_fraction"], fraction)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1e-6, 1 + 1e-6])
def test_ocean_fraction_rejects_invalid_geometry(exporter, invalid):
    grid, geometry = grid_and_geometry()
    geometry.ocean_fraction.values[0, 1, 2] = invalid
    with pytest.raises(ValueError, match="Invalid source ocean_fraction"):
        exporter.analysis_coords(grid, geometry)


def test_selected_checkpoint_loads_ema_not_raw_weights(exporter, tmp_path, monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(7)
    ema = EMATracker(model).get_state(True)
    raw = {"weight": torch.ones(1, 1)}
    files = [
        tmp_path / name
        for name in (
            "grid.npz",
            "statistics.npz",
            "normalization.npz",
            "normalization.json",
        )
    ]
    for path in files:
        path.write_text(path.name)
    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "input.npz").write_text("fixed archive identity")
    package = tmp_path / "package"
    package.mkdir()
    source = package / "__init__.py"
    source.write_text("# checkpoint source\n")
    monkeypatch.setattr(exporter.samudra, "__file__", str(source))
    payload = {
        "model": raw,
        "ema": ema,
        "updates": 250,
        "best_score": 0.5,
        "manifest": {
            "code_sha256": {source.name: exporter.digest(source)},
            "daily_identity": exporter.archive_identity(daily),
            "data_sha256": {str(path): exporter.digest(path) for path in files},
        },
    }
    checkpoint = tmp_path / "best.pt"
    torch.save(payload, checkpoint)
    monkeypatch.setattr(
        exporter,
        "SurfaceInitializedModel",
        lambda names: torch.nn.Linear(1, 1, bias=False),
    )
    data = SimpleNamespace(
        root=tmp_path,
        normalization=tmp_path,
        daily=daily,
        device="cpu",
        grid={"names": np.array(["test"])},
    )
    selected, provenance = exporter.load_selected_model(checkpoint, data)
    torch.testing.assert_close(selected.weight, torch.full((1, 1), 7.0))
    assert provenance["weights"] == "ema"
    assert provenance["checkpoint_updates"] == 250

    source.write_text("# changed source\n")
    with pytest.raises(ValueError, match="Training source changed"):
        exporter.verify_checkpoint_source(payload["manifest"])
    with pytest.raises(ValueError, match="Training source changed"):
        exporter.load_selected_model(checkpoint, data)

    (daily / "input.npz").write_text("changed archive")
    with pytest.raises(ValueError, match="differs from the checkpoint inventory"):
        exporter.load_selected_model(checkpoint, data)
