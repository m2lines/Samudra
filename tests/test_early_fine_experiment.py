# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from samudra.experiments import extent_models, initializer_models
from samudra.experiments.early_fine_data import WetCoarsener, load_grid_bounds
from samudra.experiments.early_fine_models import FineEncoder
from samudra.experiments.early_fine_training import fine_objective
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.surface_state import geographic_features


def bounds(h, w):
    x, y = np.meshgrid(np.linspace(0, 360, w + 1), np.linspace(-90, 90, h + 1))
    return {"lat_b": y, "lon_b": x}


def test_coarsener_preserves_wet_constant_and_does_not_count_land():
    remap = WetCoarsener(bounds(16, 32), bounds(4, 8))
    wet = torch.ones(3, 16, 32, dtype=torch.bool)
    wet[:, :, :2] = False
    field = torch.full((2, 3, 16, 32), 7.0)
    field[:, :, :, :2] = float("nan")
    torch.testing.assert_close(remap(field, wet), torch.full((2, 3, 4, 8), 7.0))
    torch.testing.assert_close(
        remap.integrate(torch.ones(16, 32)).sum(),
        torch.tensor(4 * np.pi, dtype=torch.float32),
    )


def test_published_gaussian_geometry_matches_centers_and_conserves_area(tmp_path):
    from pathlib import Path

    from samudra.experiments.observation_pilot import digest

    base = Path(__file__).parent / "fixtures" / "early_fine_grids"
    grids = {}
    for key in ["coarse", "fine"]:
        path = base / (key + "-bounds.npz")
        with np.load(path) as data:
            store = {k: data[k] for k in ["x", "y"]}
        grids[key] = load_grid_bounds(store, path, digest(path))
        with pytest.raises(ValueError, match="differs from qualification"):
            load_grid_bounds(store, path, "wrong digest")
        with pytest.raises(AssertionError):
            load_grid_bounds({**store, "y": store["y"] + 0.01}, path)
    coarsener = WetCoarsener(grids["fine"], grids["coarse"])
    ones = torch.ones(720, 1440)
    torch.testing.assert_close(
        coarsener.integrate(ones).sum(), torch.tensor(4 * np.pi, dtype=torch.float32)
    )
    torch.testing.assert_close(coarsener(ones, ones.bool()), torch.ones(180, 360))


@pytest.mark.parametrize("latent", [0, 10])
def test_fine_task_keeps_recurrent_state_and_trains_io_without_gold_initialization(
    monkeypatch, latent
):
    def tiny(i, o, w):
        return nn.Sequential(
            nn.Sequential(nn.Conv2d(i, 8, 1), nn.GELU()), nn.Conv2d(8, o, 1)
        )

    monkeypatch.setattr(initializer_models, "make_unet", tiny)
    monkeypatch.setattr(extent_models, "make_unet", tiny)
    names = [f"{v}_{i}" for v in ["uo", "vo", "thetao", "so"] for i in range(19)] + [
        "zos"
    ]

    def model(kind):
        torch.manual_seed(1729)
        return ObservationTransfer(
            names, "instance", kind, "unet", "observed-only", "input-adapters", latent
        )

    net = model("extent-fine")
    coarse = model("extent-unet")
    for name, p in coarse.state_dict().items():
        torch.testing.assert_close(p, net.state_dict()[name], rtol=0, atol=0)
    mask = torch.ones(77, 4, 8, dtype=torch.bool)
    fine_mask = torch.ones(77, 16, 32, dtype=torch.bool)
    lat = torch.linspace(-70, 70, 4)
    lon = torch.arange(8).float() * 45
    geo = geographic_features(lat, lon)
    fine_geo = geographic_features(
        torch.linspace(-85, 85, 16), torch.arange(32).float() * 11.25
    )
    source = SimpleNamespace(
        surface_ids=[38, 76],
        coarsen=WetCoarsener(bounds(16, 32), bounds(4, 8)),
        mask=fine_mask,
        weights=fine_mask.float(),
        names=names,
        lat=torch.linspace(-85, 85, 16),
    )
    global_data = SimpleNamespace(mask=mask, weights=mask.float(), geo=geo)
    sample = dict(
        surface=torch.randn(1, 19, 2, 16, 32),
        past=torch.randn(1, 19, 3, 16, 32),
        forcing=torch.randn(1, 6, 3, 16, 32),
        truth=torch.randn(1, 2, 77, 16, 32),
        labels=torch.randn(1, 6, 77, 16, 32),
        context=torch.cat((fine_geo, torch.zeros(2, 16, 32)), 0)[None],
    )
    net.activation_checkpointing = True
    states = []
    net.evolution.register_forward_pre_hook(
        lambda _, a: states.append(a[0].detach().clone())
    )
    outputs = []

    def hook(_, a, out):
        out.retain_grad()
        outputs.append(out)

    handle = net.evolution.register_forward_hook(hook)
    loss = fine_objective(
        net,
        source,
        sample,
        global_data,
        torch.ones(1, 19, 2, 4, 8, dtype=torch.bool),
        17,
        0.1,
        0.1,
    )
    initial = states[0].clone()
    assert states[0].shape == (1, 2, 77 + latent, 4, 8)
    torch.testing.assert_close(states[1][:, -1], outputs[0].detach())
    handle.remove()
    loss.backward()
    assert all(
        any(p.grad is not None and p.grad.count_nonzero() for p in m.parameters())
        for m in [
            net.initializer.fine_encoder,
            net.evolution.fine_decoder,
            net.initializer.net,
            net.evolution.net,
        ]
    )
    if latent:
        assert outputs[0].grad[:, 77:].count_nonzero()
        assert net.initializer.fine_encoder.net[-1].weight.grad[77:87].count_nonzero()
    # Changing all gold state targets changes loss, never the initialized input state.
    net.eval()
    states.clear()
    sample["truth"] = sample["truth"] + 100
    sample["labels"] = sample["labels"] - 100
    with torch.no_grad():
        fine_objective(
            net,
            source,
            sample,
            global_data,
            torch.ones(1, 19, 2, 4, 8, dtype=torch.bool),
            17,
            0.1,
            0.1,
        )
    torch.testing.assert_close(states[0], initial, rtol=0, atol=0)


def test_encoder_reduces_spatial_dimensions_exactly_fourfold():
    encoder = FineEncoder(87)
    value = encoder(
        torch.zeros(1, 19, 2, 24, 40),
        torch.zeros(1, 19, 3, 24, 40),
        torch.ones(1, 19, 2, 24, 40, dtype=torch.bool),
        torch.zeros(1, 5, 24, 40),
    )
    assert value.shape == (1, 2, 87, 6, 10)


def test_fine_decoder_reference_wraps_longitude():
    from samudra.experiments.early_fine_models import FineDecoder

    decoder = FineDecoder(3, 2)
    for parameter in decoder.parameters():
        nn.init.zeros_(parameter)
    state = torch.arange(8).float()[None, None, None].expand(1, 3, 4, 8)
    result = decoder(state, torch.zeros(1, 5, 4, 8), torch.ones(2, 16, 32))
    torch.testing.assert_close(result[..., 0], torch.full((1, 2, 16), 7 * 0.375))
    torch.testing.assert_close(result[..., -1], torch.full((1, 2, 16), 7 * 0.625))


def test_local_read_cache_copies_exact_bytes_and_preserves_missing_keys(tmp_path):
    from samudra.experiments.early_fine_data import LocalReadCache

    source = tmp_path / "source"
    source.mkdir()
    (source / "chunk").write_bytes(bytes(range(256)) * 64)
    cache = LocalReadCache(source, tmp_path / "cache")
    expected = (source / "chunk").read_bytes()
    assert cache["chunk"] == expected
    assert (tmp_path / "cache/chunk").read_bytes() == expected
    # Source reads are unnecessary once the immutable chunk is copied.
    (source / "chunk").unlink()
    assert "chunk" in cache
    assert cache["chunk"] == expected
    assert "missing" not in cache
    with pytest.raises(KeyError):
        cache["missing"]


def test_prefetch_keeps_explicit_seed_order_and_global_rng_on_replay():
    from samudra.experiments.early_fine_data import SamplePrefetch

    def read(seed):
        return np.random.default_rng(seed).standard_normal((2, 3))

    before = np.random.get_state()
    prefetch = SamplePrefetch(read, limit=4)
    try:
        prefetch.plan([5, 7, 11, 13])
        for seed in [7, 5]:
            np.testing.assert_array_equal(prefetch.take(seed), read(seed))
        # A restored checkpoint may return to previously consumed seeds.
        prefetch.plan([5, 7])
        for seed in [5, 7, 23]:
            np.testing.assert_array_equal(prefetch.take(seed), read(seed))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        assert before[0] == after[0] and before[2:] == after[2:]
        with pytest.raises(ValueError, match="memory bound"):
            prefetch.plan(range(5))
    finally:
        prefetch.close()


def test_prefetch_read_errors_are_not_silenced():
    from samudra.experiments.early_fine_data import SamplePrefetch

    def fail(seed):
        raise OSError("missing native chunk")

    prefetch = SamplePrefetch(fail)
    try:
        prefetch.plan([3])
        with pytest.raises(IOError, match="missing native chunk"):
            prefetch.take(3)
    finally:
        prefetch.close()


def test_early_prefetched_fields_match_original_channel_and_time_layout():
    from concurrent.futures import ThreadPoolExecutor

    import cftime

    from samudra.experiments.early_fine_data import EarlySamples, SamplePrefetch

    data = EarlySamples.__new__(EarlySamples)
    data.names = [f"field_{i}" for i in range(77)]
    data.forcing_names = ["tauuo", "tauvo", "hfds"]
    data.surface_ids = [38, 76]
    data.origins = list(range(5))
    data.store = {
        n: np.arange(30 * 2 * 4, dtype="f4").reshape(30, 2, 4) + i * 1000
        for i, n in enumerate(data.names + data.forcing_names)
    }
    data.pool = ThreadPoolExecutor(max_workers=4)
    data.prefetch = SamplePrefetch(data.read_cpu)
    data.device = "cpu"
    data.mask = torch.ones(77, 2, 4, dtype=torch.bool)
    data.mean = torch.zeros(77)
    data.std = torch.ones(77)
    data.fm = torch.zeros(1, 3, 1, 1)
    data.fs = torch.ones(1, 3, 1, 1)
    data.geo = torch.zeros(3, 2, 4)
    data.dates = [cftime.DatetimeNoLeap(1958, 1, i + 1) for i in range(30)]
    try:
        data.prefetch.plan([1729])
        result = data.sample(1729)
        start = int(np.random.default_rng(1729).integers(5))
        expected = np.stack(
            [data.store[n][start + 17 : start + 25] for n in data.names], axis=1
        )
        np.testing.assert_array_equal(result["truth"][0], expected[:2])
        np.testing.assert_array_equal(result["labels"][0], expected[2:])
        np.testing.assert_array_equal(
            result["surface"][0],
            np.stack(
                [
                    data.store[data.names[i]][start : start + 19]
                    for i in data.surface_ids
                ],
                axis=1,
            ),
        )
        np.testing.assert_array_equal(
            result["forcing"][0],
            np.stack(
                [data.store[n][start + 18 : start + 24] for n in data.forcing_names],
                axis=1,
            ),
        )
    finally:
        data.prefetch.close()
        data.pool.shutdown()


@pytest.mark.parametrize("failure", [False, True])
def test_deterministic_reference_restores_production_flags(monkeypatch, failure):
    from samudra.experiments.early_fine_training import deterministic_replay

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        with deterministic_replay():
            assert torch.are_deterministic_algorithms_enabled()
            assert torch.backends.cudnn.deterministic
            assert not torch.backends.cudnn.benchmark
            if failure:
                raise RuntimeError("injected diagnostic failure")
    except RuntimeError:
        assert failure
    assert previous == (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )


@pytest.mark.parametrize("difference", [0.0, 1e-12])
def test_early_replay_requires_bitwise_reference_equality(monkeypatch, difference):
    from samudra.experiments.early_fine_training import EarlyPilot
    from samudra.experiments.observation_joint import JointPilot

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    def verify(self):
        assert torch.are_deterministic_algorithms_enabled()
        self.resume_evidence = {
            "native_and_serialized_replay": {
                "serialized": {"rmse": difference, "max_absolute": difference}
            }
        }

    monkeypatch.setattr(JointPilot, "verify_resume_update", verify)
    pilot = EarlyPilot.__new__(EarlyPilot)
    if difference:
        with pytest.raises(ValueError, match="bitwise exact"):
            pilot.verify_resume_update()
    else:
        pilot.verify_resume_update()
        assert pilot.resume_evidence["deterministic_reference_bitwise_exact"]
