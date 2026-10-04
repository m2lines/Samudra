# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import zarr  # type: ignore[import-untyped]
from torch import nn

from samudra.experiments.extent_data import PatchSamples
from samudra.experiments.extent_models import (
    ExtentEvolution,
    LocalNet,
    geometry_planes,
    regional_padding,
)
from samudra.experiments.observation_pilot import digest
from samudra.experiments.surface_state import geographic_features
from samudra.experiments.task_schedule import TaskSchedule


def context(lat, lon):
    geo = geographic_features(lat, lon)[None]
    return torch.cat((geo, torch.zeros(1, 2, len(lat), len(lon))), 1)


def test_geometry_distinguishes_resolution_without_relabeling_crop():
    lat = torch.linspace(-20, 20, 81)
    lon = torch.arange(160).float() / 2
    ctx = context(lat, lon)
    mask = torch.ones(3, 81, 160, dtype=torch.bool)
    global_geo = geometry_planes(ctx, mask, False)
    crop_geo = geometry_planes(ctx[..., 10:50, 20:70], mask[..., 10:50, 20:70], True)
    torch.testing.assert_close(
        global_geo[:, :2, 10:49, 20:69], crop_geo[:, :2, :-1, :-1]
    )
    assert torch.all(crop_geo[:, 4] == 1)
    assert crop_geo[0, 2, 0, 0] < global_geo[0, 2, 0, 0]
    assert crop_geo[0, 3, 0, 0] < global_geo[0, 3, 0, 0]


def test_local_processor_six_step_crop_equivalence_with_supplied_states():
    torch.manual_seed(1)
    net = LocalNet(2, 2, width=4)
    globe = torch.randn(1, 2, 88, 112)
    patch = globe[..., 8:80, 16:96].clone()
    for _ in range(6):
        globe = net(globe, False)
        patch = net(patch, True)
    # 24-cell dependency radius; the core excludes every artificial boundary.
    torch.testing.assert_close(
        globe[..., 32:56, 40:72], patch[..., 24:48, 24:56], rtol=1e-5, atol=1e-6
    )
    assert not any(
        isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm, nn.LayerNorm))
        for m in net.modules()
    )


def test_regional_padding_restores_even_on_error():
    module = nn.Linear(1, 1)
    module.pad = "circular"  # type: ignore[assignment]
    with pytest.raises(RuntimeError), regional_padding(module, True):
        assert module.pad == "constant"
        raise RuntimeError("test")
    assert module.pad == "circular"


def test_checkpoint_recomputes_regional_boundary_and_routes_geometry_gradient():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(2)
    model = ExtentEvolution(2, local=True, local_width=4)
    inputs = torch.randn(1, 2, 2, 16, 24, requires_grad=True)
    forcing = torch.randn(1, 1, 3, 16, 24)
    ctx = context(torch.linspace(-20, 20, 16), torch.arange(24).float() / 4)
    mask = torch.ones(2, 16, 24, dtype=torch.bool)
    output = checkpoint(
        model, inputs, forcing, ctx, mask, 1, "om4-patch", use_reentrant=False
    )
    output.square().mean().backward()
    assert (
        model.geometry.weight.grad is not None
        and model.geometry.weight.grad.count_nonzero()
    )
    assert inputs.grad is not None and inputs.grad.count_nonzero()


def test_native_patch_sample_time_alignment_and_repeatability(tmp_path):
    root = tmp_path / "patch"
    root.mkdir()
    names = ["thetao_0", "zos"]
    shape = (30, 2, 128, 160)
    group = zarr.open_group(str(root / "fields.zarr"), mode="w")
    values = np.broadcast_to(np.arange(30, dtype="f4")[:, None, None, None], shape)
    group.create_dataset("prognostic", data=values)
    group.create_dataset("surface", data=values)
    group.create_dataset(
        "boundary",
        data=np.broadcast_to(
            np.arange(30, dtype="f4")[:, None, None, None], (30, 3, 128, 160)
        ),
    )
    group.create_dataset("verified", data=np.ones(30, dtype=bool))
    zarr.consolidate_metadata(group.store)
    import cftime

    times = np.arange(30) * 5.0
    dates = cftime.num2date(times, "days since 1975-01-03", "julian")
    stamps = np.array([str(d)[:10] for d in dates])
    np.savez(
        root / "grid.npz",
        names=np.array(names),
        mask=np.ones((2, 128, 160), dtype=bool),
        lat=np.linspace(-16, 16, 128),
        lon=np.arange(160) / 4,
        time=times,
        dates=stamps,
        time_units=np.array("days since 1975-01-03"),
        calendar=np.array("julian"),
    )
    (root / "manifest.json").write_text("{}")
    (root / "CACHE_READY.json").write_text(
        json.dumps(
            dict(
                manifest_sha256=digest(root / "manifest.json"),
                grid_sha256=digest(root / "grid.npz"),
            )
        )
    )
    for store, value in (("OM4_means.zarr", 0.0), ("OM4_stds.zarr", 1.0)):
        g = zarr.open_group(str(tmp_path / store), mode="w")
        for name in ("tauuo", "tauvo", "hfds"):
            g.create_dataset(name, data=np.array(value))
        zarr.consolidate_metadata(g.store)
    data = SimpleNamespace(
        names=names,
        device="cpu",
        mean=torch.zeros(2),
        std=torch.ones(2),
        args=SimpleNamespace(data_root=str(tmp_path)),
    )
    patches = PatchSamples(root, data)
    a, b = patches.sample(1729), patches.sample(1729)
    for key in ("surface", "truth", "labels", "forcing", "context"):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    t = a["surface"][0, 0, 0, 0, 0]
    assert a["truth"][0, 0, 0, 0, 0] == t + 17
    assert a["truth"][0, 1, 0, 0, 0] == t + 18
    assert a["forcing"][0, 0, 0, 0, 0] == t + 18
    assert a["labels"][0, -1, 0, 0, 0] == t + 24
    assert a["context"].shape == (1, 5, 128, 128)
    assert a["core"].sum() == 2 * 64 * 64


def test_exposure_budget_and_resume_patch_choice():
    schedule = TaskSchedule(2000, 2000, "mixed")
    seen = {"observation": 0, "global": 0, "patch": 0}
    for step in range(schedule.total):
        task = schedule.task(step)
        kind = (
            ("global" if schedule.counts(step)["om4"] % 2 == 0 else "patch")
            if task == "om4"
            else "observation"
        )
        seen[kind] += 1
    assert seen == {"observation": 2000, "global": 1000, "patch": 1000}


def test_native_calendar_matches_baseline_including_leap_gaps():
    from scripts.prepare_om4_patch_cache import validate_shared_time

    time = np.array([2.5, 7.5, 13.5, 18.5])
    attrs = {"units": "days since 1958-01-01", "calendar": "julian"}
    validate_shared_time(time, attrs, time.copy(), attrs)
    with pytest.raises(ValueError, match="timestamps differ"):
        validate_shared_time(time, attrs, time + 1, attrs)
    with pytest.raises(ValueError, match="timestamps differ"):
        validate_shared_time(time, attrs, time, {**attrs, "calendar": "noleap"})
    irregular = np.array([0.0, 5.0, 12.0])
    with pytest.raises(ValueError, match="Unexpected OM4 cadence"):
        validate_shared_time(irregular, attrs, irregular, attrs)


def test_omitted_patch_is_a_true_noop_and_preserves_future_sampling():
    import copy
    import time

    from samudra.experiments.extent_training import ExtentPilot
    from samudra.experiments.observation_joint import assert_exact_state

    pilot: Any = ExtentPilot.__new__(ExtentPilot)
    pilot.args = SimpleNamespace(patch_training=True, patch_mode="omit")
    pilot.schedule = TaskSchedule(6, 4, "mixed")
    pilot.completed = 1  # Second OM4 slot, hence auxiliary patch.
    pilot.model = nn.Linear(2, 1)
    pilot.optimizer = torch.optim.AdamW(pilot.model.parameters(), lr=1e-3)
    pilot.model(torch.ones(1, 2)).sum().backward()
    pilot.optimizer.step()
    before = copy.deepcopy((pilot.model.state_dict(), pilot.optimizer.state_dict()))
    rng = torch.get_rng_state().clone()
    records: list[dict[str, Any]] = []
    pilot.emit = records.append
    pilot.elapsed_seconds, pilot.started = 0, time.monotonic()
    pilot.train_update()
    assert pilot.completed == 2 and records[0]["event"] == "joint_skip"
    assert_exact_state(pilot.model.state_dict(), before[0])
    assert_exact_state(pilot.optimizer.state_dict(), before[1])
    assert torch.equal(rng, torch.get_rng_state())
    assert pilot.schedule.counts(pilot.completed) == {"om4": 2, "observation": 0}


def test_patch_learning_rate_and_gradient_contract_are_task_local():
    from samudra.experiments.extent_training import ExtentPilot

    pilot: Any = ExtentPilot.__new__(ExtentPilot)
    pilot.args = SimpleNamespace(
        patch_training=True,
        patch_mode="truth",
        patch_lr_scale=0.1,
        core_lr=1e-4,
        om4_lr=1e-4,
    )
    pilot.schedule = TaskSchedule(6, 4, "mixed")
    for slot, task, lr, components in [
        (0, "om4", 1e-4, {"initializer", "evolution"}),
        (1, "om4", 1e-5, {"evolution"}),
        (4, "observation", 1e-4, {"initializer", "evolution", "adapter"}),
    ]:
        pilot.completed = slot
        assert pilot.task_learning_rate(task) == pytest.approx(lr)
        assert pilot.required_gradient_components(task) == components


@pytest.mark.parametrize(
    "mode,leads",
    [("shared", 6), ("shared", 1), ("truth", 6), ("detach", 6), ("forecast", 6)],
)
def test_patch_horizon_and_truth_initialization_route_actual_gradients(mode, leads):
    from samudra.experiments.extent_training import patch_objective

    class Init(nn.Module):
        surface = [0, 4]

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.5))
            self.calls = 0

        def forward(self, *args):
            self.calls += 1
            return self.weight * torch.ones(1, 2, 5, 8, 8)

    class Evolution(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.9))
            self.leads = []

        def forward(self, states, forcing, context, mask, lead, task):
            self.leads.append(lead)
            return states[:, -1] * self.weight

    model = SimpleNamespace(
        initializer=Init(), evolution=Evolution(), call=lambda fn, *a: fn(*a)
    )
    sample = dict(
        surface=torch.ones(1, 19, 2, 8, 8),
        mask=torch.ones(5, 8, 8, dtype=torch.bool),
        past=torch.zeros(1, 19, 3, 8, 8),
        context=torch.zeros(1, 5, 8, 8),
        truth=torch.ones(1, 2, 5, 8, 8),
        forcing=torch.zeros(1, 6, 3, 8, 8),
        labels=torch.ones(1, 6, 5, 8, 8) * 2,
        weights=torch.ones(5, 8, 8),
        core=torch.ones(5, 8, 8, dtype=torch.bool),
        lat=torch.linspace(-10, 10, 8),
    )
    loss = patch_objective(
        model,
        sample,
        ["thetao_0", "so_0", "uo_0", "vo_0", "zos"],
        0.1,
        0.1,
        1729,
        mode=mode,
        leads=leads,
    )
    loss.backward()
    assert model.evolution.leads == list(range(1, leads + 1))
    assert model.evolution.weight.grad is not None and model.evolution.weight.grad != 0
    if mode == "truth":
        assert model.initializer.calls == 0 and model.initializer.weight.grad is None
    elif mode == "detach":
        assert model.initializer.calls == 1 and model.initializer.weight.grad is None
    else:
        assert (
            model.initializer.calls == 1 and model.initializer.weight.grad is not None
        )
    if mode in {"truth", "detach", "forecast"}:
        start = 1.0 if mode == "truth" else 0.5
        expected = (
            sum((start * 0.9**lead - 2) ** 2 for lead in range(1, leads + 1)) / leads
        )
        assert float(loss.detach()) == pytest.approx(expected, rel=1e-6)
