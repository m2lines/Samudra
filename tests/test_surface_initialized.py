# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Small CPU checks of the U-global architecture and replay boundary."""

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from samudra.config import UNetBackboneConfig
from samudra.models.surface_initialized import (
    SurfaceInitializedConfig,
    SurfaceInitializedModel,
    geometry_planes,
)
from samudra.observations.losses import (
    balanced_loss,
    channel_mse,
    completion_loss,
    corrupt_sample,
    structured_visibility,
)

NAMES = ["thetao_0", "thetao_1", "so_0", "uo_0", "vo_0", "zos"]


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(task="observation", steps=3):
    height, width = 12, 16
    latitude = torch.linspace(-75, 75, height)
    longitude = torch.arange(width) * (360 / width)
    lat, lon = torch.meshgrid(latitude.deg2rad(), longitude.deg2rad(), indexing="ij")
    xyz = torch.stack((lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()))
    context = torch.cat(
        (xyz, torch.zeros(1, height, width), torch.ones(1, height, width))
    )
    contexts = context[None, None].expand(1, 19 + steps, 5, height, width).clone()
    mask = torch.ones(len(NAMES), height, width, dtype=torch.bool)
    mask[:, 0, 0] = False
    surface = torch.randn(1, 19 + steps, 2, height, width)
    validity = mask[[0, 5]][None, None].expand_as(surface).clone()
    forcing = torch.randn(
        1, 19 + steps, 8 if task == "observation" else 3, height, width
    )
    return surface, forcing, contexts, mask, validity


def network(checkpointing=False):
    return SurfaceInitializedModel(
        NAMES, widths=(4,), activation_checkpointing=checkpointing
    )


@pytest.mark.parametrize("task", ["observation", "om4"])
def test_continuation_matches_uninterrupted_and_never_uses_future_surface(task):
    torch.manual_seed(11)
    model = network().eval()
    surface, forcing, contexts, mask, valid = example(task)
    with torch.no_grad():
        forecast, initial = model.forecast(
            surface, forcing, contexts, mask, valid, task
        )
        first, cache = model.continue_forecast(
            initial, forcing[:, 19:20], contexts[:, 19:20], mask, task=task
        )
        rest, final = model.continue_forecast(
            cache, forcing[:, 20:], contexts[:, 20:], mask, start_lead=2, task=task
        )
        torch.testing.assert_close(torch.cat((first, rest), 1), forecast)
        torch.testing.assert_close(final, forecast[:, -2:])
        surface[:, 19:] = float("nan")
        valid[:, 19:] = False
        changed, _ = model.forecast(surface, forcing, contexts, mask, valid, task)
        torch.testing.assert_close(changed, forecast)
    assert initial.shape[1:3] == (2, len(NAMES))
    assert forecast.shape[1:3] == (3, len(NAMES))
    assert torch.count_nonzero(forecast[..., 0, 0]) == 0


def test_missing_values_do_not_leak_and_known_surfaces_are_copied():
    model = network().eval()
    surface, atmosphere, contexts, mask, valid = example()
    valid[:, :, :, 3:6, 4:8] = False
    poisoned = surface.clone()
    poisoned[~valid] = float("nan")
    with torch.no_grad():
        first = model.initialize(
            surface[:, :19], atmosphere[:, :19], contexts[:, 18], mask, valid[:, :19]
        )
        second = model.initialize(
            poisoned[:, :19], atmosphere[:, :19], contexts[:, 18], mask, valid[:, :19]
        )
    torch.testing.assert_close(first, second)
    surface_state = first[:, :, model.initializer.surface]
    selected = valid[:, 17:19]
    torch.testing.assert_close(surface_state[selected], surface[:, 17:19][selected])
    assert torch.isfinite(first).all()


def test_geometry_and_original_branches_have_gradients():
    torch.manual_seed(17)
    model = network(checkpointing=True)
    surface, atmosphere, contexts, mask, valid = example(steps=1)
    valid[:, :, :, 2:8, 2:8] = False
    forecast, initial = model.forecast(surface, atmosphere, contexts, mask, valid)
    reconstruction = model.reconstruct_month(
        surface, atmosphere, contexts, mask, valid, [1.0]
    )
    loss = (
        forecast.square().mean()
        + initial.square().mean()
        + reconstruction.square().mean()
    )
    loss.backward()
    for module in (
        model.initializer.net,
        model.evolution.net,
        model.evolution.geometry,
        model.adapter[-1],
    ):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()
        )
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    assert all(m.affine for m in model.modules() if isinstance(m, nn.InstanceNorm2d))
    # Initialization is zero; geometry can subsequently learn through this projection.
    assert torch.count_nonzero(model.evolution.geometry.weight) == 0
    planes = geometry_planes(contexts[:, 0], mask)
    assert planes.shape[1] == 7
    assert torch.count_nonzero(planes[:, 4]) == 0


def test_native_forcing_bypasses_era5_adapter():
    model = network()
    surface, forcing, contexts, mask, valid = example("om4", steps=1)
    forecast, initial = model.forecast(surface, forcing, contexts, mask, valid, "om4")
    (forecast.square().mean() + initial.square().mean()).backward()
    assert all(p.grad is None for p in model.adapter.parameters())
    assert any(
        p.grad is not None for p in model.initializer.input_adapters["om4"].parameters()
    )
    with pytest.raises(ValueError, match="requires 8 channels"):
        model.forcing(forcing, "observation")


def test_corruption_is_reproducible_does_not_change_targets_or_rng():
    surface, _, _, _, validity = example()
    sample = {"surface": surface, "validity": validity}
    state = torch.get_rng_state()
    corrupted = corrupt_sample(sample, seed=42)
    assert torch.equal(state, torch.get_rng_state())
    assert torch.equal(corrupted["validity"], structured_visibility(validity, 42))
    assert torch.equal(corrupted["completion_target"], surface[:, 17:19])
    assert torch.equal(sample["validity"], validity)
    assert torch.all(corrupted["surface"][~corrupted["validity"]] == 0)
    assert corrupted["completion_valid"].any()


def test_losses_ignore_dry_nan_labels_and_balance_variable_groups():
    prediction = torch.zeros(1, 1, len(NAMES), 2, 2, requires_grad=True)
    target = torch.ones_like(prediction)
    weights = torch.ones(len(NAMES), 2, 2)
    weights[:, 0, 0] = 0
    target[..., 0, 0] = float("nan")
    torch.testing.assert_close(
        channel_mse(prediction, target, weights), torch.ones(1, 1, len(NAMES))
    )
    loss = balanced_loss(prediction, target, weights, NAMES)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    valid = weights.bool()[None, None, :2].expand(1, 1, 2, 2, 2)
    result = completion_loss(
        prediction[:, :, :2], target[:, :, :2], valid, [-30.0, 30.0]
    )
    torch.testing.assert_close(result, torch.tensor(1.0))
    zero = completion_loss(
        prediction[:, :, :2], target[:, :, :2], torch.zeros_like(valid), [-30.0, 30.0]
    )
    assert zero == 0


def test_variable_balancing_is_not_a_flat_channel_average():
    prediction = torch.zeros(1, 1, len(NAMES), 2, 2)
    target = torch.ones_like(prediction)
    target[:, :, :2] = 2  # Two temperature channels count as one variable group.
    weights = torch.ones(len(NAMES), 2, 2)
    torch.testing.assert_close(
        balanced_loss(prediction, target, weights, NAMES), torch.tensor(1.6)
    )
    torch.testing.assert_close(
        balanced_loss(prediction, target, weights, NAMES, interior_only=True),
        torch.tensor(1.75),  # thetao_1, so, uo and vo; no SST or SSH.
    )


def test_physical_state_has_all_77_channels_without_extra_memory():
    names = [
        f"{variable}_{level}"
        for variable in ("thetao", "so", "uo", "vo")
        for level in range(19)
    ] + ["zos"]
    model = SurfaceInitializedModel(names, widths=(4,), activation_checkpointing=False)
    surface, atmosphere, contexts, _, valid = example(steps=1)
    mask = torch.ones(77, *surface.shape[-2:], dtype=torch.bool)
    with torch.no_grad():
        forecast, initial = model.forecast(surface, atmosphere, contexts, mask, valid)
    assert forecast.shape[1:3] == (1, 77)
    assert initial.shape[1:3] == (2, 77)


def test_model_preset_builds_shared_backbone_config_and_applies_overrides():
    path = Path(__file__).parents[1] / "src/samudra/configs/observations/model.yaml"
    values = yaml.safe_load(path.read_text())
    values["backbone"].update(ch_width=[4], dilation=[1], n_layers=[1])
    values["backbone"]["core_block"]["activation"] = "relu"
    values["activation_checkpointing"] = False
    config = SurfaceInitializedConfig.model_validate(values)
    assert isinstance(config.backbone, UNetBackboneConfig)
    model = config.build(NAMES).eval()
    assert model.initializer.net[0].out_channels == 4
    assert model.evolution.net[0].out_channels == 4
    assert any(isinstance(layer, nn.ReLU) for layer in model.modules())
    assert all(m.affine for m in model.modules() if isinstance(m, nn.InstanceNorm2d))
    assert not model.activation_checkpointing
    with torch.no_grad():
        forecast, _ = model.forecast(*example(steps=1))
    assert torch.isfinite(forecast).all()


@pytest.mark.parametrize(
    "backbone",
    [
        {"ch_width": []},
        {"ch_width": [0, 4, 8, 16]},
        {"ch_width": [4], "dilation": [1, 2], "n_layers": [1]},
        {"ch_width": [4], "dilation": [0], "n_layers": [1]},
    ],
)
def test_model_config_rejects_invalid_backbone_before_building(backbone):
    with pytest.raises(ValueError):
        SurfaceInitializedConfig(backbone=backbone)
