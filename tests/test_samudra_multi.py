# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import cast

import pytest
import torch
import xarray as xr
from torch import nn

from samudra.datasets import InferenceDataset, TrainData
from samudra.models.modules import (
    BoundaryEncoder,
    DirectPatchDecoder,
    DirectPatchEncoder,
    PerceiverDecoder,
    PerceiverEncoder,
    ResampleProjectionDecoder,
)
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.models.samudra_multi import SamudraMulti
from samudra.utils.ctx import GridContext
from samudra.utils.device import get_device


def _bare_module(module_type: type[nn.Module]) -> nn.Module:
    module = module_type.__new__(module_type)
    nn.Module.__init__(module)
    return module


def test_selective_checkpointing_wraps_only_representation_heads(monkeypatch):
    captured: dict[str, Callable[[nn.Module], bool]] = {}

    def capture_checkpointing(
        module: nn.Module,
        check_fn: Callable[[nn.Module], bool],
    ) -> None:
        captured["check_fn"] = check_fn

    monkeypatch.setattr(
        "samudra.models.samudra_multi.apply_activation_checkpointing",
        capture_checkpointing,
    )
    encoder = cast(PerceiverEncoder, _bare_module(PerceiverEncoder))
    processor = cast(UNetBackbone, _bare_module(UNetBackbone))
    decoder = cast(PerceiverDecoder, _bare_module(PerceiverDecoder))

    SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        hist=0,
        checkpointing="selective",
        gradient_detach_interval=0,
        use_bfloat16=True,
    )

    check_fn = captured["check_fn"]
    assert check_fn(encoder)
    assert check_fn(decoder)
    assert not check_fn(processor)


def test_selective_checkpointing_recognizes_direct_representation_heads(monkeypatch):
    captured: dict[str, Callable[[nn.Module], bool]] = {}

    def capture_checkpointing(
        module: nn.Module,
        check_fn: Callable[[nn.Module], bool],
    ) -> None:
        captured["check_fn"] = check_fn

    monkeypatch.setattr(
        "samudra.models.samudra_multi.apply_activation_checkpointing",
        capture_checkpointing,
    )
    encoder = cast(DirectPatchEncoder, _bare_module(DirectPatchEncoder))
    processor = cast(UNetBackbone, _bare_module(UNetBackbone))
    decoder = cast(DirectPatchDecoder, _bare_module(DirectPatchDecoder))

    SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        hist=0,
        checkpointing="selective",
        gradient_detach_interval=0,
        use_bfloat16=True,
    )

    check_fn = captured["check_fn"]
    assert check_fn(encoder)
    assert check_fn(decoder)
    assert not check_fn(processor)


def test_identity_processor_is_not_checkpointed(monkeypatch):
    captured: dict[str, Callable[[nn.Module], bool]] = {}

    def capture_checkpointing(
        module: nn.Module,
        check_fn: Callable[[nn.Module], bool],
    ) -> None:
        captured["check_fn"] = check_fn

    monkeypatch.setattr(
        "samudra.models.samudra_multi.apply_activation_checkpointing",
        capture_checkpointing,
    )
    encoder = cast(DirectPatchEncoder, _bare_module(DirectPatchEncoder))
    processor = nn.Identity()
    decoder = cast(DirectPatchDecoder, _bare_module(DirectPatchDecoder))

    SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=encoder,
        processor=processor,
        decoder=decoder,
        hist=0,
        checkpointing="selective",
        gradient_detach_interval=0,
        use_bfloat16=True,
    )

    assert not captured["check_fn"](processor)


class _CountingProcessor(nn.Module):
    in_channels = 2
    out_channels = 2

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 1


class _IdentityGridEncoder(nn.Module):
    out_channels = 1

    def __init__(self):
        super().__init__()
        self.calls = 0

    def output_resolution(self, resolution):
        return resolution

    def forward(self, x, resolution):
        del resolution
        if x.shape[1] != 1:
            raise AssertionError("The state encoder must receive prognostics only.")
        self.calls += 1
        return x[:, :1]


class _IdentityGridDecoder(nn.Module):
    out_channels = 1

    def forward(
        self,
        x,
        output_resolution,
        source_resolution=None,
        valid_mask=None,
    ):
        del output_resolution, source_resolution, valid_mask
        return x


@pytest.mark.parametrize("iterations", [0, 1, 2, 4])
def test_process_supports_zero_to_multiple_shared_iterations(iterations):
    processor = _CountingProcessor()
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=cast(DirectPatchEncoder, _bare_module(DirectPatchEncoder)),
        processor=processor,
        decoder=cast(DirectPatchDecoder, _bare_module(DirectPatchDecoder)),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        processor_iterations=iterations,
    )
    x = torch.zeros(1, 2, 3, 4)
    resolution = (torch.linspace(-60, 60, 3), torch.arange(4) * 90)

    output = model.process(x, resolution)

    if iterations == 0:
        assert output is x
    else:
        torch.testing.assert_close(output, torch.full_like(x, iterations))
    assert processor.calls == iterations


@pytest.mark.parametrize("iterations", [1, 2, 4])
def test_repeated_processor_retains_shared_parameter_gradients(iterations):
    processor = nn.Conv2d(4, 4, kernel_size=1)
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=cast(DirectPatchEncoder, _bare_module(DirectPatchEncoder)),
        processor=processor,
        decoder=cast(DirectPatchDecoder, _bare_module(DirectPatchDecoder)),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        processor_iterations=iterations,
    )
    features = torch.randn(2, 4, 3, 4, requires_grad=True)
    resolution = (torch.linspace(-60, 60, 3), torch.arange(4) * 90)

    model.process(features, resolution).square().mean().backward()

    assert features.grad is not None
    assert processor.weight.grad is not None
    assert torch.count_nonzero(processor.weight.grad) > 0


def test_latent_forecast_encodes_once_and_aligns_depth_with_time_and_forcing():
    lat = torch.tensor([-45.0, 45.0])
    lon = torch.tensor([45.0, 135.0])
    ctx = GridContext(
        label_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        input_resolution_cpu=(lat, lon),
        output_resolution_cpu=(lat, lon),
        input_mask=torch.ones(1, 2, 2, dtype=torch.bool),
    )
    encoder = _IdentityGridEncoder()
    boundary_encoder = BoundaryEncoder(1, 1)
    boundary_calls: list[torch.Tensor] = []
    boundary_encoder.register_forward_hook(
        lambda _module, inputs, _output: boundary_calls.append(inputs[0].detach())
    )
    with torch.no_grad():
        boundary_encoder.projection.weight.fill_(1.0)
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=cast(DirectPatchEncoder, encoder),
        processor=nn.Identity(),
        decoder=cast(DirectPatchDecoder, _IdentityGridDecoder()),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        boundary_encoder=boundary_encoder,
    )
    batch = TrainData(1, 2, ctx)
    initial = torch.full((1, 1, 2, 2), 10.0)
    boundaries = [1.0, 2.0, 4.0, 8.0]
    expected_by_depth = {1: 11.0, 2: 13.0, 4: 25.0}
    for depth, forcing in enumerate(boundaries, start=1):
        label_value = expected_by_depth.get(depth, -100.0)
        batch.append(
            initial,
            torch.cat(
                (
                    torch.full_like(initial, 1000.0 + forcing),
                    torch.full_like(initial, forcing),
                ),
                dim=1,
            ),
            torch.full_like(initial, label_value),
        )

    forecasts = model.latent_forecast(batch, [1, 2, 4])

    assert encoder.calls == 1
    assert len(boundary_calls) == 4
    for call, forcing in zip(boundary_calls, boundaries):
        torch.testing.assert_close(call, torch.full_like(initial, forcing))
    for depth, expected in expected_by_depth.items():
        torch.testing.assert_close(forecasts[depth], torch.full_like(initial, expected))

    def mse(pred, target):
        return (pred - target).square().mean(dim=(0, 2, 3))

    torch.testing.assert_close(
        model(batch, loss_fn=mse, processor_depth=2), torch.zeros(1)
    )


class _LatentInferenceDataset:
    def __init__(self, ctx: GridContext):
        self.ctx = ctx
        self.boundaries = [1.0, 2.0, 4.0, 8.0]
        self.label = torch.zeros(1, 1, 2, 2)

    def __getitem__(self, step):
        del step
        return self.label, self.label, self.label

    def get_boundary(self, step):
        forcing = self.boundaries[step]
        return torch.cat(
            (
                torch.full_like(self.label, 1000.0 + forcing),
                torch.full_like(self.label, forcing),
            ),
            dim=1,
        )

    def inference_target(self, steps):
        size = len(range(*steps.indices(len(self.boundaries))))
        return self.label.expand(size, -1, -1, -1)

    def get_target_time(self, start_step, num_steps):
        return xr.DataArray(torch.arange(start_step, start_step + num_steps).numpy())


def test_latent_inference_carries_latent_state_across_output_chunks():
    lat = torch.tensor([-45.0, 45.0])
    lon = torch.tensor([45.0, 135.0])
    ctx = GridContext(
        label_mask=torch.ones(1, 2, 2, dtype=torch.bool),
        input_resolution_cpu=(lat, lon),
        output_resolution_cpu=(lat, lon),
        input_mask=torch.ones(1, 2, 2, dtype=torch.bool),
    )
    encoder = _IdentityGridEncoder()
    boundary_encoder = BoundaryEncoder(1, 1)
    with torch.no_grad():
        boundary_encoder.projection.weight.fill_(1.0)
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=cast(DirectPatchEncoder, encoder),
        processor=nn.Identity(),
        decoder=cast(DirectPatchDecoder, _IdentityGridDecoder()),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        boundary_encoder=boundary_encoder,
    ).to(get_device())
    ctx = ctx.to(get_device())
    dataset = cast(InferenceDataset, _LatentInferenceDataset(ctx))

    state = model.initialize_rollout(torch.full((1, 1, 2, 2), 10.0), ctx)
    first = model.inference(dataset, state, steps_completed=0, num_steps=2)
    second = model.inference(
        dataset, first.rollout_state, steps_completed=2, num_steps=2
    )

    assert encoder.calls == 1
    torch.testing.assert_close(
        first.prediction[0], torch.full_like(first.prediction[0], 11.0)
    )
    torch.testing.assert_close(
        first.prediction[1], torch.full_like(first.prediction[1], 13.0)
    )
    torch.testing.assert_close(
        second.prediction[0], torch.full_like(second.prediction[0], 17.0)
    )
    torch.testing.assert_close(
        second.prediction[1], torch.full_like(second.prediction[1], 25.0)
    )
    torch.testing.assert_close(second.rollout_state, torch.full_like(state, 25.0))


def test_boundary_encoder_maps_one_boundary_state_to_latent_channels():
    encoder = BoundaryEncoder(2, 4)
    boundary = torch.randn(3, 2, 2, 5)
    resolution = (torch.tensor([-45.0, 45.0]), torch.arange(5) * 72.0)

    assert encoder(boundary, resolution, resolution).shape == (3, 4, 2, 5)


def test_boundary_encoder_resamples_forcing_to_the_latent_grid():
    encoder = BoundaryEncoder(1, 4)
    boundary = torch.randn(2, 1, 2, 4)
    source_resolution = (
        torch.tensor([-45.0, 45.0]),
        torch.tensor([45.0, 135.0, 225.0, 315.0]),
    )
    target_resolution = (torch.tensor([0.0]), torch.tensor([90.0, 270.0]))

    encoded = encoder(boundary, source_resolution, target_resolution)

    assert encoded.shape == (2, 4, 1, 2)


def test_decode_uses_current_input_masks_for_channelwise_resampling():
    source_resolution = (
        torch.tensor([-45.0, 45.0]),
        torch.tensor([45.0, 135.0]),
    )
    output_resolution = (torch.tensor([0.0]), torch.tensor([90.0]))
    decoder = ResampleProjectionDecoder(
        in_channels=2,
        out_channels=2,
        coordinate_resampling=True,
        project_before_resample=True,
    )
    with torch.no_grad():
        decoder.projection.weight.copy_(torch.eye(2)[:, :, None, None])
        assert decoder.projection.bias is not None
        decoder.projection.bias.zero_()
    model = SamudraMulti(
        in_channels=2,
        out_channels=2,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=cast(DirectPatchEncoder, _bare_module(DirectPatchEncoder)),
        processor=nn.Identity(),
        decoder=decoder,
        hist=1,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
    )
    features = torch.tensor([[[[1.0, 100.0], [1.0, 1.0]], [[100.0, 2.0], [2.0, 2.0]]]])
    # The leading pair represents the older history entry. Only the final
    # current-state masks correspond to the decoder's two output channels.
    input_mask = torch.tensor(
        [
            [[False, False], [False, False]],
            [[False, False], [False, False]],
            [[True, False], [True, True]],
            [[False, True], [True, True]],
        ]
    )
    ctx = GridContext(
        label_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        input_resolution_cpu=source_resolution,
        output_resolution_cpu=output_resolution,
        input_mask=input_mask,
    )

    output = model.decode(features, source_resolution, ctx)

    torch.testing.assert_close(output, torch.tensor([[[[1.0]], [[2.0]]]]))


def test_process_rejects_negative_iterations():
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=cast(DirectPatchEncoder, _bare_module(DirectPatchEncoder)),
        processor=nn.Identity(),
        decoder=cast(DirectPatchDecoder, _bare_module(DirectPatchDecoder)),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
    )

    with pytest.raises(ValueError, match="non-negative"):
        model.process(
            torch.zeros(1, 2, 3, 4),
            (torch.linspace(-60, 60, 3), torch.arange(4) * 90),
            iterations=-1,
        )


def test_training_loss_can_preserve_zero_depth_reconstruction():
    lat = torch.tensor([-45.0, 45.0])
    lon = torch.tensor([45.0, 135.0, 225.0, 315.0])
    ctx = GridContext(
        label_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        input_resolution_cpu=(lat, lon),
        output_resolution_cpu=(lat, lon),
    )
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=DirectPatchEncoder(1, 4, (90.0, 90.0), geometry_mode="none"),
        processor=_CountingProcessor(),
        decoder=DirectPatchDecoder(4, 1, (90.0, 90.0)),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        zero_depth_reconstruction_weight=0.2,
    )
    # Match the counting processor's declared width to the actual latent width.
    model.processor = nn.Sequential(nn.Conv2d(4, 4, 1), nn.GELU())
    prognostic = torch.randn(2, 1, 2, 4)
    boundary = torch.randn(2, 1, 2, 4)
    label = torch.randn(2, 1, 2, 4)
    batch = TrainData(1, 1, ctx)
    batch.append(prognostic, boundary, label)

    def mse(pred, target):
        return (pred - target).square().mean(dim=(0, 2, 3))

    forecast = model.forward_once(prognostic, boundary, ctx)
    reconstruction = model.reconstruct_once(prognostic, boundary, ctx)
    expected = mse(forecast, label) + 0.2 * mse(reconstruction, prognostic)

    torch.testing.assert_close(model(batch, loss_fn=mse), expected)


def test_zero_depth_reconstruction_loss_supports_cross_grid_batches_with_input_mask():
    lat = torch.tensor([-45.0, 45.0])
    input_lon = torch.tensor([45.0, 135.0, 225.0, 315.0])
    output_lat = torch.tensor([0.0])
    output_lon = torch.tensor([90.0, 270.0])
    ctx = GridContext(
        label_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        input_resolution_cpu=(lat, input_lon),
        output_resolution_cpu=(output_lat, output_lon),
        input_mask=torch.ones(1, 2, 4, dtype=torch.bool),
    )
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=DirectPatchEncoder(1, 4, (90.0, 90.0), geometry_mode="none"),
        processor=nn.Identity(),
        decoder=DirectPatchDecoder(4, 1, (90.0, 90.0)),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        zero_depth_reconstruction_weight=0.2,
    )

    reconstruction = model.reconstruct_once(
        torch.randn(2, 1, 2, 4),
        torch.randn(2, 1, 2, 4),
        ctx,
    )

    assert reconstruction.shape == (2, 1, 2, 4)


def test_zero_depth_reconstruction_rejects_cross_grid_without_input_mask():
    lat = torch.tensor([-45.0, 45.0])
    input_lon = torch.tensor([45.0, 135.0, 225.0, 315.0])
    ctx = GridContext(
        label_mask=torch.ones(1, 1, 2, dtype=torch.bool),
        input_resolution_cpu=(lat, input_lon),
        output_resolution_cpu=(torch.tensor([0.0]), torch.tensor([90.0, 270.0])),
    )
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=DirectPatchEncoder(1, 4, (90.0, 90.0), geometry_mode="none"),
        processor=nn.Identity(),
        decoder=DirectPatchDecoder(4, 1, (90.0, 90.0)),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        zero_depth_reconstruction_weight=0.2,
    )

    with pytest.raises(ValueError, match="requires an input mask"):
        model.reconstruct_once(
            torch.randn(2, 1, 2, 4),
            torch.randn(2, 1, 2, 4),
            ctx,
        )
