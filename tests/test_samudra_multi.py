# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import cast

import pytest
import torch
from torch import nn

from samudra.datasets import TrainData
from samudra.models.modules import (
    DirectPatchDecoder,
    DirectPatchEncoder,
    PerceiverDecoder,
    PerceiverEncoder,
)
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.models.samudra_multi import SamudraMulti
from samudra.utils.ctx import GridContext


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
        encoder=DirectPatchEncoder(2, 4, (90.0, 90.0), geometry_mode="none"),
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


def test_zero_depth_reconstruction_loss_rejects_cross_grid_batches():
    lat = torch.tensor([-45.0, 45.0])
    input_lon = torch.tensor([45.0, 135.0, 225.0, 315.0])
    shifted_lon = input_lon + 10
    ctx = GridContext(
        label_mask=torch.ones(1, 2, 4, dtype=torch.bool),
        input_resolution_cpu=(lat, input_lon),
        output_resolution_cpu=(lat, shifted_lon),
    )
    model = SamudraMulti(
        in_channels=2,
        out_channels=1,
        pred_residuals=False,
        last_kernel_size=3,
        pad="circular",
        add_3d_coordinates=None,
        encoder=DirectPatchEncoder(2, 4, (90.0, 90.0), geometry_mode="none"),
        processor=nn.Identity(),
        decoder=DirectPatchDecoder(4, 1, (90.0, 90.0)),
        hist=0,
        checkpointing=None,
        gradient_detach_interval=0,
        use_bfloat16=False,
        zero_depth_reconstruction_weight=0.2,
    )

    with pytest.raises(ValueError, match="identical input and output grids"):
        model.reconstruct_once(torch.randn(2, 1, 2, 4), torch.randn(2, 1, 2, 4), ctx)
