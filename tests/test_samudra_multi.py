# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from typing import cast

from torch import nn

from samudra.models.modules import PerceiverDecoder, PerceiverEncoder
from samudra.models.modules.unet_backbone import UNetBackbone
from samudra.models.samudra_multi import SamudraMulti


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
