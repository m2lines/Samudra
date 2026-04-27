# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from ocean_emulators.models.modules.blocks import (
    AvgPool,
    BilinearUpsample,
    ZonallyPeriodicBilinearUpsample,
)


def _pad_like_unet(feature: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
    crop = np.array(feature.shape[2:])
    target = np.array(skip.shape[2:])
    pads = target - crop
    if np.any(pads < 0):
        raise ValueError("Downsample/upsample feature is larger than skip connection")
    pads = [
        int(pads[1] // 2),
        int(pads[1] - pads[1] // 2),
        int(pads[0] // 2),
        int(pads[0] - pads[0] // 2),
    ]
    return torch.nn.functional.pad(feature, pads)


@pytest.mark.parametrize(
    "upsampler_cls",
    [
        pytest.param(BilinearUpsample, id="bilinear"),
        pytest.param(ZonallyPeriodicBilinearUpsample, id="zonally_periodic"),
    ],
)
@pytest.mark.parametrize("width", [8, 10, 16])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
def test_downscale_then_upscale_translation_invariance(
    upsampler_cls: type[torch.nn.Module],
    width: int,
    num_blocks: int,
):
    torch.manual_seed(0)
    downsample = AvgPool()
    upsample = upsampler_cls()

    total_downscale = 2**num_blocks
    if width / total_downscale <= 1:
        # if the final plane would be a width of 1, there's no shift we can try
        # below to confirm translation invariance
        pytest.skip("Width is too small for the given number of blocks")
    divisibility_condition = width % total_downscale == 0
    batch, channels, height = 1, 3, 8
    base_input = torch.arange(
        batch * channels * height * width, dtype=torch.float32
    ).reshape(batch, channels, height, width)

    def run_pipeline(original: torch.Tensor, count) -> torch.Tensor:
        if count == 0:
            return original
        with torch.no_grad():
            x = downsample(original)
            x = torch.square(x) + 1  # a nonlinearity
            x = run_pipeline(x, count - 1)
            x = upsample(x)
            padded = _pad_like_unet(x, original)
            result = padded + original
            return result

    baseline_output = run_pipeline(base_input, num_blocks)
    repeat_output = run_pipeline(base_input, num_blocks)
    assert torch.all(baseline_output == repeat_output), (
        "Downsample/upsample pipeline should be deterministic for identical inputs"
    )

    shift = total_downscale  # shift by one column of the most-downscaled version
    rotated_input = torch.roll(base_input, shifts=shift, dims=-1)
    rotated_output = run_pipeline(rotated_input, num_blocks)
    realigned_rotated_output = torch.roll(rotated_output, shifts=-shift, dims=-1)

    delta = baseline_output - realigned_rotated_output
    is_close = torch.all(baseline_output == realigned_rotated_output)

    if upsampler_cls == BilinearUpsample:
        if is_close:
            raise ValueError(
                "We expected outputs to be different after translation when using bilinear upsamples"
            )
        else:
            pytest.xfail(
                "outputs were not identical after using bilinear upsampling, as expected"
            )
    elif not divisibility_condition:
        if is_close:
            raise ValueError(
                "We expected outputs to be different after translation when size is not divisible by 2**num_downsamples"
            )
        else:
            pytest.xfail(
                "outputs were not identical after translation when size is not divisible by 2**num_downsamples, as expected"
            )
    else:
        assert is_close, (
            "Downsample/upsample pipeline should be translation-invariant along longitude; "
            f"max |delta|={delta.abs().max().item():.6e}, delta={delta}"
        )
