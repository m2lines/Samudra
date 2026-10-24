# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
from typing import Any

import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from scripts.benchmark_monthly_microbatch import decoder_loss


def test_batched_loss_preserves_per_item_masks_duration_noise_and_mean():
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = MonthlyLatentOcean(8, 8, (8, 16), history=2, encoder_width=8).eval()
    latent = torch.randn(2, 2, 8, 2, 4)
    area = torch.ones(8, 16)
    items: list[dict[str, Any]] = []
    for index in range(2):
        target = torch.randn(1, 77, 8, 16)
        mask = torch.rand(1, 77, 8, 16) > 0.4
        if index == 1:
            mask[:, :38] = False
        items.append(
            dict(
                targets=[target] * 3,
                supports=[mask] * 3,
                index=index,
                duration=28.0 + index * 3,
                lead=index,
            )
        )
    for task in range(3):
        expected = []
        for index, item in enumerate(items):
            generator = torch.Generator().manual_seed(1729 + 100 * index + task)
            duration = item["duration"] if task == 0 else 5.0
            position = (
                duration / 2
                if task == 0
                else (2.5 + 5 * item["lead"] if task == 1 else -2.5)
            )
            expected.append(
                model.decoder.loss(
                    latent[index : index + 1],
                    item["targets"][task],
                    item["supports"][task],
                    area,
                    generator,
                    duration_days=duration,
                    position_days=position,
                )
            )
        actual = decoder_loss(model, latent, items, task, area, 0)
        torch.testing.assert_close(
            actual, torch.stack(expected).mean(), rtol=1e-6, atol=1e-6
        )
