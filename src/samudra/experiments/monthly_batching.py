# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Within-update batching with the original per-example diffusion draw order."""

import hashlib
from collections import OrderedDict
from typing import Any

import pandas as pd
import torch

from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.monthly_latent import initializer_visibility
from samudra.experiments.monthly_pilot_data import observation_batch


def digest_tensor(value):
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def corrupt(target, support, generator, correlation=0.5):
    support = support.bool().expand_as(target) & torch.isfinite(target)
    if not support.any():
        raise ValueError("No observed target support")
    clean = torch.where(support, target, 0).float()
    sigma = (
        torch.randn(target.shape[0], device=target.device, generator=generator) * 1.2
        - 1.2
    ).exp()
    noise = diffusion_noise(target.shape, support, generator, correlation)
    return dict(
        clean=clean,
        support=support,
        sigma=sigma,
        noisy=(clean + sigma[:, None, None, None] * noise) * support,
    )


def prepare_observation(data, sample, generator):
    batch = observation_batch(data, sample, "surface", generator)
    item = dict(
        fields=batch["fields"],
        available=batch["available"],
        atmosphere=sample["atmosphere"],
        contexts=sample["contexts"],
        weights=sample["month_weights"],
        duration=float(pd.Timestamp(sample["name"]).days_in_month),
    )
    monthly = corrupt(batch["target"], batch["support"], generator)
    lead = int(
        torch.randint(
            len(item["weights"]), (), generator=generator, device=item["weights"].device
        )
    )
    target = torch.zeros_like(batch["target"])
    support = torch.zeros_like(batch["support"])
    target[:, [38, 76]] = sample["surface"][:, 19 + lead]
    support[:, [38, 76]] = sample["validity"][:, 19 + lead].bool()
    native = corrupt(target, support, generator)
    target0 = torch.zeros_like(target)
    support0 = torch.zeros_like(support)
    target0[:, [38, 76]] = sample["surface"][:, 18]
    support0[:, [38, 76]] = sample["validity"][:, 18].bool()
    initial = corrupt(target0, support0, generator)
    return dict(**item, lead=lead, decodes=[monthly, native, initial])


def prepare_model(batch, generator):
    available = initializer_visibility(batch["mask"], "surface", generator)[
        None, None
    ].expand_as(batch["fields"])
    future = batch["fields"][:, 19:]
    target = (future * batch["weights"][None, :, None, None, None]).sum(1)
    monthly = corrupt(target, batch["mask"], generator)
    lead = int(
        torch.randint(
            len(batch["weights"]), (), generator=generator, device=future.device
        )
    )
    native = corrupt(future[:, lead], batch["mask"], generator)
    initial = corrupt(batch["fields"][:, 18], batch["mask"], generator)
    return dict(
        fields=batch["fields"],
        available=available,
        forcing=batch["forcing"],
        contexts=batch["contexts"],
        weights=batch["weights"],
        duration=float(batch["duration"]),
        lead=lead,
        decodes=[monthly, native, initial],
    )


def draw_digest(item):
    """Qualification only: exact corrupted-input/sigma/support/lead identity."""
    return dict(
        lead=item["lead"],
        decodes=[
            {key: digest_tensor(value) for key, value in decode.items()}
            for decode in item["decodes"]
        ],
    )


def groups(items, maximum=8):
    buckets: OrderedDict[tuple[int, int], list[dict[str, Any]]] = OrderedDict()
    for item in items:
        key = (item["fields"].shape[1], len(item["weights"]))
        buckets.setdefault(key, []).append(item)
    for bucket in buckets.values():
        for offset in range(0, len(bucket), maximum):
            yield bucket[offset : offset + maximum]


def batched_loss(model, items, source, area):
    fields = torch.cat([item["fields"] for item in items])
    available = torch.cat([item["available"] for item in items])
    if all("atmosphere" in item for item in items):
        forcing = model.adapt(torch.cat([item["atmosphere"] for item in items]))
    else:
        forcing = torch.cat([item["forcing"] for item in items])
    states = model.trajectory(
        fields, available, forcing, torch.cat([item["contexts"] for item in items])
    )
    monthly = sum(
        state
        * torch.stack([item["weights"][index] for item in items])[
            :, None, None, None, None
        ]
        for index, state in enumerate(states[1:])
    )
    native = torch.cat(
        [
            states[item["lead"] + 1][index : index + 1]
            for index, item in enumerate(items)
        ]
    )
    per_task = []
    for task, latent in enumerate((monthly, native, states[0])):
        conditions = []
        for index, item in enumerate(items):
            duration = item["duration"] if task == 0 else 5.0
            position = (
                duration / 2
                if task == 0
                else (2.5 + 5 * item["lead"] if task == 1 else -2.5)
            )
            conditions.append(
                model.decoder.condition(
                    latent[index : index + 1],
                    item["decodes"][task]["support"],
                    duration,
                    position,
                )
            )
        tensors = {
            key: torch.cat([item["decodes"][task][key] for item in items])
            for key in ("clean", "support", "noisy", "sigma")
        }
        predicted = model.decoder.core(
            tensors["noisy"],
            tensors["sigma"],
            torch.cat(conditions),
            tensors["support"],
        )
        weights = tensors["support"] * area
        denominator = weights.sum((-2, -1))
        present = denominator > 0
        errors = ((predicted - tensors["clean"]).square() * weights).sum(
            (-2, -1)
        ) / denominator.clamp_min(1e-12)
        per_item = (errors * present).sum(-1) / present.sum(-1)
        per_task.append(
            per_item * (1 + tensors["sigma"].square()) / tensors["sigma"].square()
        )
    coefficients = (0.8, 0.2, 0.1) if source == "obs" else (0.5, 0.5, 0.1)
    total = sum(
        weight * loss for weight, loss in zip(coefficients, per_task, strict=True)
    )
    return (
        total.mean(),
        dict(
            interior=per_task[0].detach(),
            surface=per_task[1].detach(),
            reconstruction=per_task[2].detach(),
        ),
        states[-1].detach().float().square().mean((1, 2, 3, 4)).sqrt(),
    )
