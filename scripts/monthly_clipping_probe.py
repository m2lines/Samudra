# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Non-mutating AdamW update magnitudes from a saved optimizer state."""

import math


def adamw_delta(parameter, gradient, state, group, scale):
    """Exact mathematical next-step delta for ordinary, non-AMSGrad AdamW."""
    if group.get("amsgrad") or group.get("maximize"):
        raise ValueError("Unsupported Adam variant")
    beta1, beta2 = group["betas"]
    step = int(state["step"]) + 1
    gradient = gradient * scale
    first = state["exp_avg"] * beta1 + gradient * (1 - beta1)
    second = state["exp_avg_sq"] * beta2 + gradient.square() * (1 - beta2)
    direction = (first / (1 - beta1**step)) / (
        (second / (1 - beta2**step)).sqrt() + group["eps"]
    )
    return -group["lr"] * (direction + group["weight_decay"] * parameter)


def update_probe(model, saved_optimizer):
    (group,) = saved_optimizer["param_groups"]
    params = list(model.named_parameters())
    assert len(params) == len(group["params"])
    total = math.sqrt(
        sum(
            float(p.grad.detach().float().square().sum())
            for _, p in params
            if p.grad is not None
        )
    )
    results = {}
    for threshold in (1.0, 5.0, math.inf):
        scale = min(1.0, threshold / (total + 1e-6))
        totals: dict[str, list[float]] = {}
        for (name, p), index in zip(params, group["params"], strict=True):
            module = name.split(".")[0]
            weight = p.detach().float().cpu()
            values = totals.setdefault(module, [0.0, 0.0, 0.0])
            values[1] += float(weight.square().sum())
            if p.grad is None:
                continue  # Adam skips moments and decay for absent gradients.
            gradient = p.grad.detach().float().cpu()
            delta = adamw_delta(
                weight, gradient, saved_optimizer["state"][index], group, scale
            )
            values[0] += float(delta.square().sum())
            values[2] += float(gradient.square().sum())
        results[str(threshold)] = {
            "clip_scale": scale,
            "lr": group["lr"],
            "modules": {
                name: {
                    "update_norm": math.sqrt(v[0]),
                    "weight_norm": math.sqrt(v[1]),
                    "update_to_weight": math.sqrt(v[0] / max(v[1], 1e-30)),
                    "gradient_norm": math.sqrt(v[2]),
                }
                for name, v in totals.items()
            },
        }
    return {"preclip_global_norm": total, "counterfactual_next_steps": results}


def channel_stats(value):
    value = value.detach().float()
    axes = tuple(i for i in range(value.ndim) if i != value.ndim - 3)
    return {
        "mean": value.mean(axes).cpu().tolist(),
        "rms": value.square().mean(axes).sqrt().cpu().tolist(),
        "max_abs": value.abs().amax(axes).cpu().tolist(),
    }
