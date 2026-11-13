# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""EMA restoration and temporary evaluation weights for the point specialist."""

from contextlib import contextmanager

import torch

from samudra.utils.ema import EMATracker


def restore_ema(model, config, state=None, step=0):
    if not 0 <= config["decay"] < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    if state is None:
        if step:
            raise ValueError("Missing EMA state for resumed updates")
        return EMATracker(model, **config)
    if "ema_params" not in state:
        raise ValueError("Resume requires EMA parameters, not just metadata")
    if int(state["num_updates"]) != step:
        raise ValueError("EMA update count does not match optimizer step")
    if (
        abs(float(state["decay"]) - config["decay"]) > 1e-7
        or state["faster_decay_at_start"] != config["faster_decay_at_start"]
    ):
        raise ValueError("EMA configuration changed on resume")
    device = next(model.parameters()).device
    state = dict(
        state, ema_params={k: v.to(device) for k, v in state["ema_params"].items()}
    )
    return EMATracker.from_state(state, model)


@contextmanager
def ema_weights(model, ema):
    """Restore raw weights and mode even if validation or checkpoint writing fails."""
    was_training = model.training
    with torch.no_grad():
        ema.store(model.parameters())
    try:
        ema.copy_to(model)
        yield
    finally:
        ema.restore(model.parameters())
        model.train(was_training)


def validation_comparison(raw, averaged, updates):
    if raw["step"] != averaged["step"] or raw["origins"] != averaged["origins"]:
        raise ValueError("Raw/EMA validation steps or dates differ")
    rows = []
    for r, e in zip(raw["per_depth"], averaged["per_depth"], strict=True):
        if r["depth"] != e["depth"] or r["count"] != e["count"]:
            raise ValueError("Raw/EMA validation populations differ")
        row = {"depth": r["depth"], "depth_m": r["depth_m"], "count": r["count"]}
        for key in ("rmse_c", "nmse", "bias_c", "bin_nll", "in_support_density_nll"):
            row[key] = (
                e[key] - r[key] if e[key] is not None and r[key] is not None else None
            )
        rows.append(row)
    nll = raw["mean_depth_bin_nll"]
    return dict(
        step=raw["step"],
        ema_updates=updates,
        raw=raw,
        ema=averaged,
        ema_minus_raw=dict(
            mean_depth_nmse=averaged["mean_depth_nmse"] - raw["mean_depth_nmse"],
            mean_depth_bin_nll=averaged["mean_depth_bin_nll"] - nll
            if nll is not None
            else None,
            per_depth=rows,
        ),
    )
