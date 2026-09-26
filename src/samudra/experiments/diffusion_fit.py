# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded optimizer-boundary checkpoints for the diffusion campaign."""

import math
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.observation_pilot import atomic_torch


def batch_at(update, size, batch_size, seed):
    """Reconstruct the paired training schedule from the completed-update count."""
    batches = size // batch_size
    if update < 0 or batches < 1:
        raise ValueError("Invalid update or training batch size")
    epoch, offset = divmod(update, batches)
    order = np.random.default_rng(seed + epoch).permutation(size)
    return order[offset * batch_size : (offset + 1) * batch_size].tolist()


def bounded_fit(
    model,
    optimizer,
    objective,
    validate,
    output,
    signature,
    *,
    max_updates,
    max_seconds,
    checkpoint_every=100,
    validate_every=500,
    max_new_updates=None,
    emit=lambda event: None,
):
    """Resume exact optimizer/model/RNG state; objectives derive data order from step.

    `max_new_updates` interrupts an invocation without declaring the stage complete.
    Limits count the full resumed stage, not a fresh allowance on each restart.
    The caller owns allocation accounting, including cache preparation and failures.
    """
    if min(max_updates, max_seconds, checkpoint_every, validate_every) <= 0:
        raise ValueError("Positive fitting limits required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    protocol = dict(
        signature=signature,
        max_updates=max_updates,
        max_seconds=max_seconds,
        validate_every=validate_every,
    )
    state = dict(step=0, elapsed=0.0, best=math.inf, complete=False)
    last = output / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol:
            raise ValueError("Resume protocol differs from the saved experiment")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        state = saved["state"]
        torch.set_rng_state(saved["cpu_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["device_rng"], device=device)
        del saved
    if state["complete"]:
        return state
    initial_step = state["step"]
    prior = state["elapsed"]
    started = time.monotonic()

    def save():
        state["elapsed"] = prior + time.monotonic() - started
        atomic_torch(
            dict(
                protocol=protocol,
                model=model.state_dict(),
                optimizer=optimizer.state_dict(),
                state=state.copy(),
                cpu_rng=torch.get_rng_state(),
                device_rng=torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            ),
            last,
        )

    def validation():
        model.eval()
        with torch.no_grad():
            score = float(validate(state["step"]))
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite validation objective")
        if score < state["best"]:
            state["best"] = score
            atomic_torch(
                dict(protocol=protocol, model=model.state_dict(), state=state.copy()),
                output / "best.pt",
            )
        emit(dict(event="validation", step=state["step"], objective=score))
        model.train()

    if not math.isfinite(state["best"]):
        validation()
        save()
    model.train()
    while (
        state["step"] < max_updates and prior + time.monotonic() - started < max_seconds
    ):
        if (
            max_new_updates is not None
            and state["step"] - initial_step >= max_new_updates
        ):
            break
        optimizer.zero_grad(set_to_none=True)
        loss = objective(state["step"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training objective")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        state["step"] += 1
        emit(
            dict(
                event="update",
                step=state["step"],
                loss=float(loss.detach()),
                gradient_norm=float(norm),
            )
        )
        if state["step"] % validate_every == 0:
            validation()
        if state["step"] % checkpoint_every == 0:
            save()
    state["complete"] = (
        state["step"] >= max_updates
        or prior + time.monotonic() - started >= max_seconds
    )
    # Score the actual final weights even if the budget ends between validations.
    if state["complete"] and state["step"] % validate_every:
        validation()
    save()
    return state
