# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""How strongly does the 12-step training rollout amplify a perturbation?

For each weight snapshot and each fixed validation window, infer the initial
state pair, then push a tangent vector through the 12 six-hour processor steps
with forward-mode differentiation, in float32 and eval mode. Two measures:

- ``tangent_growth[k]``: |δx_k| / |δx_0| for a random wet-cell perturbation of
  the initial pair, at each step k (all channels, and surface only). This is the
  linearized error growth the training loss sees.
- ``propagator_norm``: the largest singular value of the 12-step propagator
  ∂x_12/∂x_0, by power iteration on JᵀJ (jvp + vjp). Values far above the typical
  level mean some perturbation direction grows explosively within the window.

Norms are in normalized units over wet cells. The hypothesis under test: these
rise before each loss spike, for the observation task in particular.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.func import jvp, vjp

from samudra.observations.config import ObservationTrainConfig
from samudra.observations.pipeline import map_tensors
from samudra.observations.snapshot import LLC_STATE_NAMES
from samudra.observations.snapshot_data import SnapshotSources
from samudra.utils.ema import EMATracker

SURFACE = [LLC_STATE_NAMES.index(n) for n in ("thetao_0", "zos")]


def wet_norm(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """L2 norm over wet cells of [batch, ..., channel, y, x] values."""
    return torch.where(mask, values, 0).square().sum().sqrt()


def window_stability(model, sample, task, mask, *, seed, iterations):
    model.eval()
    with torch.no_grad():
        initial = model.initialize(
            sample["surface"][:, :29],
            sample["atmosphere"][:, :29],
            sample["contexts"][:, 28],
            mask,
            sample["validity"][:, :29],
            task,
        ).float()
    forcing = sample["atmosphere"][:, 29:]
    contexts = sample["contexts"][:, 29:]

    def rollout(states):
        predictions, _ = model.continue_forecast(
            states, forcing, contexts, mask, task=task
        )
        return predictions

    generator = torch.Generator(device=initial.device).manual_seed(seed)
    direction = torch.randn(initial.shape, generator=generator, device=initial.device)
    direction = torch.where(mask, direction, 0)
    direction = direction / wet_norm(direction, mask)
    _, tangent = jvp(rollout, (initial,), (direction,))
    tangent = tangent.detach()
    surface_mask = mask[SURFACE]
    growth = [float(wet_norm(tangent[:, k], mask)) for k in range(tangent.shape[1])]
    surface_initial = float(wet_norm(direction[:, :, SURFACE], surface_mask))
    surface_growth = [
        float(wet_norm(tangent[:, k, SURFACE], surface_mask)) / surface_initial
        for k in range(tangent.shape[1])
    ]

    def final(states):
        return rollout(states)[:, -1]

    vector, sigma = direction, float("nan")
    for _ in range(iterations):
        _, forward = jvp(final, (initial,), (vector,))
        _, pullback = vjp(final, initial)
        (back,) = pullback(torch.where(mask, forward, 0))
        back = torch.where(mask, back, 0)
        sigma = float(wet_norm(back, mask)) ** 0.5
        vector = back / wet_norm(back, mask)
    return {
        "tangent_growth": growth,
        "surface_tangent_growth": surface_growth,
        "propagator_norm": sigma,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--which", choices=["raw", "ema"], default="raw")
    parser.add_argument("--windows", type=int, default=4, help="Per source")
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cfg = ObservationTrainConfig.model_validate(yaml.safe_load(args.config.read_text()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sources = SnapshotSources(cfg, torch.device("cpu"))
    model = cfg.model.build(LLC_STATE_NAMES).to(device)
    mask = sources.obs.mask.to(device)
    windows = {}
    for source in sources.names:
        candidates = sources.origins[source, "validation"]
        chosen = candidates[
            np.unique(np.linspace(0, len(candidates) - 1, args.windows, dtype=int))
        ]
        windows[source] = [
            (int(o), map_tensors(sources.load(source, int(o)), lambda t: t.to(device)))
            for o in chosen
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as stream:
        for path in args.weights:
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(saved["model"])
            if args.which == "ema":
                EMATracker.from_state(saved["ema"], model).copy_to(model)
            for source, items in windows.items():
                task = "llc" if source == "llc" else "observation"
                for origin, sample in items:
                    result = window_stability(
                        model,
                        sample,
                        task,
                        mask,
                        seed=origin,
                        iterations=args.iterations,
                    )
                    record = dict(
                        weights=str(path),
                        updates=saved["updates"],
                        which=args.which,
                        source=source,
                        origin=origin,
                        **result,
                    )
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()


if __name__ == "__main__":
    main()
