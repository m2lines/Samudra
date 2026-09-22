# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Deterministic D transfer without changing any pretrained parameter shapes."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from samudra.experiments.initializer_models import HistoryInitializer
from samudra.experiments.surface_state import Evolution


class ObservationTransfer(nn.Module):
    def __init__(self, names):
        super().__init__()
        self.initializer = HistoryInitializer(names, "wide", True)
        self.evolution = Evolution(len(names), [128, 192, 256, 384], "ar")
        self.adapter = nn.Sequential(
            nn.Conv2d(8, 32, 1), nn.GELU(), nn.Conv2d(32, 3, 1)
        )
        last = self.adapter[-1]
        assert isinstance(last, nn.Conv2d)
        assert last.bias is not None
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.activation_checkpointing = True

    def load_core(self, state):
        # Strict core loading catches missing buffers, extra keys and wrong shapes.
        core = nn.Module()
        core.add_module("initializer", self.initializer)
        core.add_module("evolution", self.evolution)
        core.load_state_dict(state, strict=True)

    def train(self, mode=True):
        super().train(mode)
        # Preserve source running means/variances even during recomputation.
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def set_phase(self, phase):
        if phase not in ("adapter", "reconstruction", "joint", "frozen"):
            raise ValueError(phase)
        self.initializer.requires_grad_(phase in ("reconstruction", "joint"))
        self.evolution.requires_grad_(phase == "joint")
        self.adapter.requires_grad_(phase != "frozen")
        self.train()

    def adapt(self, atmosphere):
        b, t, _, h, w = atmosphere.shape
        return self.adapter(atmosphere.reshape(b * t, 8, h, w)).reshape(b, t, 3, h, w)

    def call(self, function, *args):
        if self.training and self.activation_checkpointing and torch.is_grad_enabled():
            return checkpoint(function, *args, use_reentrant=False)
        return function(*args)

    def initialize(self, surface, atmosphere, context, mask, validity):
        return self.call(
            self.initializer, surface, self.adapt(atmosphere), context, mask, validity
        )

    def forecast(self, surface, atmosphere, contexts, mask, validity):
        """Return physical-state-interface normalized outputs, with 6 or 7 leads."""
        initial = self.initialize(
            surface[:, :19], atmosphere[:, :19], contexts[:, 18], mask, validity[:, :19]
        )
        states = initial
        forcing = self.adapt(atmosphere[:, 19:])
        predictions = []
        for step in range(forcing.shape[1]):
            # One forcing frame supplied each call, and the exact interval context.
            prediction = self.call(
                self.evolution,
                states,
                forcing[:, step : step + 1],
                contexts[:, 19 + step],
                mask,
                1,
            )
            predictions.append(prediction)
            states = torch.stack((states[:, -1], prediction), 1)
        return torch.stack(predictions, 1), initial

    def reconstruct_month(
        self, surface, atmosphere, contexts, mask, validity, month_weights
    ):
        mean = None
        for step in range(len(month_weights)):
            end = 20 + step
            initial = self.initialize(
                surface[:, end - 19 : end],
                atmosphere[:, end - 19 : end],
                contexts[:, end - 1],
                mask,
                validity[:, end - 19 : end],
            )
            value = initial[:, -1] * month_weights[step]
            mean = value if mean is None else mean + value
        return mean


def main():
    """Real-grid checkpoint/gradient qualification; not a scientific skill score."""
    import argparse
    import hashlib
    import json
    from pathlib import Path

    import numpy as np

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(1729)
    grid = np.load(args.grid)
    model = ObservationTransfer(grid["names"].tolist())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model"]
    model.load_core(state)
    del state
    model = model.cuda().eval()
    mask = torch.as_tensor(grid["mask"], device="cuda", dtype=torch.float32)
    h, w = mask.shape[-2:]
    surface = (
        torch.randn(1, 19, 2, h, w, device="cuda") * mask[model.initializer.surface]
    )
    forcing = torch.randn(1, 19, 8, h, w, device="cuda")
    context = torch.zeros(1, 5, h, w, device="cuda")
    valid = mask[model.initializer.surface].expand_as(surface)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        adapted = model.initialize(surface, forcing, context, mask, valid)
        original = model.initializer(
            surface, torch.zeros(1, 19, 3, h, w, device="cuda"), context, mask
        )
        torch.testing.assert_close(adapted, original, atol=0, rtol=0)
        assert torch.isfinite(adapted).all()
    model.set_phase("reconstruction")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = model.initialize(surface, forcing, context, mask, valid)
        result[:, :, 40:52].float().square().mean().backward()
    gradients = {"initializer": model.initializer, "adapter": model.adapter}
    norms = {}
    for name, module in gradients.items():
        norm = (
            sum(
                float(p.grad.float().square().sum())
                for p in module.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError(f"No finite nonzero gradient reached {name}")
        norms[name] = norm
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with Path(args.checkpoint).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "checkpoint_sha256": digest,
        "strict_load": True,
        "zero_adapter_source_equivalence": "bitwise exact",
        "gradient_norms": norms,
        "shape": list(result.shape),
        "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
        "scope": "synthetic full-grid contract and gradient check; observational fitting still required",
    }
    (output / "QUALIFIED.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
