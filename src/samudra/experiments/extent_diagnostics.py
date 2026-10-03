# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-only task-gradient diagnostic at frozen, previously selected weights."""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from samudra.experiments.extent_training import ExtentPilot, patch_objective
from samudra.experiments.observation_joint import om4_objective
from samudra.experiments.observation_pilot import atomic_json, digest


def run(root, baseline_root):
    root, baseline_root = Path(root), Path(baseline_root)
    manifest = json.loads((root / "probe-U-patch-lr01" / "manifest.json").read_text())
    options = dict(manifest["arguments"])
    options.update(
        output=str(root / "gradient-diagnostic"),
        wandb_mode="disabled",
        joint_probe=True,
        patch_mode="shared",
        patch_leads=6,
        patch_lr_scale=1.0,
    )
    pilot = ExtentPilot(argparse.Namespace(**options))
    assert pilot.patches is not None
    parameters = {
        component: [
            p
            for name, p in getattr(pilot.model, component).named_parameters()
            if "input_adapters" not in name
        ]
        for component in ("initializer", "evolution")
    }
    indices = (0, 33, 101, 211)
    evidence: dict[str, Any] = dict(
        protocol="Four deterministic training examples per task; mean loss gradients before clipping. Shared parameters only, excluding task-specific input adapters. No optimizer steps; not a causal attribution or representative spatial survey.",
        checkpoints={},
    )
    for arm in ("U-global", "U-multitask"):
        checkpoint = baseline_root / arm / "best.pt"
        pilot.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)["model"],
            strict=True,
        )
        pilot.model.train()
        result = dict(
            checkpoint=str(checkpoint), sha256=digest(checkpoint), tasks={}, cosines={}
        )
        gradients = {}
        for task in ("observation", "global_om4", "native_patch"):
            pilot.model.zero_grad(set_to_none=True)
            losses = []
            truth_losses = []
            provenance = []
            for i in indices:
                seed = 1729 + 1000000 + i
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if task == "observation":
                        sample = pilot.data.load(pilot.training[i])
                        sample["mask_seed"] = seed
                        loss = pilot.objective(sample, "joint")
                        provenance.append(pilot.training[i].stem)
                    elif task == "global_om4":
                        coverage = pilot.data.load(pilot.training[i])["validity"][
                            :, :19
                        ]
                        loss = om4_objective(
                            pilot.model, pilot.om4, [i], 0.1, seed, coverage, 0.1
                        )
                        provenance.append(dict(train_origin_index=i))
                    else:
                        sample = pilot.patches.sample(seed)
                        loss = patch_objective(
                            pilot.model, sample, pilot.om4.names, 0.1, 0.1, seed
                        )
                        with torch.no_grad():
                            truth_loss = patch_objective(
                                pilot.model,
                                sample,
                                pilot.om4.names,
                                0.1,
                                0.1,
                                seed,
                                mode="truth",
                            )
                        truth_losses.append(float(truth_loss))
                        provenance.append(sample["provenance"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Nonfinite diagnostic loss: {arm}/{task}"
                        )
                    (loss / len(indices)).backward()
                    losses.append(float(loss.detach()))
            gradients[task] = {
                component: torch.cat(
                    [
                        (
                            p.grad.detach().cpu()
                            if p.grad is not None
                            else torch.zeros_like(p, device="cpu")
                        ).flatten()
                        for p in params
                    ]
                )
                for component, params in parameters.items()
            }
            result["tasks"][task] = dict(
                losses=losses,
                mean_loss=sum(losses) / len(losses),
                shared_gradient_norms={
                    k: float(g.norm()) for k, g in gradients[task].items()
                },
                provenance=provenance,
            )
            if truth_losses:
                result["tasks"][task]["true_initial_state_forecast_losses"] = (
                    truth_losses
                )
                result["tasks"][task]["truth_loss_caveat"] = (
                    "Forecast only; shared loss also includes 0.1 reconstruction and 0.1 completion. Do not interpret ratio as a pure initializer effect."
                )
        for left, right in (
            ("observation", "global_om4"),
            ("observation", "native_patch"),
            ("global_om4", "native_patch"),
        ):
            result["cosines"][left + " vs " + right] = {
                component: float(
                    torch.dot(gradients[left][component], gradients[right][component])
                    / (
                        gradients[left][component].norm()
                        * gradients[right][component].norm()
                    ).clamp_min(1e-30)
                )
                for component in parameters
            }
        evidence["checkpoints"][arm] = result
        atomic_json(evidence, root / "gradient-diagnostic" / "RESULTS.json")
    atomic_json(
        dict(checkpoints=list(evidence["checkpoints"]), optimizer_steps=0),
        root / "gradient-diagnostic" / "COMPLETE.json",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline-root", required=True)
    args = parser.parse_args()
    run(args.root, args.baseline_root)
