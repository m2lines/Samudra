# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Real-grid OM4 fit and observation-gradient qualification; never a skill result."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from samudra.experiments.diffusion_beta_qualify import data_inputs
from samudra.experiments.diffusion_observations import forecast_observation_crps
from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples


def gradient_norm(module):
    return math.sqrt(
        sum(
            float(p.grad.float().square().sum())
            for p in module.parameters()
            if p.grad is not None
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=["A", "B"], required=True)
    parser.add_argument("--sampling-steps", type=int, default=16)
    parser.add_argument("--fit-steps", type=int, default=10)
    args = parser.parse_args()
    if args.fit_steps < 2 or args.sampling_steps < 2:
        raise ValueError("Qualification needs at least two fit and sampling steps")
    args.output.mkdir(parents=True, exist_ok=False)
    if not (args.root / "DATA_READY.json").is_file():
        raise ValueError("Verified staged data required")
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    chosen = json.loads((args.root / "checkpoints/selection.json").read_text())
    upstream = args.root / "checkpoints" / chosen["selected"]
    manifest = json.loads((upstream / "manifest.json").read_text())
    for name, key in [
        ("grid.npz", "grid_sha256"),
        ("statistics.npz", "statistics_sha256"),
        ("SHA256SUMS", "data_manifest_sha256"),
    ]:
        if digest(data.root / name) != manifest[key]:
            raise ValueError(f"Observation contract changed: {name}")
    source = args.root / "checkpoints/om4-source/selected.pt"
    source_contract = json.loads((source.parent / "QUALIFIED.json").read_text())
    expected = source_contract["checkpoint_sha256"]
    if source_contract["normalization"] != "instance":
        raise ValueError("OM4 source normalization differs")
    scaling = source_contract["state_scaling"]
    if scaling["statistics_sha256"] != manifest["statistics_sha256"]:
        raise ValueError("OM4 source observation statistics differ")
    for name in ("mean", "std"):
        torch.testing.assert_close(
            data.tensor(scaling[name]), data.tensor(data.grid[name]), rtol=0, atol=0
        )
    if digest(source) != expected:
        raise ValueError("Pre-observation source checkpoint changed")
    core = ObservationTransfer(list(data.grid["names"]), "instance")
    core.load_core(torch.load(source, map_location="cpu", weights_only=False)["model"])
    model = (
        JointPhysicalForecast(
            core, stochastic=args.arm == "B", sampling_steps=args.sampling_steps
        )
        .cuda()
        .train()
    )
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    inputs, target, mask, weights, date = data_inputs(args.root, data)
    h, w = mask.shape[-2:]
    count = core.initializer.history
    nsurface = count * 2
    surface = inputs[:, :nsurface].reshape(1, count, 2, h, w)
    context = inputs[:, 2 * nsurface : 2 * nsurface + 5]
    past = inputs[:, 2 * nsurface + 5 :].reshape(1, count, 3, h, w)
    truth = target.reshape(1, 2, 77, h, w)
    started = time.monotonic()
    report = dict(
        arm=args.arm,
        sampling_steps=args.sampling_steps,
        om4_origin=date,
        source_checkpoint_sha256=expected,
        code_commit=os.environ["SAMUDRA_CODE_COMMIT"],
        data_manifest_sha256=manifest["data_manifest_sha256"],
        scope="Training-only fitting, gradient and cost probe; not production or skill comparison",
    )
    atomic_json(report, args.output / "manifest.json")

    def om4_loss():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.pretraining_loss(
                surface,
                past,
                context,
                truth,
                data.mask,
                weights[:77],
                torch.Generator(device="cuda").manual_seed(91),
            )

    losses = []
    for step in range(args.fit_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = om4_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        print(
            json.dumps(dict(event="om4_fit", step=step + 1, loss=losses[-1])),
            flush=True,
        )
    with torch.no_grad():
        final_fit = float(om4_loss())
    if not math.isfinite(final_fit) or final_fit >= losses[0]:
        raise ValueError("Anchored OM4 fitting did not improve")
    report.update(
        om4_losses=losses,
        om4_final_loss=final_fit,
        fit_seconds=time.monotonic() - started,
    )
    path = data.paths("train")[0]
    sample = data.load(path)
    args_model = (
        sample["surface"],
        sample["atmosphere"],
        sample["contexts"],
        data.mask,
        sample["validity"],
    )
    # The fitting probe uses training observations only, never validation/test labels.
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prediction, initial = model.forecast(
            *args_model,
            generator=torch.Generator(device="cuda").manual_seed(331),
            members=2 if args.arm == "B" else 1,
        )
        loss = (
            forecast_observation_crps(data, prediction, sample)
            if args.arm == "B"
            else data.forecast_loss(prediction[0], sample)
        )
    if not bool(torch.isfinite(prediction).all()) or not bool(torch.isfinite(loss)):
        raise FloatingPointError("Nonfinite sampled observation forecast/loss")
    loss.backward()
    norms = {
        name: gradient_norm(module)
        for name, module in [
            ("initializer", core.initializer),
            ("adapter", core.adapter),
            ("decoder", model.decoder),
        ]
    }
    if not all(math.isfinite(v) and v > 0 for v in norms.values()):
        raise ValueError(
            f"Observation gradients did not reach all trained components: {norms}"
        )
    if any(p.grad is not None for p in core.evolution.parameters()):
        raise ValueError("Frozen dynamics accumulated parameter gradients")
    torch.cuda.synchronize()
    report.update(
        observation_month=path.stem,
        observation_loss=float(loss.detach()),
        observation_forward_backward_seconds=time.monotonic() - started,
        observation_peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
        gradient_norms=norms,
        dynamics_frozen=True,
        mean_member_spread=float(
            initial.detach().float().std(0, unbiased=False).mean()
        ),
    )
    atomic_torch(
        dict(model=model.state_dict(), optimizer=optimizer.state_dict(), report=report),
        args.output / "fit.pt",
    )
    restored = torch.load(
        args.output / "fit.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(restored["model"], strict=True)
    optimizer.load_state_dict(restored["optimizer"])
    del restored
    with torch.no_grad():
        reloaded_loss = float(om4_loss())
    if not math.isclose(reloaded_loss, final_fit, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("Full A/B checkpoint reload changed the OM4 objective")
    report["strict_reload_om4_loss"] = reloaded_loss
    report["checkpoint_sha256"] = digest(args.output / "fit.pt")
    atomic_json(report, args.output / "QUALIFIED.json")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
