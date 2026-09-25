# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded beta qualification, not production or a held-out comparison."""

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import xarray as xr
from torch.utils.checkpoint import checkpoint

from samudra.experiments.joint_diffusion import (
    JointInteriorDecoder,
    channel_balanced_mse,
    denoising_loss,
)
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import Pilot, atomic_json, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import geographic_features


def data_inputs(root, data):
    """One declared training origin; raw original OM4 forcing normalization."""
    source = xr.open_zarr(root / "data/om4_onedeg_v3/OM4.zarr", consolidated=True)
    end = 2800
    date = source.time.values[end]
    if date.year >= 2013:
        raise ValueError("Qualification origin must precede validation")
    names = list(data.grid["names"])
    times = slice(end - 18, end + 1)
    raw = np.stack(
        [source[n].isel(time=times).values for n in ("thetao_0", "zos")], axis=1
    )
    surface = (data.tensor(raw)[None] - data.mean[:, :, [38, 76]]) / data.std[
        :, :, [38, 76]
    ]
    surface = torch.nan_to_num(surface) * data.mask[[38, 76]]
    means = xr.open_zarr(root / "data/om4_onedeg_v3/OM4_means.zarr", consolidated=True)
    stds = xr.open_zarr(root / "data/om4_onedeg_v3/OM4_stds.zarr", consolidated=True)
    forcing = np.stack(
        [
            (source[n].isel(time=times).values - float(means[n].values))
            / float(stds[n].values)
            for n in ("tauuo", "tauvo", "hfds")
        ],
        axis=1,
    )
    past = data.tensor(np.nan_to_num(forcing))[None]
    phase = 2 * math.pi * (date.dayofyr - 1) / 365.25
    geo = geographic_features(
        data.tensor(data.grid["lat"]), data.tensor(data.grid["lon"])
    )
    season = data.tensor([math.sin(phase), math.cos(phase)])[:, None, None].expand(
        2, *geo.shape[-2:]
    )
    context = torch.cat((geo, season))[None]
    mask = data.mask[[38, 76]].expand(1, 19, 2, *geo.shape[-2:])
    inputs = torch.cat(
        (surface.flatten(1, 2), mask.flatten(1, 2), context, past.flatten(1, 2)), 1
    )
    target = np.stack(
        [source[n].isel(time=slice(end - 1, end + 1)).values for n in names], axis=1
    )
    target = (
        torch.nan_to_num((data.tensor(target)[None] - data.mean) / data.std) * data.mask
    )
    weights = (
        data.mask * torch.cos(torch.deg2rad(data.tensor(data.grid["lat"])))[:, None]
    )
    # Known initial SST/SSH are copied, not denoising targets.
    weights[[38, 76]] = 0
    return (
        inputs,
        target.flatten(1, 2),
        data.mask.repeat(2, 1, 1),
        weights.repeat(2, 1, 1),
        str(date),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--role",
        choices=["baseline", "deterministic", "diffusion", "geometry"],
        required=True,
    )
    args = parser.parse_args()
    root = args.root
    output = root / "runs/qualification" / args.role
    output.mkdir(parents=True, exist_ok=True)
    if (output / "QUALIFIED.json").exists():
        raise FileExistsError(
            "Qualification already completed; use a new attempt directory"
        )
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    chosen = json.loads((root / "checkpoints/selection.json").read_text())
    saved = root / "checkpoints" / chosen["selected"]
    manifest = json.loads((saved / "manifest.json").read_text())
    best = json.loads((saved / "best.json").read_text())
    if digest(saved / "best.pt") != best["checkpoint_sha256"]:
        raise ValueError("Baseline weights changed")
    data = Samples(root / "data/observations", "cuda")
    data.use_observation_normalization()
    for name, key in (
        ("grid.npz", "grid_sha256"),
        ("statistics.npz", "statistics_sha256"),
        ("SHA256SUMS", "data_manifest_sha256"),
    ):
        if digest(data.root / name) != manifest[key]:
            raise ValueError(f"Baseline data contract differs: {name}")
    if manifest["arguments"]["normalization"] != "instance":
        raise ValueError("Unqualified upstream architecture change")
    model = ObservationTransfer(list(data.grid["names"]), "instance").cuda()
    model.load_state_dict(
        torch.load(saved / "best.pt", map_location="cpu", weights_only=False)["model"],
        strict=True,
    )
    report = {
        "role": args.role,
        "baseline": chosen,
        "code_commit": os.environ["SAMUDRA_CODE_COMMIT"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "scope": "Implementation qualification only; no production or held-out skill claim",
    }
    start = time.monotonic()
    if args.role == "baseline":
        evaluator = Pilot.__new__(Pilot)
        evaluator.args = SimpleNamespace(strict_velocity_support=True)
        evaluator.model, evaluator.data = model, data
        result = evaluator.evaluate(data.paths("validation"))
        reference = json.loads((saved / "selection-reference.json").read_text())
        score = selection_score(
            result, reference["control"], reference["spectral_keys"]
        )
        # Same BF16 model on a different GPU family: measure and bound drift.
        if not math.isclose(score, best["score"], rel_tol=0.01, abs_tol=1e-4):
            raise ValueError(
                f"Beta score {score} differs from upstream {best['score']}"
            )
        report.update(
            score=score,
            upstream_score=best["score"],
            validation=result,
            tolerance="1% relative or 1e-4 absolute",
        )
    elif args.role == "geometry":
        # Cheap shape/gradient probe; not a physical half-degree result.
        net = JointInteriorDecoder(8, 4, width=8).cuda()
        latent = torch.randn(1, 8, 45, 90, device="cuda", requires_grad=True)
        mask = torch.ones(4, 360, 720, device="cuda")
        target = torch.randn(1, 4, 360, 720, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = denoising_loss(
                net,
                latent,
                target,
                mask,
                mask,
                torch.Generator(device="cuda").manual_seed(1),
            )
        loss.backward()
        if (
            latent.grad is None
            or not bool(torch.isfinite(latent.grad).all())
            or not bool(latent.grad.abs().sum() > 0)
        ):
            raise ValueError("Half-degree target loss does not reach fixed-grid latent")
        report.update(
            latent_grid=[45, 90],
            target_grid=[360, 720],
            synthetic_probe=True,
            production_geometry_ready=False,
        )
    else:
        inputs, target, mask, weights, date = data_inputs(root, data)
        initializer = model.initializer.net
        model.evolution.requires_grad_(False)
        model.adapter.requires_grad_(False)
        decoder = JointInteriorDecoder(154, 154).cuda()
        optimizer = torch.optim.AdamW(
            list(initializer.parameters()) + list(decoder.parameters()), lr=1e-4
        )

        # Fixed sigma/noise during tiny fitting makes a decreasing objective meaningful.
        def objective():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                latent = checkpoint(initializer, inputs, use_reentrant=False)
                if args.role == "diffusion":
                    return denoising_loss(
                        decoder,
                        latent,
                        target,
                        mask,
                        weights,
                        torch.Generator(device="cuda").manual_seed(91),
                    )
                prediction = decoder(
                    torch.zeros_like(target), torch.ones(1, device="cuda"), latent, mask
                )
                return channel_balanced_mse(prediction, target, weights).mean()

        losses = []
        first_gradient = None
        for step in range(10):
            optimizer.zero_grad(set_to_none=True)
            loss = objective()
            if not bool(torch.isfinite(loss)):
                raise ValueError("Nonfinite fit loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                list(initializer.parameters()) + list(decoder.parameters()),
                1.0,
                error_if_nonfinite=True,
            )
            gradient = sum(
                float(p.grad.abs().sum())
                for p in initializer.parameters()
                if p.grad is not None
            )
            if gradient <= 0 or not math.isfinite(gradient):
                raise ValueError("Initializer received no finite nonzero gradient")
            first_gradient = gradient if first_gradient is None else first_gradient
            optimizer.step()
            losses.append(float(loss.detach()))
            print(
                json.dumps(
                    {"step": step + 1, "loss": losses[-1], "gradient_norm": float(norm)}
                ),
                flush=True,
            )
        with torch.no_grad():
            final_loss = float(objective())
        if final_loss >= losses[0]:
            raise ValueError(
                "Ten-step fitting did not improve the fixed training-only objective"
            )
        checkpoint_path = output / "fit.pt"
        torch.save(
            {
                "initializer": initializer.state_dict(),
                "decoder": decoder.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        initializer.load_state_dict(restored["initializer"], strict=True)
        decoder.load_state_dict(restored["decoder"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        with torch.no_grad():
            reloaded_loss = float(objective())
        if not math.isclose(final_loss, reloaded_loss, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("Strict reload changed fitting loss")
        report.update(
            losses=losses,
            final_loss=final_loss,
            reloaded_loss=reloaded_loss,
            initializer_gradient=first_gradient,
            training_origin=date,
            input_sha256=hashlib.sha256(inputs.cpu().numpy().tobytes()).hexdigest(),
            fields=154,
            checkpoint_sha256=digest(checkpoint_path),
        )
    report.update(
        elapsed_seconds=time.monotonic() - start,
        peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
    )
    atomic_json(report, output / "QUALIFIED.json")


if __name__ == "__main__":
    main()
