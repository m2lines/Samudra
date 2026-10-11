#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Synthetic native-quarter U-global memory diagnostic; never saves weights.

Run each case in a separate process: om4/6, observation/6, observation/7.
The archived objective includes completion and reconstruction (including every
observation reconstruction initializer call), not merely an evolution forward.
OM4's archived objective has exactly six leads and is deliberately not extended.
This measures one-GPU microbatch-one execution, not effective-batch-eight DDP
throughput, loader performance, real-data fit, or scientific model skill. All
labels/coverage are synthetic on an all-wet regular grid of the required shape;
the future native-IAP observation operator is not exercised by this archive.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

SOURCE_COMMIT = "56aac9375a59a7204cd54d85a0c5a3b8e9383d59"  # pragma: allowlist secret -- public provenance hash
SOURCE_TREE_SHA256 = "2f4379de0bb64c672558203a0bd4638fb6c1cbbe0918b08611450c65c4195273"  # pragma: allowlist secret -- public provenance hash
MANIFEST_SHA256 = "36afc152f19a3f1e2d124b663aba68103eeff7d80069bce6492a1ea0e00b04c0"  # pragma: allowlist secret -- public provenance hash
CONTRACT = {
    "normalization": "instance",
    "evolution_architecture": "extent-unet",
    "initializer_architecture": "unet",
    "surface_policy": "observed-only",
    "surface_fill": "zero",
    "task_conditioning": "input-adapters",
    "latent_channels": 0,
    "completion_weight": 0.1,
    "reconstruction_weight": 0.1,
    "core_lr": 0.0001,
    "om4_lr": 0.0001,
    "warmup_steps": 0,
    "accumulate": 8,
    "seed": 1729,
    "from_scratch": True,
}


def verify_inputs(source, manifest_path):
    digest = hashlib.sha256()
    files = sorted((source / "src").rglob("*.py"))
    for path in files:
        digest.update(path.relative_to(source).as_posix().encode() + b"\0")
        digest.update(path.read_bytes() + b"\0")
    if digest.hexdigest() != SOURCE_TREE_SHA256:
        raise ValueError("Source Python tree differs from archived 56aac937")
    content = manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != MANIFEST_SHA256:
        raise ValueError(
            "Manifest differs from the completed six-thousand-update U-global"
        )
    manifest = json.loads(content)
    if any(manifest["arguments"].get(k) != v for k, v in CONTRACT.items()):
        raise ValueError("U-global architecture or objective contract differs")
    if any(name == "samudra" or name.startswith("samudra.") for name in sys.modules):
        raise RuntimeError("Archive must be selected before importing samudra")
    sys.path.insert(0, str(source / "src"))
    return manifest


def synthetic_objective(model, manifest, task, leads, device, *, shape=(720, 1440)):
    import numpy as np
    import torch

    from samudra.constants import build_om4_layout
    from samudra.experiments.observation_joint import om4_objective
    from samudra.experiments.observation_pilot import Pilot
    from samudra.experiments.observation_training import Samples
    from samudra.experiments.surface_state import advance_season, geographic_features

    height, width = shape  # CLI uses native quarter; small CPU contract tests only.
    names = tuple(build_om4_layout().prognostic_var_names)
    generator = torch.Generator(device=device).manual_seed(91273)

    def noise(*shape):
        return torch.randn(shape, generator=generator, device=device) * 0.1

    lat = torch.linspace(-89.875, 89.875, height, device=device)
    lon = torch.arange(width, device=device) * 0.25 + 0.125
    mask = torch.ones(77, height, width, dtype=torch.bool, device=device)
    geo = geographic_features(lat, lon)
    context = torch.cat(
        (
            geo,
            torch.zeros(1, height, width, device=device),
            torch.ones(1, height, width, device=device),
        )
    )[None]
    valid = (
        torch.rand(
            (1, 19 + leads, 2, height, width), generator=generator, device=device
        )
        > 0.1
    )
    surface = noise(1, 19 + leads, 2, height, width)
    surface = torch.where(valid, surface, 0)
    if task == "om4":
        past = noise(1, 19, 3, height, width)
        truth = noise(1, 2, 77, height, width)
        forcing = noise(1, 6, 3, height, width)
        labels = noise(1, 6, 77, height, width)
        om4_data = SimpleNamespace(
            names=names,
            mask=mask,
            weights=mask.float() * lat.deg2rad().cos()[None, :, None],
            lat=lat,
            trainset=None,
            model_sample=lambda *_: (
                surface[:, :19],
                past,
                context,
                truth,
                forcing,
                labels,
            ),
        )
        return lambda: om4_objective(
            model, om4_data, [0], 0.1, 1729, valid[:, :19], 0.1
        )

    # Construct only the archived objective's data interface. No NPZ, simulation
    # cache, checkpoint, selection metric, or experiment constructor is involved.
    data = Samples.__new__(Samples)
    data.device = device
    data.grid = dict(lat=lat.cpu().numpy(), mask=mask.cpu().numpy())
    data.mask = mask
    data.mean = torch.tensor(manifest["effective_mean"], device=device)[
        None, None, :, None, None
    ]
    data.std = torch.tensor(manifest["effective_std"], device=device)[
        None, None, :, None, None
    ]
    data.area = lat.deg2rad().cos()[:, None]
    data.ts_indices = list(range(38, 52)) + list(range(57, 71))
    data.ts_mask = mask[data.ts_indices].clone()
    data.ts_mask[0] = False
    data.ts_scale = data.std[0, :, data.ts_indices]
    data.surface_scale = data.std[:, :, [38, 76]]
    normalized_interior = noise(1, 28, height, width)
    physical_interior = (
        normalized_interior * data.std[0, :, data.ts_indices]
        + data.mean[0, :, data.ts_indices]
    )
    physical_surface = (
        noise(1, leads, 2, height, width) * data.surface_scale
        + data.mean[:, :, [38, 76]]
    )
    sample = dict(
        surface=surface,
        atmosphere=noise(1, 19 + leads, 8, height, width),
        contexts=torch.stack(
            [advance_season(context, (i - 18) * 5) for i in range(19 + leads)], 1
        ),
        validity=valid,
        month_weights=torch.full((leads,), 1 / leads, device=device),
        interior=physical_interior,
        raw_surface=physical_surface,
        mask_seed=1729,
    )
    pilot = Pilot.__new__(Pilot)
    pilot.model = model
    pilot.data = data
    pilot.args = SimpleNamespace(**manifest["arguments"])
    # Ensure the archive's NumPy latitude path is exercised by completion_loss.
    assert isinstance(data.grid["lat"], np.ndarray)
    return lambda: pilot.objective(sample, "joint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", choices=["om4", "observation"], required=True)
    parser.add_argument("--leads", type=int, choices=[6, 7], required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="CPU source/config/model validation only; no forward or fitting",
    )
    args = parser.parse_args()
    if args.task == "om4" and args.leads != 6:
        parser.error("The unchanged archived OM4 objective has exactly six leads")
    if not 2 <= args.updates <= 5:
        parser.error("Diagnostic updates must be between two and five")
    manifest = verify_inputs(args.source.resolve(), args.manifest)
    import torch

    from samudra.constants import build_om4_layout
    from samudra.experiments.observation_model import ObservationTransfer

    torch.set_num_threads(1)
    torch.manual_seed(CONTRACT["seed"])
    model = ObservationTransfer.from_arguments(
        build_om4_layout().prognostic_var_names, manifest["arguments"]
    )
    model.update_batchnorm = (
        True  # Exact from-scratch Pilot behavior; norm is InstanceNorm.
    )
    model.set_phase("joint")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 62_967_680 or not model.activation_checkpointing:
        raise ValueError("Unexpected U-global size or checkpointing policy")
    record = dict(
        diagnostic="synthetic fields; no data or science qualification; no weights saved",
        timing_scope="Forward, backward, gradient audit, clipping and optimizer; excludes data preparation and final-loss evaluation",
        diagnostic_script_sha256=hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        source_commit=SOURCE_COMMIT,
        source_tree_sha256=SOURCE_TREE_SHA256,
        manifest_sha256=MANIFEST_SHA256,
        training_producer=manifest["code_commit"],
        source_arguments={k: manifest["arguments"][k] for k in CONTRACT},
        task=args.task,
        leads=args.leads,
        shape=[720, 1440],
        microbatch=1,
        effective_batch=1,
        production_effective_batch=8,
        parameters=parameters,
        activation_checkpointing=True,
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        python=sys.version,
        machine=platform.machine(),
        slurm_job_id=os.getenv("SLURM_JOB_ID"),
        image=os.getenv("QUARTER_SMOKE_IMAGE"),
    )
    if args.validate_only:
        print(json.dumps(dict(record, status="cpu-contract-validated"), indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Native-quarter backward diagnostic requires CUDA")
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        raise RuntimeError(
            "This per-sample memory diagnostic is not a DDP qualification"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    record["steps"] = []

    def persist():
        (args.output / "result.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )

    def memory():
        return dict(
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            host_peak_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        )

    try:
        torch.cuda.set_device(0)
        record["gpu"] = torch.cuda.get_device_name(0)
        record["gpu_total_gib"] = (
            torch.cuda.get_device_properties(0).total_memory / 2**30
        )
        model = model.cuda()
        objective = synthetic_objective(
            model, manifest, args.task, args.leads, torch.device("cuda:0")
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        record["resident_before_backward_gib"] = torch.cuda.memory_allocated() / 2**30
        record["status"] = "running"
        persist()
        for step in range(args.updates):
            optimizer.zero_grad(set_to_none=True)
            begin = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = objective()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite diagnostic loss")
            loss.backward()
            reached = {}
            for name in ("initializer", "evolution", "adapter"):
                grads = [
                    p.grad
                    for p in getattr(model, name).parameters()
                    if p.grad is not None
                ]
                if any(not torch.isfinite(g).all() for g in grads):
                    raise FloatingPointError(f"Nonfinite {name} gradients")
                reached[name] = any(bool(g.count_nonzero()) for g in grads)
            required = {"initializer", "evolution"} | (
                {"adapter"} if args.task == "observation" else set()
            )
            if any(not reached[name] for name in required) or (
                args.task == "om4" and reached["adapter"]
            ):
                raise ValueError(f"Unexpected gradient routing: {reached}")
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            torch.cuda.synchronize()
            measured = dict(
                update=step + 1,
                loss=float(loss.detach()),
                gradient_norm=float(norm),
                gradients_finite=True,
                reached=reached,
                seconds=time.perf_counter() - begin,
                **memory(),
            )
            record["steps"].append(measured)
            persist()
            print(json.dumps(measured), flush=True)
            del loss
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            final_loss = objective()
        torch.cuda.synchronize()
        if not torch.isfinite(final_loss):
            raise FloatingPointError("Nonfinite final loss")
        record.update(
            status="completed",
            initial_loss=record["steps"][0]["loss"],
            final_loss=float(final_loss),
            measured_update_seconds=statistics.median(
                s["seconds"] for s in record["steps"][1:]
            ),
            total_seconds=time.perf_counter() - started,
            **memory(),
        )
        persist()
    except Exception as error:
        record.update(
            status="failed",
            error=repr(error),
            total_seconds=time.perf_counter() - started,
            **memory(),
        )
        persist()
        (args.output / "traceback.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
