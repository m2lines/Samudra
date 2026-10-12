# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Small matched continuation wave: spatial losses and latent rollout stability."""

import argparse
import copy
import gc
import json
import math
import os
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from samudra.experiments.diffusion_correlated_pilot import build_wave
from samudra.experiments.diffusion_global import (
    make_model,
    native_objective,
    observation_parts,
    rng,
)
from samudra.experiments.joint_diffusion import denoising_loss
from samudra.experiments.missingness import structured_visibility
from samudra.experiments.observation_pilot import atomic_json, atomic_torch, digest
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_state import advance_season
from samudra.experiments.task_schedule import sample_indices

ARMS = (
    "control",
    "multiscale",
    "replay",
    "multiscale-replay",
    "unroll12",
    "latent-jitter",
)
V2_ARMS = (
    "control",
    "replay",
    "replay-bound",
    "pushforward",
    "curvature",
    "block",
    "curvature-replay",
    "block-replay",
)
OBS_FAST_ARMS = ("control", "fast", "hybrid")
DIRECT_ARMS = ("residual", "direct", "residual-warm", "direct-warm")
DIRECT_PARENT_SHA = "36e99d3c669daa3be0f2120ae1ecb95042dcbb82f66022f2d2b082143f7ba38b"  # pragma: allowlist secret
TIMESTEP_ARMS = ("control", "supervised", "consistency-weak", "consistency-strong")
PARENT_SHA = "853b3353c6d926754a15f19437493a7497813ce2b88cb058fe1a616cc838e6e5"  # pragma: allowlist secret


class LatentReplay:
    """Four on-policy chains. No physical predictions or target interiors are inputs.

    A fresh chain burns in 6/18/36/60 steps without gradients. Each use advances
    six differentiable steps against the corresponding OM4 targets, then saves
    the detached endpoint. Reinitialize after four uses or at the year boundary;
    each stored state is at most two optimizer updates old when reused. The entire
    possible 360-day trajectory must lie within the original training source.
    """

    def __init__(self, wave, fresh_each=False, bound=False):
        self.wave = wave
        self.fresh_each = fresh_each
        self.bound = bound
        self.annual = wave.dataset(wave.source, steps=72)
        self.slots = {}
        self.last_info = {}

    def loss(self, model, slot, count, coverage, seed, multiscale=0.0):
        wave = self.wave
        entry = self.slots.get(slot)
        fresh = (
            self.fresh_each
            or entry is None
            or entry["uses"] >= 4
            or entry["lead"] + 6 > 72
        )
        if fresh:
            index = sample_indices(len(self.annual), 991729, count, 8)[slot]
            lead = (6, 18, 36, 60)[slot % 4]
            dataset = wave.dataset(wave.source, steps=lead)
            with torch.no_grad():
                surface, past, context, _, forcing, _ = wave.model_sample(
                    dataset, [index]
                )
                visible = structured_visibility(
                    wave.mask[model.surface].bool().expand_as(surface)
                    & coverage.bool(),
                    seed,
                )
                latent, _, _ = model.encode_native(
                    torch.where(visible, surface, 0),
                    past,
                    context,
                    wave.mask,
                    visible,
                    task="om4",
                )
                reference = latent.detach()
                for step in range(lead):
                    latent = model.processor(
                        latent,
                        forcing[:, step],
                        advance_season(context, (step + 1) * 5),
                        task="om4",
                    )
                latent = latent.detach()
            entry = dict(
                index=index, lead=lead, latent=latent, uses=0, reference=reference
            )
        assert entry is not None
        index, lead = entry["index"], entry["lead"]
        # At origin + lead, this window's truth is [lead-1, lead]; labels begin
        # at lead+1. Only forcing, context and targets are used from this window.
        _, _, context, truth, forcing, targets = wave.model_sample(
            wave.trainset, [index + lead]
        )
        contexts = torch.stack(
            [advance_season(context, (i + 1) * 5) for i in range(6)], 1
        )
        states = model.processor.rollout(
            entry["latent"].detach(), forcing, contexts, task="om4"
        )
        sequence = torch.cat((truth, targets), 1)
        generator = rng(wave.device, seed + 4000000)
        losses = []
        for step in range(1, 7):
            losses.append(
                denoising_loss(
                    model.decoder,
                    states[step].flatten(1, 2),
                    sequence[:, step : step + 2].flatten(1, 2),
                    wave.mask.repeat(2, 1, 1),
                    wave.weights.repeat(2, 1, 1),
                    generator,
                    checkpoint_denoiser=model.training,
                    spatial_weight=0.5,
                    multiscale_weight=multiscale,
                )
            )
        self.slots[slot] = dict(
            index=index,
            lead=lead + 6,
            latent=states[-1].detach(),
            uses=entry["uses"] + 1,
            reference=entry.get("reference", entry["latent"]),
        )
        self.last_info = dict(
            origin_index=index,
            lead_start_days=lead * 5,
            lead_end_days=(lead + 6) * 5,
            refreshed=fresh,
        )
        result = torch.stack(losses).mean()
        if self.bound:
            from samudra.experiments.diffusion_structure_losses import (
                excess_latent_loss,
            )

            penalty = excess_latent_loss(states[1:], entry["reference"])
            self.last_info["bound_penalty"] = float(penalty.detach())
            result = result + 0.1 * penalty
        return result

    def state_dict(self):
        return {
            k: {
                name: value.cpu() if torch.is_tensor(value) else value
                for name, value in v.items()
            }
            for k, v in self.slots.items()
        }

    def load_state_dict(self, state):
        self.slots = {
            k: {
                name: value.to(self.wave.device) if torch.is_tensor(value) else value
                for name, value in v.items()
            }
            for k, v in state.items()
        }


def observation_update_is_fast(arm, count):
    """Use one mode for all eight microbatches; counts are zero-based obs updates."""
    if arm not in OBS_FAST_ARMS or count < 0:
        raise ValueError("Unknown observation arm or negative update count")
    return arm == "fast" or (arm == "hybrid" and count % 4 != 3)


def main(campaign_default="v1"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=tuple(
            dict.fromkeys(
                (*ARMS, *V2_ARMS, *TIMESTEP_ARMS, *DIRECT_ARMS, *OBS_FAST_ARMS)
            )
        ),
        required=True,
    )
    parser.add_argument(
        "--campaign",
        choices=("v1", "v2", "timestep", "direct", "obs-fast"),
        default=campaign_default,
    )
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cap = 256 if args.campaign == "v2" else 128
    if not 2 <= args.updates <= cap or args.updates % 2:
        raise ValueError(f"This campaign requires an even budget <= {cap}")
    if (
        args.arm
        not in {
            "v1": ARMS,
            "v2": V2_ARMS,
            "timestep": TIMESTEP_ARMS,
            "direct": DIRECT_ARMS,
            "obs-fast": OBS_FAST_ARMS,
        }[args.campaign]
    ):
        raise ValueError("Arm does not belong to campaign")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(271829)
    output = args.root / ("smoke" if args.smoke else "runs") / args.arm
    output.mkdir(parents=True, exist_ok=True)
    direct_campaign = args.campaign == "direct"
    fast_campaign = args.campaign == "obs-fast"
    new_parent = direct_campaign or fast_campaign
    parent_step = 16256 if new_parent else 16000
    parent_count = 8128 if new_parent else 8000
    parent_sha = DIRECT_PARENT_SHA if new_parent else PARENT_SHA
    parent = args.root / f"checkpoints/parent-{parent_step}.pt"
    if digest(parent) != parent_sha:
        raise ValueError("Parent checkpoint differs")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if saved["step"] != parent_step or saved["counts"] != {
        "om4": parent_count,
        "observation": parent_count,
    }:
        raise ValueError("Parent step or task counts differ")
    data = Samples(
        args.root / "data/observations",
        device,
        surface_fill="zero",
        global_observations=True,
    )
    data.use_observation_normalization()
    for key, path in (
        ("observation_manifest", data.root / "SHA256SUMS"),
        ("stats", data.root / "statistics.npz"),
        ("grid", data.root / "grid.npz"),
        ("reference", args.root / "checkpoints/selection-reference.json"),
    ):
        if digest(path) != saved["contract"][key]:
            raise ValueError(f"Changed input: {key}")
    training = data.paths("train")
    if len(training) != 243 or not bool((data.area > 0).all()):
        raise ValueError("Changed cohort or latitude mask")
    base_contract = saved["contract"]
    spec = dict(
        version=2 if args.campaign == "v2" else 1,
        arm=args.arm,
        campaign=args.campaign,
        structure_aux=next(
            (kind for kind in ("curvature", "block") if kind in args.arm), ""
        ),
        structure_weight=2.0,
        fresh_pushforward=args.arm == "pushforward",
        latent_bound_weight=0.1 if new_parent or args.arm == "replay-bound" else 0.0,
        parent_sha256=parent_sha,
        producer=os.environ["SAMUDRA_CODE_COMMIT"],
        seed=271829,
        updates=args.updates,
        effective_batch=8,
        alternating_tasks=f"OM4 then observation; {args.updates // 2} each at full budget",
        optimizer="fresh AdamW for all arms",
        learning_rate="cosine 1e-5 to 1e-6",
        clip=1.0,
        multiscale_weight=0.5 if "multiscale" in args.arm else 0.0,
        spatial="existing nearest-neighbor term retained; optional four directions at lags 1/2/4/8 divided by grid distance",
        stability="last four OM4 microbatches use intervention; first four retain original objective",
        replay="four detached on-policy latent chains; burn-in 30/90/180/300d; six steps per use; reset after four uses or 360d",
        jitter=0.05 if args.arm == "latent-jitter" else 0.0,
        long_unroll="12 differentiable steps, decode last six plus initial; original six-step cohort preserved in first four microbatches",
        global_observations=True,
        sampling_steps=32,
        members_train=2,
        compile_decoder=True,
        parent_contract=base_contract,
    )
    if args.campaign == "v1":
        for key in (
            "campaign",
            "structure_aux",
            "structure_weight",
            "fresh_pushforward",
            "latent_bound_weight",
        ):
            spec.pop(key)
    if args.campaign == "timestep":
        spec.update(
            version=3,
            timestep_physical_weight=0.0 if args.arm == "control" else 0.5,
            timestep_consistency_weight={
                "control": 0.0,
                "supervised": 0.0,
                "consistency-weak": 0.1,
                "consistency-strong": 1.0,
            }[args.arm],
            stability="Five-day inference; auxiliary ten-day endpoint on all OM4 microbatches; no replay",
            timestep="Shared processor; ordered two-interval forcing/context adapter; endpoint-only consistency; decoder uses short-path day-five scaffold",
        )
    if direct_campaign:
        spec.update(
            version=4,
            processor_prediction="direct"
            if args.arm.startswith("direct")
            else "residual",
            parent_step=parent_step,
            parent_counts={"om4": parent_count, "observation": parent_count},
            # A truncated conversion can destabilize recurrence even if its
            # one-step fit improves. Qualify the actual production warmup.
            warmup_steps=256 if args.arm.endswith("-warm") else 0,
            warmup_learning_rate="cosine 1e-3 to 1e-5",
            warmup_weight_decay=0.0,
            warmup_batch=1,
            warmup_target="Frozen parent next latent on short pre-2013 OM4 trajectories; processor only; separate exposure counts",
            warmup_assessment="Four reserved pre-2013 origins excluded with temporal guard from warmup fitting; teacher-forced and 30-day autoregressive agreement",
            continuation_optimizer="Fresh AdamW after warmup; no warmup optimizer state carried forward",
            stability="All arms retain replay and soft latent bound on last four OM4 microbatches",
        )
    cache_manifest = None
    if fast_campaign:
        from samudra.experiments.diffusion_observation_fast import (
            observation_parts_fast,
        )

        cache_root = args.root / (
            "cache-observation-smoke" if args.smoke else "cache-observation"
        )
        if args.arm != "control":
            cache_manifest = json.loads((cache_root / "COMPLETE.json").read_text())
            if (
                cache_manifest["parent_sha256"] != parent_sha
                or cache_manifest["training_contract"] != base_contract
            ):
                raise ValueError("Observation imputation cache parent/data mismatch")
            if not cache_manifest.get("complete"):
                raise ValueError("Observation imputation cache is incomplete")
            if (
                cache_manifest["updates"] != args.updates
                or cache_manifest["version"] != 1
            ):
                raise ValueError("Observation imputation cache budget/version mismatch")
        spec.update(
            version=5,
            processor_prediction="residual",
            parent_step=parent_step,
            parent_counts={"om4": parent_count, "observation": parent_count},
            observation_objective={
                "control": "full 32-step sampled CRPS",
                "fast": "single denoiser clean-target projection",
                "hybrid": "three fast observation updates then one full CRPS",
            }[args.arm],
            imputation_cache_manifest_sha256=digest(cache_root / "COMPLETE.json")
            if cache_manifest
            else None,
            imputation_storage="float16 normalized physical states; converted to float32 by objective",
            observation_training_readouts="Full updates use two sampled members; fast updates use a single denoiser call per state and no generated members",
            members_train=0 if args.arm == "fast" else 2,
            fast_readouts_per_state=1,
            full_crps_members=2,
            stability="All arms retain replay and soft latent bound on last four OM4 microbatches",
        )
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Changed continuation protocol")
    atomic_json(spec, protocol)
    model = make_model(data.grid["names"].tolist(), device)
    model.load_state_dict(saved["model"], strict=True)
    teacher = (
        copy.deepcopy(model.processor).eval().requires_grad_(False)
        if direct_campaign and spec["warmup_steps"]
        else None
    )
    del saved
    if direct_campaign and spec["processor_prediction"] == "direct":
        from samudra.experiments.diffusion_direct import install_direct_processor

        install_direct_processor(model)
    if args.campaign == "timestep":
        from samudra.experiments.diffusion_timestep import install_timestep_processor

        install_timestep_processor(model)
    model.decoder.structure_aux = spec.get("structure_aux", "")
    model.decoder.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    loader = SimpleNamespace(root=args.root, output=output, export_only=True)
    wave = build_wave(loader, data)
    del wave.model
    del wave.initializer
    gc.collect()
    # wave.sample needs initializer.surface even though the initializer itself
    # is not used; avoid retaining that unused network on the GPU.
    wave.initializer = SimpleNamespace(surface=model.surface)
    wave.prepare(wave.trainset)
    if wave.run:
        wave.run.finish()
        wave.run = None
    longer = wave.dataset(wave.source, steps=12)
    replay = LatentReplay(
        wave,
        fresh_each=spec.get("fresh_pushforward", False),
        bound=bool(spec.get("latent_bound_weight", 0.0)),
    )
    completed = 0
    last = output / "last.pt"
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["intervention"] != spec:
            raise ValueError("Resume protocol differs")
        resume_model = dict(state["model"])
        if args.campaign == "timestep":
            resume_model.update(
                {
                    "processor.timestep_adapter." + key: value
                    for key, value in state["timestep_adapter"].items()
                }
            )
        if direct_campaign:
            from samudra.experiments.diffusion_direct import load_latent_checkpoint

            load_latent_checkpoint(model, state)
        else:
            model.load_state_dict(resume_model, strict=True)
        optimizer.load_state_dict(state["optimizer"])
        replay.load_state_dict(state["replay"])
        completed = state["step"] - parent_step
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], device)
        del state
    run = None
    if not args.smoke:
        import wandb

        run = wandb.init(
            project="default",
            entity="ocean_emulators",
            group=f"diffusion-interventions-{args.campaign}",
            name=args.arm,
            id=(
                f"diffint-{args.campaign}-{args.root.name}-{args.arm}"
                if fast_campaign
                else f"diffint-{args.campaign}-{args.arm}"
            ),
            resume="allow",
            dir=str(output),
            config=spec,
            mode="online",
        )
        if run.disabled or not run.url:
            raise RuntimeError("Online W&B initialization failed")
        print(json.dumps(dict(event="wandb", url=run.url)), flush=True)
    stop = False

    def stop_requested(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stop_requested)
    signal.signal(signal.SIGUSR1, stop_requested)

    warmup_summary = None
    if direct_campaign and teacher is not None:
        from samudra.experiments.diffusion_direct_train import warmup_processor

        if last.exists():
            warmup_summary = json.loads((output / "warmup-complete.json").read_text())
        else:
            warmup_summary = warmup_processor(
                model, teacher, wave, output, spec, run, lambda: stop
            )
        del teacher
        gc.collect()
        # Warmup uses only processor parameters and a separate optimizer.
        # This already-created continuation optimizer has no parameter history.
        if not last.exists() and optimizer.state:
            raise RuntimeError("Continuation optimizer unexpectedly has warmup state")

    def save(snapshot=False):
        model_state = model.state_dict()
        auxiliary_state = {}
        if args.campaign == "timestep":
            model_state = {
                key: value
                for key, value in model_state.items()
                if not key.startswith("processor.timestep_adapter.")
            }
            auxiliary_state["timestep_adapter"] = (
                model.processor.timestep_adapter.state_dict()
            )
        if new_parent:
            auxiliary_state["processor_prediction"] = spec["processor_prediction"]
        if direct_campaign:
            auxiliary_state["warmup"] = warmup_summary
        payload = dict(
            model=model_state,
            optimizer=optimizer.state_dict(),
            contract=base_contract,
            intervention=spec,
            step=parent_step + completed,
            counts={
                "om4": parent_count + (completed + 1) // 2,
                "observation": parent_count + completed // 2,
            },
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device),
            replay=replay.state_dict(),
            **auxiliary_state,
        )
        atomic_torch(payload, last)
        if snapshot:
            atomic_torch(
                {
                    k: payload[k]
                    for k in (
                        "model",
                        "contract",
                        "intervention",
                        "step",
                        "counts",
                        *auxiliary_state,
                    )
                },
                output / f"step-{completed:04d}.pt",
            )
        atomic_json(
            dict(
                completed=completed,
                complete=completed == args.updates,
                counts=payload["counts"],
            ),
            output / "state.json",
        )

    verified_cache_months = set()
    started = time.monotonic()
    while completed < args.updates and not stop:
        task = "om4" if completed % 2 == 0 else "observation"
        count = completed // 2
        lr = 1e-6 + 9e-6 * (1 + math.cos(math.pi * completed / (args.updates - 1))) / 2
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        model.train()
        ids = sample_indices(
            len(training) if task == "observation" else len(wave.trainset),
            271829 if task == "observation" else 371829,
            count,
            8,
        )
        total = 0.0
        parts: dict[str, float] = {}
        replay_info = []
        update_started = time.monotonic()
        for micro, index in enumerate(ids):
            seed = 2271829 + (100000 if task == "om4" else 0) + count * 8 + micro
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "observation":
                    sample = data.load(training[index])
                    fast_update = fast_campaign and observation_update_is_fast(
                        args.arm, count
                    )
                    if fast_update:
                        assert cache_manifest is not None
                        cache_file = cache_root / f"{sample['name']}.pt"
                        if sample["name"] not in verified_cache_months:
                            if (
                                digest(training[index])
                                != cache_manifest["source_files"][training[index].name]
                            ):
                                raise ValueError("Changed imputation training source")
                            if (
                                digest(cache_file)
                                != cache_manifest["files"][cache_file.name]
                            ):
                                raise ValueError("Changed observation imputation cache")
                            verified_cache_months.add(sample["name"])
                        imputation = torch.load(
                            cache_file, map_location="cpu", weights_only=True
                        )
                        objectives = observation_parts_fast(
                            model,
                            data,
                            sample,
                            seed,
                            imputation,
                            multiscale=spec["multiscale_weight"],
                        )
                    else:
                        objectives = observation_parts(
                            model,
                            data,
                            sample,
                            seed,
                            multiscale=spec["multiscale_weight"],
                        )
                    for label, loss, measurements in objectives:
                        if not bool(torch.isfinite(loss)):
                            raise FloatingPointError(label)
                        (loss / 8).backward()
                        total += float(loss.detach()) / 8
                        for key, value in measurements.items():
                            parts[key] = parts.get(key, 0.0) + value / 8
                else:
                    coverage = data.load(training[(count * 8 + micro) % len(training)])[
                        "validity"
                    ][:, :19]
                    if micro >= 4 and (
                        new_parent or "replay" in args.arm or args.arm == "pushforward"
                    ):
                        loss = replay.loss(
                            model,
                            micro - 4,
                            count,
                            coverage,
                            seed,
                            spec["multiscale_weight"],
                        )
                        replay_info.append(replay.last_info)
                    else:
                        long = micro >= 4 and args.arm == "unroll12"
                        if long:
                            index = sample_indices(len(longer), 371829, count, 8)[micro]
                        loss = native_objective(
                            model,
                            wave,
                            index,
                            coverage,
                            seed,
                            multiscale=spec["multiscale_weight"],
                            jitter=spec["jitter"] if micro >= 4 else 0.0,
                            horizon=12 if long else 6,
                            dataset=longer if long else None,
                        )
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("native")
                    (loss / 8).backward()
                    total += float(loss.detach()) / 8
                    if args.campaign == "timestep" and args.arm != "control":
                        from samudra.experiments.diffusion_timestep import (
                            auxiliary_loss,
                        )

                        auxiliary, measurements = auxiliary_loss(
                            model,
                            wave,
                            index,
                            coverage,
                            seed,
                            physical_weight=spec["timestep_physical_weight"],
                            consistency_weight=spec["timestep_consistency_weight"],
                        )
                        if not bool(torch.isfinite(auxiliary)):
                            raise FloatingPointError("timestep auxiliary")
                        (auxiliary / 8).backward()
                        total += float(auxiliary.detach()) / 8
                        for key, value in measurements.items():
                            parts[key] = parts.get(key, 0.0) + float(value) / 8
        if fast_campaign and args.smoke and task == "observation":
            for label, module in (
                ("initializer", model.encoder),
                ("processor", model.processor),
                ("decoder", model.decoder),
            ):
                grads = [
                    parameter.grad.detach().float().square().sum()
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                magnitude = (
                    torch.stack(grads).sum().sqrt()
                    if grads
                    else torch.tensor(0.0, device=device)
                )
                if not bool(torch.isfinite(magnitude)) or not bool(magnitude > 0):
                    raise FloatingPointError(
                        f"Missing/nonfinite {label} observation gradient"
                    )
                parts[f"gradient_norm/{label}"] = float(magnitude)
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        completed += 1
        torch.cuda.synchronize()
        event = dict(
            event="update",
            arm=args.arm,
            update=completed,
            task=task,
            **(
                {
                    "observation_mode": "fast"
                    if observation_update_is_fast(args.arm, count)
                    else "full"
                }
                if fast_campaign and task == "observation"
                else {}
            ),
            loss=total,
            parts=parts,
            seconds=time.monotonic() - update_started,
            gradient_norm=float(norm),
            lr=lr,
            peak_gib=torch.cuda.max_memory_allocated() / 2**30,
            replay=replay_info,
        )
        print(json.dumps(event), flush=True)
        with (output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if run:
            run.log(
                {
                    "loss/" + task: total,
                    "gradient_norm": float(norm),
                    "lr": lr,
                    "seconds": event["seconds"],
                    **{task + "/" + k: v for k, v in parts.items()},
                },
                step=completed + spec.get("warmup_steps", 0),
            )
        due = time.monotonic() - started > args.hours * 3600 or stop
        if (
            completed % 16 == 0
            or completed in (32, 64, 128, 256)
            or completed == args.updates
            or due
        ):
            save(snapshot=completed in (32, 64, 128, 256) or completed == args.updates)
        if due:
            break
    if run:
        run.finish()
    if completed != args.updates:
        raise RuntimeError(
            f"Pilot paused at {completed}/{args.updates}; resume from last.pt"
        )
    atomic_json(
        dict(
            complete=True,
            updates=completed,
            checkpoint_sha256=digest(last),
            seconds=time.monotonic() - started,
            **(
                {
                    "warmup": warmup_summary,
                    "processor_prediction": spec["processor_prediction"],
                    "counts": {
                        "om4": parent_count + completed // 2,
                        "observation": parent_count + completed // 2,
                    },
                }
                if new_parent
                else {}
            ),
        ),
        output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
