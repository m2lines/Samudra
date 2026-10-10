# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Small matched continuation wave: spatial losses and latent rollout stability."""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--arm", choices=tuple(dict.fromkeys((*ARMS, *V2_ARMS))), required=True
    )
    parser.add_argument("--campaign", choices=("v1", "v2"), default="v1")
    parser.add_argument("--updates", type=int, default=128)
    parser.add_argument("--hours", type=float, default=10)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cap = 256 if args.campaign == "v2" else 128
    if not 2 <= args.updates <= cap or args.updates % 2:
        raise ValueError(f"This campaign requires an even budget <= {cap}")
    if args.arm not in (V2_ARMS if args.campaign == "v2" else ARMS):
        raise ValueError("Arm does not belong to campaign")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    torch.manual_seed(271829)
    output = args.root / ("smoke" if args.smoke else "runs") / args.arm
    output.mkdir(parents=True, exist_ok=True)
    parent = args.root / "checkpoints/parent-16000.pt"
    if digest(parent) != PARENT_SHA:
        raise ValueError("Parent checkpoint differs")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if saved["step"] != 16000 or saved["counts"] != {"om4": 8000, "observation": 8000}:
        raise ValueError("Expected cooled 8k/8k parent")
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
        latent_bound_weight=0.1 if args.arm == "replay-bound" else 0.0,
        parent_sha256=PARENT_SHA,
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
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text()) != spec:
        raise ValueError("Changed continuation protocol")
    atomic_json(spec, protocol)
    model = make_model(data.grid["names"].tolist(), device)
    model.load_state_dict(saved["model"], strict=True)
    del saved
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
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        replay.load_state_dict(state["replay"])
        completed = state["step"] - 16000
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
            id=f"diffint-{args.campaign}-{args.arm}",
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

    def save(snapshot=False):
        payload = dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            contract=base_contract,
            intervention=spec,
            step=16000 + completed,
            counts={
                "om4": 8000 + (completed + 1) // 2,
                "observation": 8000 + completed // 2,
            },
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(device),
            replay=replay.state_dict(),
        )
        atomic_torch(payload, last)
        if snapshot:
            atomic_torch(
                {
                    k: payload[k]
                    for k in ("model", "contract", "intervention", "step", "counts")
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
                    for label, loss, measurements in observation_parts(
                        model, data, sample, seed, multiscale=spec["multiscale_weight"]
                    ):
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
                        "replay" in args.arm or args.arm == "pushforward"
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
                step=completed,
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
        ),
        output / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
