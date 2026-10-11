# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Direct-next-latent continuation and short training-era transfer warmup."""

import json
import math
import time

import torch

from samudra.experiments.observation_pilot import atomic_json, atomic_torch
from samudra.experiments.surface_state import advance_season
from samudra.experiments.task_schedule import sample_indices


def warmup_indices(wave):
    """Separate fitting and assessment windows, all ending before 2013."""
    times = wave.source.time.values
    allowed = [
        index
        for index in range(len(wave.trainset))
        if int(str(times[index + 24])[:4]) < 2013
    ]
    if len(allowed) < 200:
        raise ValueError("Too few pre-2013 windows for guarded warmup split")
    # Each sample includes 19 historical states and six future states. Leave
    # 25 source intervals between the latest fit origin and assessment origins.
    assessment = allowed[-100::25]
    fitting = [index for index in allowed if index < assessment[0] - 25]
    if len(assessment) != 4 or not fitting:
        raise ValueError("Invalid warmup fit/assessment split")
    return fitting, assessment


@torch.no_grad()
def teacher_sample(model, teacher, wave, index):
    surface, past, context, _, forcing, _ = wave.model_sample(wave.trainset, [index])
    initial, _, _ = model.encode_native(surface, past, context, wave.mask, task="om4")
    contexts = torch.stack(
        [advance_season(context, (step + 1) * 5) for step in range(6)], 1
    )
    states = teacher.rollout(initial, forcing, contexts, task="om4")
    return states, forcing, contexts


def relative_error(predicted, target):
    """New-slot squared error divided by target energy, per example."""
    error = (predicted[:, -1].float() - target[:, -1].float()).square().mean()
    scale = target[:, -1].float().square().mean().clamp_min(1e-6)
    return error / scale


@torch.no_grad()
def assess_transfer(model, teacher, wave, indices):
    model.eval()
    records = []
    for index in indices:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states, forcing, contexts = teacher_sample(model, teacher, wave, index)
            rollout = states[0]
            one_step_errors, rollout_errors = [], []
            for step in range(6):
                direct = model.processor(
                    states[step], forcing[:, step], contexts[:, step], task="om4"
                )
                rollout = model.processor(
                    rollout, forcing[:, step], contexts[:, step], task="om4"
                )
                one_step_errors.append(float(relative_error(direct, states[step + 1])))
                rollout_errors.append(float(relative_error(rollout, states[step + 1])))
        records.append(
            dict(
                origin_index=index,
                origin=str(wave.source.time.values[index + 18]),
                teacher_forced_relative_mse=one_step_errors,
                autoregressive_relative_mse=rollout_errors,
            )
        )
    if not all(
        math.isfinite(value)
        for record in records
        for key in ("teacher_forced_relative_mse", "autoregressive_relative_mse")
        for value in record[key]
    ):
        raise FloatingPointError("Nonfinite processor warmup assessment")
    return dict(
        records=records,
        teacher_forced_relative_mse=sum(
            sum(record["teacher_forced_relative_mse"]) for record in records
        )
        / (6 * len(records)),
        day30_autoregressive_relative_mse=sum(
            record["autoregressive_relative_mse"][-1] for record in records
        )
        / len(records),
    )


def warmup_processor(model, teacher, wave, output, spec, run, stop_requested):
    """Fit next-latent mapping without decoder/initializer updates or obs exposure.

    Teacher and encoder remain frozen throughout this phase. Only processor
    parameters enter its optimizer. A separate checkpoint permits interruption;
    the normal continuation optimizer is fresh after this function returns.
    """
    fitting, assessment = warmup_indices(wave)
    total = spec["warmup_steps"]
    path = output / "warmup-last.pt"
    complete_path = output / "warmup-complete.json"
    split = dict(
        fitting_count=len(fitting),
        fitting_first=fitting[0],
        fitting_last=fitting[-1],
        assessment_indices=assessment,
        last_target_date=str(wave.source.time.values[assessment[-1] + 24]),
        temporal_guard_source_intervals=25,
    )
    model.eval()
    teacher.eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.processor.parameters(), lr=1e-3, weight_decay=0.0
    )
    completed, history = 0, []
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["protocol"] != spec or state["split"] != split:
            raise ValueError("Warmup resume protocol or data split changed")
        model.processor.load_state_dict(state["processor"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        completed, history = state["completed"], state["history"]
        before = state["before"]
        torch.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], wave.device)
        if not 0 <= completed <= total:
            raise ValueError("Warmup checkpoint exceeds requested budget")
        if completed == total and complete_path.exists():
            return json.loads(complete_path.read_text())
    else:
        before = assess_transfer(model, teacher, wave, assessment)
        atomic_json(dict(split=split, metrics=before), output / "warmup-before.json")
    print(
        json.dumps(dict(event="warmup_assessment_before", metrics=before)), flush=True
    )
    started = time.monotonic()

    def save():
        atomic_torch(
            dict(
                protocol=spec,
                split=split,
                processor=model.processor.state_dict(),
                optimizer=optimizer.state_dict(),
                completed=completed,
                history=history,
                before=before,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(wave.device),
            ),
            path,
        )

    while completed < total and not stop_requested():
        # Each update draws one distinct-in-stream origin, then one of its six
        # teacher transitions; no target interior values are passed to the model.
        index = fitting[sample_indices(len(fitting), 671829, completed, 1)[0]]
        step = completed % 6
        lr = (
            1e-5
            + (1e-3 - 1e-5)
            * (1 + math.cos(math.pi * completed / max(1, total - 1)))
            / 2
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            states, forcing, contexts = teacher_sample(model, teacher, wave, index)
        model.processor.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model.processor(
                states[step], forcing[:, step], contexts[:, step], task="om4"
            )
            loss = relative_error(prediction, states[step + 1])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite processor warmup loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            model.processor.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        completed += 1
        event = dict(
            event="warmup_update",
            update=completed,
            origin_index=index,
            lead_days=(step + 1) * 5,
            relative_mse=float(loss.detach()),
            gradient_norm=float(norm),
            lr=lr,
        )
        history.append(event)
        print(json.dumps(event), flush=True)
        with (output / "warmup-events.jsonl").open("a") as stream:
            stream.write(json.dumps(event) + "\n")
        if run:
            run.log(
                {
                    "warmup/relative_mse": event["relative_mse"],
                    "warmup/lr": lr,
                    "warmup/gradient_norm": event["gradient_norm"],
                },
                step=completed,
            )
        if completed % 16 == 0 or completed == total or stop_requested():
            save()
    if completed != total:
        save()
        raise RuntimeError(f"Processor warmup paused at {completed}/{total}")
    after = assess_transfer(model, teacher, wave, assessment)
    summary = dict(
        complete=True,
        updates=completed,
        batch_size=1,
        additional_om4_origin_samples=completed,
        observation_updates=0,
        physical_target_supervision=False,
        seconds_this_invocation=time.monotonic() - started,
        split=split,
        before=before,
        after=after,
        interpretation="Transfer diagnostic only; poor teacher agreement makes negative direct-continuation results inconclusive",
    )
    atomic_json(summary, complete_path)
    print(json.dumps(dict(event="warmup_complete", summary=summary)), flush=True)
    if run:
        run.summary["warmup"] = summary
    return summary


if __name__ == "__main__":
    from samudra.experiments.diffusion_interventions import main

    main(campaign_default="direct")
