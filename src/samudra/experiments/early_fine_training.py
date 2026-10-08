# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Paired early coarse/fine training with persistent optional latent state."""

import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from samudra.experiments.early_fine_data import EarlySamples
from samudra.experiments.missingness import (
    completion_loss,
    state_mask,
    structured_visibility,
)
from samudra.experiments.observation_joint import JointPilot, om4_objective
from samudra.experiments.observation_pilot import atomic_json
from samudra.experiments.surface_state import advance_season, balanced_loss


@contextmanager
def deterministic_replay():
    """Use a repeatable numerical reference only for serialization diagnostics."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise RuntimeError(
            "Replay qualification needs a deterministic cuBLAS workspace at launch"
        )
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, cudnn = (
        torch.backends.cudnn.benchmark,
        torch.backends.cudnn.deterministic,
    )
    try:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn


def fine_objective(
    model,
    source,
    sample,
    global_data,
    coverage,
    seed,
    reconstruction_weight,
    completion_weight,
):
    coarse_mask = global_data.mask.bool()
    surface_ids = source.surface_ids
    coarse_context = torch.cat(
        (
            global_data.geo[None],
            sample["context"][:, 3:, :1, :1].expand(-1, -1, *coarse_mask.shape[-2:]),
        ),
        1,
    )
    coarse_surface = source.coarsen(sample["surface"], source.mask[surface_ids])
    coarse_past = source.coarsen(sample["past"], source.mask[surface_ids[0]][None])
    coarse_forcing = source.coarsen(
        sample["forcing"], source.mask[surface_ids[0]][None]
    )
    valid_coarse = source.coarsen.integrate(source.mask.float()) > 0
    mask = coarse_mask & valid_coarse
    available = mask[surface_ids].expand_as(coarse_surface)
    visible = structured_visibility(available & coverage.bool(), seed)
    fine_visible = (
        F.interpolate(visible.flatten(0, 1).float(), scale_factor=4, mode="nearest")
        .reshape_as(sample["surface"])
        .bool()
        & source.mask[surface_ids]
    )
    initial = model.call(
        model.initializer,
        torch.where(visible, coarse_surface, 0),
        coarse_past,
        coarse_context,
        mask,
        visible,
        "om4",
    )
    correction = model.call(
        model.initializer.fine_encoder,
        torch.where(fine_visible, sample["surface"], 0),
        sample["past"],
        fine_visible,
        sample["context"],
    )
    correction = correction.clone()
    correction[:, :, surface_ids] = 0
    initial = (initial + correction) * state_mask(mask, model.initializer.channels)
    truth = source.coarsen(sample["truth"], source.mask)
    labels = source.coarsen(sample["labels"], source.mask)
    weights = global_data.weights * mask
    states = initial
    losses = []
    for lead in range(1, 7):
        context = advance_season(coarse_context, (lead - 1) * 5)
        value = model.call(
            model.evolution, states, coarse_forcing, context, mask, lead, "om4"
        )
        fine = model.call(model.evolution.fine_decoder, value, context, source.mask)
        losses.append(
            0.5
            * balanced_loss(value[:, :77], labels[:, lead - 1], weights, source.names)
            + 0.5
            * balanced_loss(
                fine, sample["labels"][:, lead - 1], source.weights, source.names
            )
        )
        states = torch.stack((states[:, -1], value), 1)
    rec_coarse = balanced_loss(initial[:, :, :77], truth, weights, source.names, True)
    decoded = []
    for i in range(2):
        decoded.append(
            model.call(
                model.evolution.fine_decoder,
                initial[:, i],
                advance_season(coarse_context, (i - 1) * 5),
                source.mask,
            )
        )
    decoded = torch.stack(decoded, 1)
    rec_fine = balanced_loss(
        decoded, sample["truth"], source.weights, source.names, True
    )
    hidden = (
        source.mask[surface_ids].expand_as(fine_visible)[:, -2:] & ~fine_visible[:, -2:]
    )
    completion = completion_loss(
        decoded[:, :, surface_ids], sample["surface"][:, -2:], hidden, source.lat
    )
    return (
        torch.stack(losses).mean()
        + reconstruction_weight * 0.5 * (rec_coarse + rec_fine)
        + completion_weight * completion
    )


class EarlyPilot(JointPilot):
    def verify_resume_update(self):
        with deterministic_replay():
            super().verify_resume_update()
        differences = self.resume_evidence["native_and_serialized_replay"]
        if any(
            value != 0
            for difference in differences.values()
            for value in difference.values()
        ):
            raise ValueError("Deterministic reference replay was not bitwise exact")
        self.resume_evidence["deterministic_reference_bitwise_exact"] = True
        self.resume_evidence["reference_scope"] = (
            "Serialization diagnostic only; production uses its unchanged numerical backend"
        )

    def __init__(self, args):
        super().__init__(args)
        self.early = EarlySamples(
            args.early_data, self.om4, args.evolution_architecture == "extent-fine"
        )
        if args.joint_probe:
            seeds = [args.seed + 1100008, args.seed + 1100024]
            for seed in seeds:
                self.early.prefetch.plan([seed])
                prefetched = self.early.prefetch.take(seed)
                serial = self.early.read_cpu(seed)
                if prefetched[0] != serial[0]:
                    raise ValueError("Prefetch changed the sampled origin")
                for left, right in zip(prefetched[1], serial[1], strict=True):
                    np.testing.assert_array_equal(left, right)
                del prefetched, serial
            atomic_json(
                dict(
                    seeds=seeds,
                    cpu_arrays_exact=True,
                    producer=self.manifest["code_commit"],
                ),
                self.out / "IO_EQUIVALENT.json",
            )
        self.observation_warm_pool = ThreadPoolExecutor(max_workers=2)
        self.observation_warm = [
            self.observation_warm_pool.submit(self.data.read_sample, path)
            for path in [*self.training, *self.validation]
        ]
        self.timings = {k: [] for k in ["recent", "early", "observation"]}
        self.stop_requested = False
        self.interrupted = False
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda *_: setattr(self, "stop_requested", True))
        self.emit(
            dict(
                event="early_ready",
                origins=len(self.early.origins),
                fine=self.early.fine,
                parameters=sum(p.numel() for p in self.model.parameters()),
                latent_channels=args.latent_channels,
            )
        )

    def om4_objective(
        self,
        model,
        data,
        ids,
        reconstruction_weight,
        mask_seed=None,
        coverage=None,
        completion_weight=0.0,
    ):
        count = self.schedule.counts(self.completed)["om4"]
        if count % 2 == 0:
            return om4_objective(
                model,
                data,
                ids,
                reconstruction_weight,
                mask_seed,
                coverage,
                completion_weight,
            )
        if mask_seed is None or coverage is None:
            raise ValueError("Early tasks require reproducible masks and coverage")
        sample = self.early.sample(mask_seed)
        if self.early.fine:
            return fine_objective(
                model,
                self.early,
                sample,
                self.om4,
                coverage,
                mask_seed,
                reconstruction_weight,
                completion_weight,
            )
        view = SimpleNamespace(
            names=self.early.names,
            mask=self.early.mask,
            weights=self.early.weights,
            lat=self.early.lat,
            model_sample=lambda *_: tuple(
                sample[k]
                for k in ["surface", "past", "context", "truth", "forcing", "labels"]
            ),
            trainset=None,
        )
        return om4_objective(
            model,
            view,
            ids,
            reconstruction_weight,
            mask_seed,
            coverage,
            completion_weight,
        )

    def train_update(self):
        if self.stop_requested:
            self.checkpoint(self.resume)
            self.interrupted = True
            raise InterruptedError(
                "Checkpoint saved before preemption/time-limit requeue"
            )
        # CPU-only lookahead across the next two early updates. All examples
        # retain the exact original mask/sample seeds and consumption order.
        seeds = []
        for step in range(self.completed, self.schedule.total):
            count = self.schedule.counts(step)["om4"]
            if self.schedule.task(step) == "om4" and count % 2:
                seeds.extend(
                    self.args.seed + 1100000 + count * self.args.accumulate + micro
                    for micro in range(self.args.accumulate)
                )
                if len(seeds) >= 2 * self.args.accumulate:
                    break
        self.early.prefetch.plan(seeds)
        for future in self.observation_warm:
            if future.done():
                future.result()  # Do not swallow asynchronous read failures.
        task = self.schedule.task(self.completed)
        task = (
            ("early" if self.schedule.counts(self.completed)["om4"] % 2 else "recent")
            if task == "om4"
            else task
        )
        begin = time.monotonic()
        super().train_update()
        self.timings[task].append(time.monotonic() - begin)
        if self.completed % 10 == 0:
            self.checkpoint(self.resume)
        if self.early.fine:
            for module in [
                self.model.initializer.fine_encoder,
                self.model.evolution.fine_decoder,
            ]:
                active = any(
                    p.grad is not None and bool(p.grad.count_nonzero())
                    for p in module.parameters()
                )
                if active != (task == "early"):
                    raise ValueError("Fine I/O gradient routing differs from task")

    def emit(self, record):
        if record.get("event") == "joint_train":
            record = dict(record)
            record.update(
                om4_recent=(record["om4"] + 1) // 2, om4_early=record["om4"] // 2
            )
        super().emit(record)

    def run_joint(self):
        try:
            super().run_joint()
        except InterruptedError:
            if not self.interrupted:
                raise
            atomic_json(
                dict(global_step=self.completed, reason="preemption signal"),
                self.out / "PREEMPTED.json",
            )
            raise SystemExit(75)
        finally:
            self.early.prefetch.close()
            self.early.pool.shutdown(wait=True, cancel_futures=True)
            self.observation_warm_pool.shutdown(wait=True, cancel_futures=True)
        counts = self.schedule.counts(self.completed)
        atomic_json(
            dict(
                om4_recent=(counts["om4"] + 1) // 2,
                om4_early=counts["om4"] // 2,
                observation=counts["observation"],
            ),
            self.out / "EXPOSURE.json",
        )
        if self.args.joint_probe:
            if not (self.out / "JOINT_QUALIFIED.json").exists():
                raise RuntimeError("Incomplete probe")
            means = {k: sum(v) / len(v) for k, v in self.timings.items()}
            hours = (
                1000 * means["recent"]
                + 1000 * means["early"]
                + 2000 * means["observation"]
            ) / 3600
            atomic_json(
                dict(
                    mean_seconds=means,
                    training_hours_for_4000_updates=hours,
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30,
                    note="Includes real I/O, backward and optimizer; excludes validation and initial cache warmup",
                ),
                self.out / "THROUGHPUT.json",
            )
