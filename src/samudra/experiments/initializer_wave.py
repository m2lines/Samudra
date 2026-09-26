# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Resumable capacity/history study with fixed dynamics and paired-origin metrics."""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from samudra.config import DataConfig
from samudra.datasets import TorchTrainDataset
from samudra.experiments.initializer_diagnostics import ReconstructionDiagnostics
from samudra.experiments.initializer_models import HistoryInitializer
from samudra.experiments.normalization import configure_normalization
from samudra.experiments.observation_training import Samples
from samudra.experiments.surface_adaptation import (
    AdaptationState,
    state_fingerprint,
    thermohaline_mse,
)
from samudra.experiments.surface_state import (
    Evolution,
    advance_season,
    balanced_loss,
    channel_mse,
)
from samudra.experiments.surface_wave import (
    Experiment,
    atomic_save,
    batches,
    data_config,
    training_batches,
)
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation

ARMS = {
    "A": ("unet", False),
    "B": ("unet", True),
    "C": ("wide", False),
    "D": ("wide", True),
    "E": ("swin", False),
    "F": ("swin", True),
}


class Pair(torch.nn.Module):
    def __init__(self, initializer, evolution, joint=False):
        super().__init__()
        self.initializer, self.evolution, self.joint = initializer, evolution, joint
        self.evolution.requires_grad_(joint)

    def train(self, mode=True):
        super().train(mode)
        if not self.joint:
            self.evolution.eval()
        return self

    def evolve(self, states, forcing, context, mask):
        predictions = []
        for lead in range(1, 7):
            prediction = self.evolution(
                states, forcing, advance_season(context, (lead - 1) * 5), mask, lead
            )
            predictions.append(prediction)
            states = torch.stack((states[:, -1], prediction), 1)
        return torch.stack(predictions, 1)

    def forward(self, surface, past, context, mask, forcing):
        initial = self.initializer(surface, past, context, mask)
        prediction = (
            self.evolve(initial, forcing, context, mask) if self.joint else initial
        )
        return prediction, initial


class InitializerWave(Experiment):
    def build_data_config(self, args):
        config = data_config(args).model_dump(mode="json")
        config["input_steps"] = 19
        return DataConfig.model_validate(config)

    def context_config(self):
        # This second bundle only supplies read-only validation/test context.
        # Its unused training range is shortened to satisfy split validation;
        # the actual training source remains the original full training period.
        config = self.config.model_dump(mode="json")
        config["sources"][0]["train_time"]["end"] = "2013-07-31"
        config["sources"][0]["val_time"]["start"] = "2013-08-01"
        config["sources"][0]["inference_times"][0]["start"] = "2014-08-06"
        return DataConfig.model_validate(config)

    def build_initializer(self, args):
        architecture, expanded = ARMS[args.arm]
        return configure_normalization(
            HistoryInitializer(self.names, architecture, expanded),
            getattr(args, "normalization", "batch"),
        )

    def dataset(self, source, steps=6):
        return TorchTrainDataset(
            input_source=source,
            label_source=None,
            prognostic_var_names=self.names,
            boundary_var_names=self.bundle.data_layout.boundary_var_names,
            input_steps=19,
            output_steps=1,
            steps=steps,
            normalize_before_mask=True,
            masked_fill_value=0.0,
        )

    def __init__(self, args):
        super().__init__(args)
        self.native_mean, self.native_std = self.mean.clone(), self.std.clone()
        self.scaling_contract = None
        if getattr(args, "observation_normalization_root", None):
            scaling = Samples(args.observation_normalization_root, "cpu")
            scaling.use_observation_normalization()
            if list(scaling.grid["names"]) != list(self.names):
                raise ValueError("Observation/OM4 channel order differs")
            self.mean = scaling.mean[0, 0, :, 0, 0].to(self.device)
            self.std = scaling.std[0, 0, :, 0, 0].to(self.device)
            self.scaling_contract = {
                "root": args.observation_normalization_root,
                "statistics_sha256": hashlib.sha256(
                    (
                        Path(args.observation_normalization_root) / "statistics.npz"
                    ).read_bytes()
                ).hexdigest(),
                "mean": self.mean.cpu().tolist(),
                "std": self.std.cpu().tolist(),
                "policy": "Shared observation-training scales; original OM4 forcing normalization unchanged",
            }
        self.pretrain = Path(args.wave1_root) / "ar/pretrain-best.pt"
        evolution = Evolution(self.channels, [128, 192, 256, 384], "ar").to(self.device)
        configure_normalization(evolution, getattr(args, "normalization", "batch"))
        if not getattr(args, "fresh_evolution", False):
            saved = torch.load(self.pretrain, map_location="cpu", weights_only=False)[
                "model"
            ]
            evolution.load_state_dict(
                {
                    k.removeprefix("evolution."): v
                    for k, v in saved.items()
                    if k.startswith("evolution.")
                }
            )
        self.model = Pair(self.initializer, evolution, args.phase == "joint").to(
            self.device
        )
        if args.initial_checkpoint:
            saved = torch.load(
                args.initial_checkpoint, map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(saved["model"])
        self.original_dynamics = state_fingerprint(evolution)
        self.context_bundle = self.context_config().build(
            LocalLocation(path=args.data_root)
        )
        self.trainset = self.dataset(self.source)
        self.valset = self.dataset(self.context_bundle.val_sources[0])
        self.val_ids = list(range(0, len(self.valset), 6))[: args.val_origins]
        self.train_probe_ids = np.linspace(
            0, len(self.trainset) - 1, len(self.val_ids), dtype=int
        ).tolist()
        self.verified_sources = set()
        self.initial_views = {}
        if self.rank == 0:
            extra = {
                "state_scaling": self.scaling_contract,
                "architecture": ARMS[args.arm][0],
                "expanded_inputs": ARMS[args.arm][1],
                "initializer_parameters": sum(
                    p.numel() for p in self.initializer.parameters()
                ),
                "evolution_initialization": "random"
                if getattr(args, "fresh_evolution", False)
                else "pretrained",
                "initial_evolution_fingerprint": self.original_dynamics,
                "train_origins": len(self.trainset),
                "validation_origins": [
                    str(self.valset.sources[0].time.values[i + 18])
                    for i in self.val_ids
                ],
                "context_source_config": self.context_config().model_dump(mode="json"),
                "context_policy": "19 frames fetched in all arms; short arms consume last six surfaces only; earlier validation/test context is read-only",
                "training_selection": "equal subsurface T and S normalized MSE; both reconstructed times in reconstruction phase; six forecast leads in joint phase",
            }
            (self.out / "study-manifest.json").write_text(
                json.dumps(extra, indent=2) + "\n"
            )
            self.emit({"event": "study_ready", **extra})

    def prepare(self, dataset):
        key = id(dataset.sources[0])
        if key in self.verified_sources:
            return
        # Parent warms through native Rust normalization and verifies full batches.
        next(iter(self.loader(dataset, [[0]])))
        ids = [0, len(dataset) - 1]
        reference = next(iter(self.native_loader(dataset, [ids])))
        surface, past, context, truth, forcing, labels = self.sample(dataset, ids)
        hist, boundary = reference.get_initial_input()
        h, w = self.mask.shape[-2:]
        full = hist.reshape(2, 19, self.channels, h, w)
        torch.testing.assert_close(
            surface, full[:, :, self.initializer.surface], rtol=0, atol=0
        )
        torch.testing.assert_close(truth, full[:, -2:], rtol=0, atol=0)
        torch.testing.assert_close(
            past, boundary.reshape(2, 19, 3, h, w), rtol=0, atol=0
        )
        torch.testing.assert_close(
            labels,
            torch.stack([reference.get_label(i) for i in range(len(reference))], 1),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            forcing,
            torch.stack(
                [reference.get_input(i)[1][:, -3:] for i in range(len(reference))], 1
            ),
            rtol=0,
            atol=0,
        )
        compact = self.initial_sample(dataset, ids)
        for actual, expected in zip(
            compact, (surface, past, context, truth), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.verified_sources.add(key)
        self.emit(
            {
                "event": "initializer_sampling_verified",
                "device_cache": key in self.frame_caches,
                "source_first_time": str(dataset.sources[0].time.values[0]),
            }
        )

    def sample(self, dataset, ids):
        cache = self.frame_caches.get(id(dataset.sources[0]))
        if cache is None:
            batch = next(iter(self.native_loader(dataset, [ids])))
            hist, boundary = batch.get_initial_input()
            h, w = self.mask.shape[-2:]
            full = hist.reshape(len(ids), 19, self.channels, h, w)
            surface = full[:, :, self.initializer.surface]
            past = boundary.reshape(len(ids), 19, 3, h, w)
            truth = full[:, -2:]
            forcing = torch.stack(
                [batch.get_input(i)[1][:, -3:] for i in range(len(batch))], 1
            )
            labels = torch.stack([batch.get_label(i) for i in range(len(batch))], 1)
        else:
            offsets = torch.tensor(ids, device=self.device)[:, None]
            history = offsets + torch.arange(19, device=self.device)
            channels = torch.tensor(self.initializer.surface, device=self.device)
            surface = cache.prognostic[history[:, :, None], channels]
            past = cache.boundary[history]
            truth = cache.prognostic[history[:, -2:]]
            future = offsets + torch.arange(18, 18 + dataset.steps, device=self.device)
            forcing = cache.boundary[future]
            labels = cache.prognostic[future + 1]
        dates = dataset.sources[0].time.values[np.asarray(ids) + 18]
        phases = [2 * math.pi * (t.dayofyr - 1) / 365.25 for t in dates]
        season = surface.new_tensor([[math.sin(p), math.cos(p)] for p in phases])
        season = season[:, :, None, None].expand(-1, -1, *self.mask.shape[-2:])
        context = torch.cat((self.geo.expand(len(ids), -1, -1, -1), season), 1)
        return surface, past, context, truth, forcing, labels

    def initial_sample(self, dataset, ids):
        """Read only surface history and the two reconstructed states for A/B.

        Both views retain native Rust preparation, normalization and masking.
        The generic full rollout reader remains the parity reference in prepare().
        """
        key = id(dataset.sources[0])
        if key in self.frame_caches:
            return self.sample(dataset, ids)[:4]
        if key not in self.initial_views:

            def view(names, history):
                return TorchTrainDataset(
                    input_source=dataset.sources[0],
                    label_source=None,
                    prognostic_var_names=names,
                    boundary_var_names=self.bundle.data_layout.boundary_var_names,
                    input_steps=history,
                    output_steps=1,
                    steps=1,
                    normalize_before_mask=True,
                    masked_fill_value=0.0,
                )

            self.initial_views[key] = (
                view([self.names[i] for i in self.initializer.surface], 19),
                view(self.names, 2),
            )
        surface_view, interior_view = self.initial_views[key]
        surface_batch = next(iter(self.native_loader(surface_view, [ids])))
        history, boundary = surface_batch.get_initial_input()
        h, w = self.mask.shape[-2:]
        surface = history.reshape(len(ids), 19, len(self.initializer.surface), h, w)
        past = boundary.reshape(len(ids), 19, 3, h, w)
        interior_batch = next(
            iter(self.native_loader(interior_view, [[i + 17 for i in ids]]))
        )
        truth = interior_batch.get_initial_input()[0].reshape(
            len(ids), 2, self.channels, h, w
        )
        dates = dataset.sources[0].time.values[np.asarray(ids) + 18]
        phases = [2 * math.pi * (t.dayofyr - 1) / 365.25 for t in dates]
        season = surface.new_tensor([[math.sin(p), math.cos(p)] for p in phases])
        season = season[:, :, None, None].expand(-1, -1, h, w)
        context = torch.cat((self.geo.expand(len(ids), -1, -1, -1), season), 1)
        return surface, past, context, truth

    def convert_state(self, values, channels):
        if self.scaling_contract is None:
            return values
        old_mean = self.native_mean[channels][None, None, :, None, None]
        old_std = self.native_std[channels][None, None, :, None, None]
        mean = self.mean[channels][None, None, :, None, None]
        std = self.std[channels][None, None, :, None, None]
        result = (values * old_std + old_mean - mean) / std
        return result * self.mask[channels][None, None]

    def model_initial_sample(self, dataset, ids):
        surface, past, context, truth = self.initial_sample(dataset, ids)
        return (
            self.convert_state(surface, self.initializer.surface),
            past,
            context,
            self.convert_state(truth, slice(None)),
        )

    def model_sample(self, dataset, ids):
        surface, past, context, truth, forcing, labels = self.sample(dataset, ids)
        return (
            self.convert_state(surface, self.initializer.surface),
            past,
            context,
            self.convert_state(truth, slice(None)),
            forcing,
            self.convert_state(labels, slice(None)),
        )

    def score(self, dataset, indices, forecast=False):
        self.prepare(dataset)
        self.sync_buffers(self.model)
        self.model.eval()
        totals = torch.zeros(4, device=self.device)
        with torch.no_grad():
            for ids in batches(indices[self.rank :: self.world], self.args.batch_size):
                surface, past, context, truth, forcing, labels = self.model_sample(
                    dataset, ids
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    initial = self.initializer(surface, past, context, self.mask)
                    prediction = (
                        self.model.evolve(initial, forcing, context, self.mask)
                        if forecast
                        else initial
                    )
                rec = thermohaline_mse(
                    channel_mse(initial, truth, self.weights), self.names
                ).mean()
                target = labels if forecast else truth
                ts = thermohaline_mse(
                    channel_mse(prediction, target, self.weights), self.names
                ).mean()
                full = balanced_loss(
                    prediction, target, self.weights, self.names, not forecast
                )
                totals += torch.stack(
                    (
                        ts * len(ids),
                        rec * len(ids),
                        full * len(ids),
                        ts.new_tensor(len(ids)),
                    )
                )
        self.reduce(totals)
        self.model.train()
        return {
            "ts_mse": float(totals[0] / totals[3]),
            "reconstruction_ts_mse": float(totals[1] / totals[3]),
            "full_mse": float(totals[2] / totals[3]),
            "origins": int(totals[3]),
        }

    def fit(self):
        args = self.args
        model = self.model
        last, best = self.out / "last.pt", self.out / "best.pt"
        state: AdaptationState = dict(
            epoch=0,
            cursor=0,
            step=0,
            elapsed=0.0,
            best=float("inf"),
            bad_checks=0,
            complete=False,
        )
        signature = {
            "arm": args.arm,
            "phase": args.phase,
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "accumulate": args.accumulate,
            "warmup_steps": getattr(args, "warmup_steps", 0),
            "world_size": self.world,
            "data_config": self.config.model_dump(mode="json"),
            "data_root": args.data_root,
        }
        if self.scaling_contract is not None:
            signature["state_scaling"] = self.scaling_contract
        if getattr(args, "normalization", "batch") != "batch":
            signature["normalization"] = args.normalization
        if getattr(args, "fresh_evolution", False):
            signature["fresh_evolution"] = True
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        if last.exists():
            saved = torch.load(last, map_location=self.device, weights_only=False)
            if saved["signature"] != signature:
                raise ValueError("Resume training protocol differs from checkpoint")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
        self.prepare(self.trainset)
        self.prepare(self.valset)
        if not math.isfinite(state["best"]):
            metrics = self.score(self.valset, self.val_ids, args.phase == "joint")
            state["best"] = metrics["ts_mse"]
            if self.rank == 0:
                atomic_save({"model": model.state_dict(), "state": state.copy()}, best)
            self.emit({"event": "baseline", "phase": args.phase, **metrics})
        wrapped = (
            DDP(
                model,
                device_ids=[self.device.index],
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if self.world > 1
            else model
        )
        started = time.monotonic()
        prior = state["elapsed"]
        last_save = last_validation = started
        optimizer.zero_grad(set_to_none=True)
        while not state["complete"]:
            schedule = training_batches(
                len(self.trainset),
                args.batch_size,
                self.world,
                self.rank,
                args.seed,
                state["epoch"],
            )
            # Each epoch ends on an optimizer boundary, preserving resume semantics.
            schedule = schedule[: len(schedule) // args.accumulate * args.accumulate]
            if state["cursor"] >= len(schedule):
                state["epoch"] += 1
                state["cursor"] = 0
                continue
            for ids in schedule[state["cursor"] :]:
                warmup = getattr(args, "warmup_steps", 0)
                factor = min(1.0, (state["step"] + 1) / warmup) if warmup else 1.0
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * factor
                surface, past, context, truth, forcing, labels = self.model_sample(
                    self.trainset, ids
                )
                boundary = (state["cursor"] + 1) % args.accumulate == 0
                sync = (
                    wrapped.no_sync()
                    if isinstance(wrapped, DDP) and not boundary
                    else contextlib.nullcontext()
                )
                with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, initial = wrapped(
                        surface, past, context, self.mask, forcing
                    )
                    reconstruction = balanced_loss(
                        initial, truth, self.weights, self.names, True
                    )
                    loss = (
                        balanced_loss(prediction, labels, self.weights, self.names)
                        + 0.1 * reconstruction
                        if args.phase == "joint"
                        else reconstruction
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite loss")
                    (loss / args.accumulate).backward()
                state["cursor"] += 1
                if not boundary:
                    continue
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                state["step"] += 1
                now = time.monotonic()
                state["elapsed"] = prior + now - started
                flags = torch.tensor(
                    [
                        state["elapsed"] >= args.hours * 3600
                        or datetime.datetime.now(datetime.UTC)
                        >= datetime.datetime.fromisoformat(
                            args.deadline.replace("Z", "+00:00")
                        )
                        or bool(args.max_steps and state["step"] >= args.max_steps),
                        now - last_validation >= args.validation_seconds,
                        now - last_save >= args.checkpoint_seconds,
                    ],
                    device=self.device,
                    dtype=torch.int32,
                )
                if self.world > 1:
                    dist.broadcast(flags, 0)
                done, validate, save = map(bool, flags.tolist())
                if state["step"] % 20 == 0 or state["step"] == 1:
                    self.emit(
                        {
                            "phase": args.phase,
                            "step": state["step"],
                            "loss_rank0": float(loss.detach()),
                            "elapsed_seconds": state["elapsed"],
                            "learning_rate": args.learning_rate,
                            "global_batch": args.batch_size
                            * self.world
                            * args.accumulate,
                        }
                    )
                if validate or done:
                    metrics = self.score(
                        self.valset, self.val_ids, args.phase == "joint"
                    )
                    train = self.score(
                        self.trainset, self.train_probe_ids, args.phase == "joint"
                    )
                    if not all(math.isfinite(v) for v in metrics.values()):
                        raise FloatingPointError("Nonfinite validation")
                    improved = metrics["ts_mse"] < state["best"]
                    state["bad_checks"] = 0 if improved else state["bad_checks"] + 1
                    if improved:
                        state["best"] = metrics["ts_mse"]
                        if self.rank == 0:
                            atomic_save(
                                {"model": model.state_dict(), "state": state.copy()},
                                best,
                            )
                    done = done or (
                        state["bad_checks"] >= args.patience
                        and state["elapsed"] >= args.min_hours * 3600
                    )
                    last_validation = time.monotonic()
                    self.emit(
                        {
                            "event": "validation",
                            "phase": args.phase,
                            "step": state["step"],
                            "elapsed_seconds": state["elapsed"],
                            "best_ts_mse": state["best"],
                            "bad_checks": state["bad_checks"],
                            "train_probe": train,
                            **metrics,
                        }
                    )
                milestone = state["step"] in getattr(args, "milestone_steps", [])
                if milestone:
                    self.sync_buffers(model)
                    if self.rank == 0:
                        atomic_save(
                            {"model": model.state_dict(), "state": state.copy()},
                            self.out / f"step-{state['step']:05d}.pt",
                        )
                state["complete"] = done
                if save or validate or done:
                    state["elapsed"] = prior + time.monotonic() - started
                    self.sync_buffers(model)
                    if self.rank == 0:
                        atomic_save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "signature": signature,
                                "state": state.copy(),
                            },
                            last,
                        )
                    last_save = time.monotonic()
                if done:
                    break
        self.barrier()
        model.load_state_dict(
            torch.load(best, map_location=self.device, weights_only=False)["model"]
        )
        if (
            args.phase != "joint"
            and state_fingerprint(model.evolution) != self.original_dynamics
        ):
            raise ValueError("Fixed evolution changed")
        metrics = self.score(self.valset, self.val_ids, args.phase == "joint")
        training = self.score(
            self.trainset, self.train_probe_ids, args.phase == "joint"
        )
        if self.rank == 0:
            (self.out / "TRAIN_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "state": state,
                        "selected_validation": metrics,
                        "selected_train_probe": training,
                        "code_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
                        "checkpoint_sha256": hashlib.sha256(
                            best.read_bytes()
                        ).hexdigest(),
                    },
                    indent=2,
                )
                + "\n"
            )

    def evaluate_wave(self):
        import csv

        self.model.load_state_dict(
            torch.load(
                self.out / "best.pt", map_location=self.device, weights_only=False
            )["model"]
        )
        self.model.eval()
        self.frame_caches.clear()
        self.verified_sources.clear()
        torch.cuda.empty_cache()
        source = native_om4_source(
            self.context_bundle.inference_source,
            LocalLocation(path=Path(self.args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(self.args.readers),
        )
        dataset = self.dataset(source)
        self.prepare(dataset)
        indices = list(range(0, len(dataset), 6))
        if self.args.max_steps:
            indices = indices[: self.world * 2]
        diagnostics = ReconstructionDiagnostics(self, source, indices)
        regions = {
            "global": torch.ones_like(self.lat, dtype=torch.bool),
            "tropics": self.lat.abs() <= 20,
            "extratropics": self.lat.abs() > 20,
        }
        output = self.out / f"heldout-rank{self.rank}.csv"
        tmp = output.with_suffix(".tmp")
        with tmp.open("w", newline="") as stream, torch.no_grad():
            fields = [
                "mode",
                "region",
                "origin_time",
                "lead_days",
                "channel",
                "normalized_mse",
                "physical_mse",
                "prediction_second_moment",
                "target_second_moment",
            ]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for ids in batches(indices[self.rank :: self.world], self.args.batch_size):
                surface, past, context, truth, forcing, labels = self.model_sample(
                    dataset, ids
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    initial = self.initializer(surface, past, context, self.mask)
                    inferred = self.model.evolve(initial, forcing, context, self.mask)
                    true = self.model.evolve(truth, forcing, context, self.mask)
                diagnostics.add(initial[:, -1], truth[:, -1], ids)
                estimates = {
                    "inferred": torch.cat((initial[:, -1:], inferred), 1),
                    "true": torch.cat((truth[:, -1:], true), 1),
                    "inferred_persistence": initial[:, -1:].expand(-1, 7, -1, -1, -1),
                    "true_persistence": truth[:, -1:].expand(-1, 7, -1, -1, -1),
                }
                targets = torch.cat((truth[:, -1:], labels), 1)
                for mode, prediction in estimates.items():
                    for region, selector in regions.items():
                        errors = (
                            channel_mse(
                                prediction,
                                targets,
                                self.weights * selector[None, :, None],
                            )
                            .cpu()
                            .numpy()
                        )
                        weights = self.weights * selector[None, :, None]
                        denominator = weights.sum((-2, -1)).clamp_min(1e-12)
                        moments = []
                        for values in (prediction, targets):
                            physical = (
                                values.float() * self.std[None, None, :, None, None]
                                + self.mean[None, None, :, None, None]
                            )
                            moments.append(
                                (
                                    (physical.square() * weights).sum((-2, -1))
                                    / denominator
                                )
                                .cpu()
                                .numpy()
                            )
                        std2 = self.std.square().cpu().numpy()
                        for j, index in enumerate(ids):
                            for lead in range(7):
                                for c, name in enumerate(self.names):
                                    writer.writerow(
                                        dict(
                                            mode=mode,
                                            region=region,
                                            origin_time=str(
                                                source.time.values[index + 18]
                                            ),
                                            lead_days=lead * 5,
                                            channel=name,
                                            normalized_mse=float(errors[j, lead, c]),
                                            physical_mse=float(
                                                errors[j, lead, c] * std2[c]
                                            ),
                                            prediction_second_moment=float(
                                                moments[0][j, lead, c]
                                            ),
                                            target_second_moment=float(
                                                moments[1][j, lead, c]
                                            ),
                                        )
                                    )
                self.emit(
                    {
                        "event": "evaluation_batch",
                        "origins": len(ids),
                        "last_origin": str(source.time.values[ids[-1] + 18]),
                    }
                )
        tmp.replace(output)
        diagnostics.finish()
        self.barrier()
        if self.rank == 0:
            (self.out / "EVAL_COMPLETE.json").write_text(
                json.dumps(
                    {
                        "origins": len(indices),
                        "ranks": self.world,
                        "code_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
                    }
                )
                + "\n"
            )

    def run_wave(self):
        try:
            if not self.args.evaluate_only:
                self.fit()
            if self.args.evaluate_only or (
                self.args.max_steps and not getattr(self.args, "train_only", False)
            ):
                self.evaluate_wave()
            if self.rank == 0:
                (self.out / "COMPLETE.json").write_text(
                    json.dumps(
                        {
                            "phase": self.args.phase,
                            "arm": self.args.arm,
                            "max_steps": self.args.max_steps,
                            "evaluate_only": self.args.evaluate_only,
                            "code_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
                            "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                        }
                    )
                    + "\n"
                )
        except BaseException:
            (self.out / f"exception-rank{self.rank}.log").write_text(
                traceback.format_exc()
            )
            raise
        finally:
            if self.run:
                self.run.finish()
            if dist.is_initialized():
                dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalization", choices=["batch", "instance"], default="batch"
    )
    parser.add_argument("--fresh-evolution", action="store_true")
    parser.add_argument("--observation-normalization-root")
    parser.add_argument("--deadline", default="2026-09-24T17:00:00Z")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument(
        "--phase", choices=["reconstruction", "joint"], default="reconstruction"
    )
    parser.add_argument("--data-root", default="/scratch/jr7309/data/om4_onedeg_v3")
    parser.add_argument(
        "--wave1-root", default="/scratch/jr7309/runs/2026-09-18-surface-wave1"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--readers", type=int, default=8)
    parser.add_argument("--device-cache", action="store_true", default=True)
    parser.add_argument("--device-cache-reserve-gib", type=float, default=30)
    parser.add_argument("--val-origins", type=int, default=12)
    parser.add_argument("--checkpoint-seconds", type=int, default=180)
    parser.add_argument("--validation-seconds", type=int, default=1200)
    parser.add_argument("--min-hours", type=float, default=2)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--milestone-steps", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--wandb-mode", choices=["online", "disabled", "offline"], default="online"
    )
    args = parser.parse_args()
    if args.hours <= 0 or args.accumulate < 1:
        parser.error("positive hours and accumulation required")
    if not args.evaluate_only and datetime.datetime.now(
        datetime.UTC
    ) >= datetime.datetime.fromisoformat(args.deadline.replace("Z", "+00:00")):
        parser.error("Training deadline has passed")
    InitializerWave(args).run_wave()


if __name__ == "__main__":
    main()
