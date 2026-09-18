# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded wave-one training/evaluation, using canonical Rust model batches.

Run with torchrun -m samudra.experiments.surface_wave --help. Every task has a
wall-time training cap, resumable phase checkpoints, and machine-readable output.
"""

import argparse
import copy
import csv
import datetime
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from samudra.config import DataConfig
from samudra.datasets import TorchTrainDataset
from samudra.experiments.frame_cache import PreparedFrameCache
from samudra.experiments.surface_state import (
    Evolution,
    Forecast,
    Initializer,
    balanced_loss,
    channel_mse,
    geographic_features,
)
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.train_data_loader import build_train_batch_loader
from samudra.utils.location import LocalLocation


class PhaseState(TypedDict):
    epoch: int
    cursor: int
    step: int
    elapsed: float
    best: float
    complete: bool


def atomic_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def batches(indices, batch_size):
    return [
        list(map(int, indices[i : i + batch_size]))
        for i in range(0, len(indices), batch_size)
    ]


def training_batches(size, batch_size, world, rank, seed, epoch):
    indices = np.random.default_rng(seed + epoch).permutation(size)
    indices = indices[: size // (world * batch_size) * world * batch_size]
    return batches(indices[rank::world], batch_size)


def data_config(args):
    return DataConfig.model_validate(
        {
            "input_steps": 6,
            "output_steps": 1,
            "loading": {
                "type": "rust",
                "max_concurrent_reads": args.readers,
                "prefetch_batches": 2,
                "prefetch_to_device": True,
            },
            "sources": [
                {
                    "type": "om4",
                    "prognostic_vars_key": "thermo_dynamic_all",
                    "boundary_vars_key": "tau_hfds",
                    "data_location": "OM4.zarr",
                    "data_means_location": "OM4_means.zarr",
                    "data_stds_location": "OM4_stds.zarr",
                    "train_time": {"start": "1975-01-03", "end": "2013-10-04"},
                    "val_time": {"start": "2013-10-05", "end": "2014-10-05"},
                    "inference_times": [{"start": "2014-10-10", "end": "2022-12-24"}],
                }
            ],
        }
    )


class Experiment:
    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(self.device)
        if self.world > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.frame_caches: dict[int, PreparedFrameCache] = {}
        self.config = data_config(args)
        self.bundle = self.config.build(LocalLocation(path=args.data_root))
        self.source = self.bundle.train_sources[0]
        self.names = self.bundle.data_layout.prognostic_var_names
        self.channels = len(self.names)
        self.mask = self.source.masks.prognostic.to(self.device)
        lat, lon = self.source.resolution
        self.lat = lat.to(self.device).float()
        self.geo = geographic_features(self.lat, lon.to(self.device).float())
        self.weights = self.mask.float() * torch.deg2rad(self.lat).cos()[None, :, None]
        self.std = torch.tensor(
            self.source.statistics(self.names).std, device=self.device
        )
        self.mean = torch.tensor(
            self.source.statistics(self.names).mean, device=self.device
        )
        self.initializer = Initializer(self.names, args.widths).to(self.device)
        self.run = None
        if self.rank == 0:
            manifest = {
                "arguments": vars(args),
                "data": self.config.model_dump(mode="json"),
                "world_size": self.world,
                "channels": self.names,
                "depths_m": list(self.bundle.data_layout.depth_levels),
                "code_commit": os.environ.get("SAMUDRA_CODE_COMMIT", "unknown"),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "device": torch.cuda.get_device_name(self.device),
                "normalization": "existing mean/std files; date range explicitly approved",
                "boundary": "OM4 tauuo/tauvo/hfds at interval-start frame, five-day cadence",
                "claims": "model-world feasibility only; no observational or daily skill claim",
            }
            (self.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
            (self.out / f"manifest-{manifest['slurm_job_id']}.json").write_text(
                json.dumps(manifest, indent=2)
            )
            import wandb

            self.run = wandb.init(
                project="surface-initialized-ocean",
                name=args.name,
                dir=str(self.out),
                config=manifest,
                id=args.name,
                resume="allow",
                mode=args.wandb_mode,
            )
            print(json.dumps({"event": "ready", **manifest}), flush=True)

    def barrier(self):
        if self.world > 1:
            dist.barrier()

    def reduce(self, tensor):
        if self.world > 1:
            dist.all_reduce(tensor)
        return tensor

    def dataset(self, source, steps=6):
        return TorchTrainDataset(
            input_source=source,
            label_source=None,
            prognostic_var_names=self.names,
            boundary_var_names=self.bundle.data_layout.boundary_var_names,
            input_steps=6,
            output_steps=1,
            steps=steps,
            normalize_before_mask=True,
            masked_fill_value=0.0,
        )

    def native_loader(self, dataset, sampler):
        return build_train_batch_loader(
            [dataset],
            sampler,
            self.device,
            self.config.loading,
            pin_memory=True,
            multiprocessing_context=None,
            worker_seed=self.args.seed,
        )

    def loader(self, dataset, sampler):
        if not self.args.device_cache:
            return self.native_loader(dataset, sampler)
        source = dataset.sources[0]
        if len(dataset.sources) != 1 or dataset.stride != 1:
            raise ValueError("Resident wave cache requires one source and unit stride")
        key = id(source)
        if key not in self.frame_caches:
            warm = self.dataset(source, 1)
            frames = len(source.time)
            shape = tuple(self.mask.shape[-2:])
            needed = frames * (self.channels + 3) * math.prod(shape) * 4
            free, _ = torch.cuda.mem_get_info(self.device)
            if needed + self.args.device_cache_reserve_gib * 2**30 > free:
                raise MemoryError(
                    f"Resident cache needs {needed / 2**30:.2f} GiB plus "
                    f"{self.args.device_cache_reserve_gib} GiB reserve; "
                    f"only {free / 2**30:.2f} GiB free"
                )
            cache = PreparedFrameCache(frames, self.channels, 3, shape, self.device)
            indices = list(range(0, len(warm), warm.input_steps))
            if indices[-1] != len(warm) - 1:
                indices.append(len(warm) - 1)
            schedule = batches(indices, self.args.batch_size)
            self.emit(
                {
                    "event": "cache_warm_start",
                    "frames": frames,
                    "cache_gib": needed / 2**30,
                }
            )
            for ids, batch in zip(
                schedule, self.native_loader(warm, schedule), strict=True
            ):
                cache.record(warm.shard.window_plan(ids), batch)
            # These are the only boundary frames that any window can request.
            if not cache.prognostic_ready.all() or not cache.boundary_ready[:-1].all():
                raise ValueError("Incomplete resident cache coverage")
            # Verify shuffled, repeated, edge and full-rollout requests through
            # the independent native path before trusting the cache for training.
            probes = [[len(dataset) - 1, 0], [len(dataset) // 2, len(dataset) // 2]]
            for ids, reference in zip(
                probes, self.native_loader(dataset, probes), strict=True
            ):
                cached = cache.batch(dataset, ids)
                for actual_step, expected_step in zip(cached, reference, strict=True):
                    for actual, expected in zip(
                        actual_step, expected_step, strict=True
                    ):
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.frame_caches[key] = cache
            self.emit(
                {
                    "event": "cache_ready",
                    "frames": frames,
                    "cache_gib": needed / 2**30,
                    "native_equivalence": "exact",
                }
            )
        cache = self.frame_caches[key]
        return (cache.batch(dataset, ids) for ids in sampler)

    def inputs(self, batch, dataset, indices):
        history = batch.get_initial_input()[0]
        b, _, h, w = history.shape
        origin_times = dataset.sources[0].time.values[np.asarray(indices) + 5]
        phases = [2 * math.pi * (t.dayofyr - 1) / 365.25 for t in origin_times]
        season = history.new_tensor([[math.sin(p), math.cos(p)] for p in phases])
        context = torch.cat(
            (
                self.geo.expand(b, -1, -1, -1),
                season[:, :, None, None].expand(-1, -1, h, w),
            ),
            1,
        )
        forcing = torch.stack(
            [batch.get_input(i)[1][:, -3:] for i in range(len(batch))], 1
        )
        labels = torch.stack([batch.get_label(i) for i in range(len(batch))], 1)
        return history, forcing, context, labels

    def emit(self, metrics):
        if self.rank == 0:
            metrics["time_utc"] = datetime.datetime.now(datetime.UTC).isoformat()
            metrics["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
            metrics["host_peak_gib_rank0"] = (
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
            )
            with (self.out / "progress.jsonl").open("a") as file:
                file.write(json.dumps(metrics) + "\n")
            print(json.dumps(metrics), flush=True)
            if self.run:
                self.run.log(metrics)

    def sync_buffers(self, model):
        if self.world > 1:
            for buffer in model.buffers():
                dist.broadcast(buffer, 0)

    def validation(self, model, mode, initializer=False):
        self.sync_buffers(model)
        model.eval()
        dataset = self.dataset(self.bundle.val_sources[0], 1 if initializer else 6)
        indices = list(range(0, len(dataset), 6))[: self.args.val_origins]
        sampler = batches(indices[self.rank :: self.world], self.args.batch_size)
        total = torch.zeros(2, device=self.device)
        with torch.no_grad():
            for ids, batch in zip(sampler, self.loader(dataset, sampler), strict=True):
                history, forcing, context, labels = self.inputs(batch, dataset, ids)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if initializer:
                        prediction = model(history, context, self.mask)
                        target = history.reshape(
                            len(ids), 6, self.channels, *history.shape[-2:]
                        )[:, -2:]
                    else:
                        prediction, _ = model(
                            history,
                            forcing,
                            context,
                            self.mask,
                            mode,
                            list(range(1, 7)),
                        )
                        target = labels
                    loss = balanced_loss(
                        prediction, target, self.weights, self.names, initializer
                    )
                total += total.new_tensor([float(loss) * len(ids), len(ids)])
        self.reduce(total)
        model.train()
        return (total[0] / total[1]).item()

    def train_phase(self, model, phase, seconds, initializer=False, inferred=False):
        path = self.out / f"{phase}-last.pt"
        best_path = self.out / f"{phase}-best.pt"
        state: PhaseState = {
            "epoch": 0,
            "cursor": 0,
            "step": 0,
            "elapsed": 0.0,
            "best": float("inf"),
            "complete": False,
        }
        lr = 1e-4 if inferred else 3e-4
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01
        )
        if path.exists():
            saved = torch.load(path, map_location=self.device, weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
        if state["complete"]:
            model.load_state_dict(
                torch.load(best_path, map_location=self.device, weights_only=False)[
                    "model"
                ]
            )
            return
        wrapped = (
            DDP(model, device_ids=[self.device.index], broadcast_buffers=False)
            if self.world > 1
            else model
        )
        dataset = self.dataset(self.source, 1 if initializer else 6)
        start = time.monotonic()
        prior_elapsed = state["elapsed"]
        last_save = start
        max_steps = self.args.max_steps
        mode = "inferred" if inferred else "true"
        baseline = self.validation(model, mode, initializer)
        if not math.isfinite(baseline):
            raise FloatingPointError("Non-finite initial validation loss")
        if state["best"] == float("inf"):
            state["best"] = baseline
            if self.rank == 0:
                atomic_save(
                    {"model": model.state_dict(), "state": state.copy()}, best_path
                )
        self.emit({"phase": phase, "event": "start", "validation_loss": baseline})
        done = False
        while not done:
            sampler = training_batches(
                len(dataset),
                self.args.batch_size,
                self.world,
                self.rank,
                self.args.seed,
                state["epoch"],
            )
            remaining = sampler[state["cursor"] :]
            if not remaining:
                state["epoch"] += 1
                state["cursor"] = 0
                continue
            for ids, batch in zip(
                remaining, self.loader(dataset, remaining), strict=True
            ):
                history, forcing, context, labels = self.inputs(batch, dataset, ids)
                progress = min(
                    1.0, (prior_elapsed + time.monotonic() - start) / max(seconds, 1)
                )
                if max_steps:
                    progress = state["step"] / max_steps
                # A uses 1/3/6-step curriculum in pretraining; joint always six.
                if initializer:
                    leads = [1]
                elif model.evolution.kind == "direct":
                    leads = [1 + state["step"] % 6]
                else:
                    horizon = (
                        6
                        if inferred or progress >= 0.5
                        else (3 if progress >= 0.2 else 1)
                    )
                    leads = list(range(1, horizon + 1))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if initializer:
                        prediction = wrapped(history, context, self.mask)
                        target = history.reshape(
                            len(ids), 6, self.channels, *history.shape[-2:]
                        )[:, -2:]
                        loss = balanced_loss(
                            prediction, target, self.weights, self.names, True
                        )
                    else:
                        prediction, initial = wrapped(
                            history, forcing, context, self.mask, mode, leads
                        )
                        loss = balanced_loss(
                            prediction,
                            labels[:, [lead - 1 for lead in leads]],
                            self.weights,
                            self.names,
                        )
                        if inferred:
                            target_initial = history.reshape(
                                len(ids), 6, self.channels, *history.shape[-2:]
                            )[:, -2:]
                            loss = loss + 0.1 * balanced_loss(
                                initial, target_initial, self.weights, self.names, True
                            )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss at {phase} step {state['step']}"
                    )
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                state["step"] += 1
                state["cursor"] += 1
                now = time.monotonic()
                state["elapsed"] = prior_elapsed + now - start
                # Rank zero controls phase boundaries so all ranks call identical collectives.
                flags = torch.tensor(
                    [
                        state["elapsed"] >= seconds
                        or bool(max_steps and state["step"] >= max_steps),
                        now - last_save >= self.args.checkpoint_seconds,
                    ],
                    device=self.device,
                    dtype=torch.int32,
                )
                if self.world > 1:
                    dist.broadcast(flags, 0)
                done, checkpoint = map(bool, flags.tolist())
                if state["step"] % 20 == 0 or done:
                    reduced_loss = self.reduce(loss.detach().float()) / self.world
                    self.emit(
                        {
                            "phase": phase,
                            "step": state["step"],
                            "loss": float(reduced_loss),
                            "elapsed_seconds": state["elapsed"],
                            "leads": leads,
                            "samples_per_second": state["step"]
                            * self.args.batch_size
                            * self.world
                            / state["elapsed"],
                        }
                    )
                if checkpoint or done:
                    score = self.validation(model, mode, initializer)
                    state["complete"] = done
                    # Include validation/checkpoint time in the next recorded budget.
                    state["elapsed"] = prior_elapsed + time.monotonic() - start
                    if score < state["best"]:
                        state["best"] = score
                        if self.rank == 0:
                            atomic_save(
                                {"model": model.state_dict(), "state": state.copy()},
                                best_path,
                            )
                    if self.rank == 0:
                        atomic_save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "state": state.copy(),
                            },
                            path,
                        )
                    self.emit(
                        {
                            "phase": phase,
                            "step": state["step"],
                            "validation_loss": score,
                            "best_validation_loss": state["best"],
                            "phase_complete": done,
                        }
                    )
                    self.barrier()
                    last_save = time.monotonic()
                if done:
                    break
        del wrapped
        model.load_state_dict(
            torch.load(best_path, map_location=self.device, weights_only=False)["model"]
        )

    def fit_climatology(self):
        """Monthly means from evenly spaced training origins, no held-out labels."""
        dataset = self.dataset(self.source, 1)
        indices = list(range(0, len(dataset), 6))
        if self.args.max_steps:
            indices = indices[:24]
        sampler = batches(indices[self.rank :: self.world], self.args.batch_size)
        sums = torch.zeros(
            (12, self.channels, *self.mask.shape[-2:]), device=self.device
        )
        counts = torch.zeros(12, device=self.device)
        for ids, batch in zip(sampler, self.loader(dataset, sampler), strict=True):
            current = batch.get_initial_input()[0][:, -self.channels :]
            times = dataset.sources[0].time.values[np.asarray(ids) + 5]
            for i, date in enumerate(times):
                sums[date.month - 1] += current[i]
                counts[date.month - 1] += 1
        self.reduce(sums)
        self.reduce(counts)
        # Short smoke runs may miss months; never silently allow that in a real run.
        if not self.args.max_steps and (counts == 0).any():
            raise ValueError("Empty climatology month")
        climatology = sums / counts.clamp_min(1)[:, None, None, None]
        if self.rank == 0:
            atomic_save(
                {
                    "monthly": climatology.cpu(),
                    "counts": counts.cpu(),
                    "stride": 6,
                    "train_only": True,
                },
                self.out / "climatology.pt",
            )
        self.barrier()

    def evaluate_initializer(self):
        self.sync_buffers(self.initializer)
        self.initializer.eval()
        dataset = self.dataset(self.bundle.val_sources[0], 1)
        indices = list(range(0, len(dataset), 6))[: self.args.val_origins]
        sampler = batches(indices[self.rank :: self.world], self.args.batch_size)
        climatology = torch.load(
            self.out / "climatology.pt", map_location=self.device, weights_only=False
        )["monthly"]
        sums = torch.zeros((2, self.channels), device=self.device)
        count = torch.zeros((), device=self.device)
        with torch.no_grad():
            for ids, batch in zip(sampler, self.loader(dataset, sampler), strict=True):
                history, _, context, _ = self.inputs(batch, dataset, ids)
                truth = history.reshape(
                    len(ids), 6, self.channels, *history.shape[-2:]
                )[:, -2:]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = self.initializer(history, context, self.mask)
                dates = dataset.sources[0].time.values
                months = [
                    [dates[i + offset].month - 1 for offset in (4, 5)] for i in ids
                ]
                seasonal = climatology[torch.tensor(months, device=self.device)].clone()
                seasonal[:, :, self.initializer.surface] = truth[
                    :, :, self.initializer.surface
                ]
                for mode, estimate in enumerate((prediction, seasonal)):
                    sums[mode] += channel_mse(estimate, truth, self.weights).sum((0, 1))
                count += 2 * len(ids)
        self.reduce(sums)
        self.reduce(count)
        if self.rank == 0:
            errors = (sums / count).sqrt().cpu().numpy()
            std = self.std.cpu().numpy()
            with (self.out / "initializer_metrics.csv").open("w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["split", "mode", "channel", "normalized_rmse", "physical_rmse"]
                )
                for mode, label in enumerate(
                    ("inferred", "climatology_with_observed_surface")
                ):
                    for c, name in enumerate(self.names):
                        writer.writerow(
                            [
                                "validation",
                                label,
                                name,
                                errors[mode, c],
                                errors[mode, c] * std[c],
                            ]
                        )
        self.barrier()

    def evaluate(self, model, climatology):
        # Held-out data use a separate cache; training frames are no longer needed.
        self.frame_caches.clear()
        torch.cuda.empty_cache()
        self.sync_buffers(model)
        model.eval()
        source = self.bundle.inference_source
        if source is None:
            raise ValueError("No held-out evaluation source")
        # Standard DataConfig intentionally uses a Python/Dask inference reader.
        # This experiment evaluates independent windows with the Rust batch API.
        # Its evaluation pool is only active after training loaders have closed.
        source = native_om4_source(
            source,
            LocalLocation(path=Path(self.args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(self.args.readers),
        )
        dataset = self.dataset(source)
        indices = list(range(0, len(dataset), 6))
        if self.args.max_steps:
            indices = indices[: self.world * 2]
        sampler = batches(indices[self.rank :: self.world], self.args.batch_size)
        # region, lead, channel; each origin has equal weight within each region.
        regions = {
            "global": torch.ones_like(self.lat, dtype=torch.bool),
            "tropics": self.lat.abs() <= 20,
            "extratropics": self.lat.abs() > 20,
        }
        modes = ("inferred", "true", "inferred_persistence", "climatology")
        sums = torch.zeros(
            (len(modes), len(regions), 6, self.channels, 3), device=self.device
        )
        count = torch.zeros((), device=self.device)
        with torch.no_grad():
            for ids, batch in zip(sampler, self.loader(dataset, sampler), strict=True):
                history, forcing, context, labels = self.inputs(batch, dataset, ids)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    inferred, initial = model(
                        history,
                        forcing,
                        context,
                        self.mask,
                        "inferred",
                        list(range(1, 7)),
                    )
                    truth, _ = model(
                        history, forcing, context, self.mask, "true", list(range(1, 7))
                    )
                persistence = initial[:, -1:].expand_as(inferred)
                dates = source.time.values
                months = [
                    [dates[i + 5 + lead].month - 1 for lead in range(1, 7)] for i in ids
                ]
                seasonal = climatology[torch.tensor(months, device=self.device)]
                for mi, prediction in enumerate(
                    (inferred, truth, persistence, seasonal)
                ):
                    for ri, region in enumerate(regions.values()):
                        weights = self.weights * region[None, :, None]
                        squared_error = channel_mse(prediction, labels, weights)
                        # Physical second moments retained to diagnose velocity energy/variance.
                        physical = (
                            prediction.float() * self.std[None, None, :, None, None]
                            + self.mean[None, None, :, None, None]
                        )
                        observed = (
                            labels.float() * self.std[None, None, :, None, None]
                            + self.mean[None, None, :, None, None]
                        )
                        denom = weights.sum((-2, -1)).clamp_min(1e-12)
                        p2 = (physical.square() * weights).sum((-2, -1)) / denom
                        t2 = (observed.square() * weights).sum((-2, -1)) / denom
                        sums[mi, ri] += torch.stack((squared_error, p2, t2), -1).sum(0)
                count += len(ids)
        self.reduce(sums)
        self.reduce(count)
        if self.rank == 0:
            values = (sums / count).cpu().numpy()
            std = self.std.cpu().numpy()
            with (self.out / "heldout_metrics.csv").open("w") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "mode",
                        "region",
                        "lead_days",
                        "channel",
                        "normalized_rmse",
                        "physical_rmse",
                        "prediction_second_moment",
                        "target_second_moment",
                        "origins",
                    ]
                )
                for mi, mode in enumerate(modes):
                    for ri, region in enumerate(regions):
                        for lead in range(6):
                            for c, name in enumerate(self.names):
                                mse, p2, t2 = values[mi, ri, lead, c]
                                rmse = math.sqrt(float(mse))
                                writer.writerow(
                                    [
                                        mode,
                                        region,
                                        (lead + 1) * 5,
                                        name,
                                        rmse,
                                        rmse * std[c],
                                        p2,
                                        t2,
                                        int(count.item()),
                                    ]
                                )
            self.emit({"event": "heldout_complete", "origins": int(count.item())})

    def execute(self, *, close_group=True):
        if self.args.task == "initializer":
            self.fit_climatology()
            self.train_phase(
                self.initializer,
                "initializer",
                self.args.hours * 3600,
                initializer=True,
            )
            self.evaluate_initializer()
        else:
            init_path = Path(self.args.initializer_dir) / "initializer-best.pt"
            self.initializer.load_state_dict(
                torch.load(init_path, map_location=self.device, weights_only=False)[
                    "model"
                ]
            )
            evolution = Evolution(self.channels, self.args.widths, self.args.task).to(
                self.device
            )
            model = Forecast(self.initializer, evolution).to(self.device)
            for p in self.initializer.parameters():
                p.requires_grad_(False)
            self.train_phase(model, "pretrain", self.args.hours * 3600 * 0.7)
            # Save the matched oracle/inferred diagnostic before modifying either model.
            for mode in ("true", "inferred"):
                self.emit(
                    {
                        "phase": "pretrain",
                        "initialization": mode,
                        "validation_loss": self.validation(model, mode),
                    }
                )
            if not self.args.stop_after_pretrain:
                for p in self.initializer.parameters():
                    p.requires_grad_(True)
                self.train_phase(
                    model, "joint", self.args.hours * 3600 * 0.3, inferred=True
                )
                climatology = torch.load(
                    Path(self.args.initializer_dir) / "climatology.pt",
                    map_location=self.device,
                    weights_only=False,
                )["monthly"]
                self.evaluate(model, climatology)
        if self.rank == 0:
            (
                self.out
                / (
                    "PRETRAIN_COMPLETE.json"
                    if self.args.stop_after_pretrain
                    else "COMPLETE.json"
                )
            ).write_text(
                json.dumps(
                    {
                        "task": self.args.task,
                        "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    }
                )
            )
            if self.run:
                self.run.finish()
        self.barrier()
        if self.world > 1 and close_group:
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["initializer", "ar", "direct", "smoke"], required=True
    )
    parser.add_argument("--data-root", default="/scratch/jr7309/data/om4_onedeg_v3")
    parser.add_argument("--output", required=True)
    parser.add_argument("--initializer-dir")
    parser.add_argument("--stop-after-pretrain", action="store_true")
    parser.add_argument("--name", required=True)
    parser.add_argument("--hours", type=float, required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Smoke only: steps per phase and reduced evaluation",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 192, 256, 384])
    parser.add_argument("--readers", type=int, default=4)
    parser.add_argument(
        "--device-cache",
        action="store_true",
        help="Cache prepared OM4 frames in GPU memory after exact native-reader verification",
    )
    parser.add_argument("--device-cache-reserve-gib", type=float, default=24.0)
    parser.add_argument("--val-origins", type=int, default=12)
    parser.add_argument("--checkpoint-seconds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--wandb-mode", choices=["online", "disabled", "offline"], default="online"
    )
    args = parser.parse_args()
    if args.hours <= 0 or (args.task in ("ar", "direct") and not args.initializer_dir):
        parser.error("positive --hours and forecast --initializer-dir required")
    if args.task == "smoke":
        if not args.max_steps:
            parser.error("smoke requires --max-steps")
        for task in ("initializer", "ar", "direct"):
            selected = copy.deepcopy(args)
            selected.task = task
            selected.output = str(Path(args.output) / task)
            selected.name = args.name + "-" + task
            selected.initializer_dir = str(Path(args.output) / "initializer")
            Experiment(selected).execute(close_group=False)
        if dist.is_initialized():
            dist.destroy_process_group()
    else:
        Experiment(args).execute()


if __name__ == "__main__":
    main()
