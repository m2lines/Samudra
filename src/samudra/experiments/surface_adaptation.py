# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Controlled adaptation of wave-one AR checkpoints; no new pretraining."""

import argparse
import copy
import csv
import datetime
import faulthandler
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from samudra.experiments.surface_state import (
    Evolution,
    Forecast,
    balanced_loss,
    channel_mse,
)
from samudra.experiments.surface_wave import (
    Experiment,
    atomic_save,
    batches,
    training_batches,
)
from samudra.rust_data import create_rust_io_runtime, native_om4_source
from samudra.utils.location import LocalLocation


class AdaptationState(TypedDict):
    epoch: int
    cursor: int
    step: int
    elapsed: float
    best: float
    bad_checks: int
    complete: bool


VARIABLES = ("thetao", "so", "uo", "vo", "zos", "sst")


def variable_indices(names, variable):
    if variable == "sst":
        return [names.index("thetao_0")]
    return [
        i
        for i, name in enumerate(names)
        if (name == variable or name.startswith(variable + "_")) and name != "thetao_0"
    ]


def thermohaline_mse(per_channel, names):
    """Equal subsurface-temperature/salinity score, retaining sample/lead axes."""
    return torch.stack(
        [
            per_channel[..., variable_indices(names, v)].mean(-1)
            for v in ("thetao", "so")
        ],
        -1,
    ).mean(-1)


def state_fingerprint(module):
    """Fingerprint weights and buffers, including batch-normalization statistics."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class AdaptiveForecast(Forecast):
    def __init__(self, initializer, evolution, arm):
        super().__init__(initializer, evolution)
        if arm not in ("initializer", "evolution", "joint"):
            raise ValueError(arm)
        self.arm = arm
        self.initializer.requires_grad_(arm != "evolution")
        self.evolution.requires_grad_(arm != "initializer")
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Frozen evolution must remain differentiable with respect to its input.
        # eval() fixes BN buffers; no_grad()/detach() would break arm A.
        if self.arm == "initializer":
            self.evolution.eval()
        if self.arm == "evolution":
            self.initializer.eval()
        return self

    def frozen_module(self):
        if self.arm == "initializer":
            return self.evolution
        if self.arm == "evolution":
            return self.initializer
        return None


class Adaptation(Experiment):
    def __init__(self, args):
        super().__init__(args)
        self.validation_id = 0
        self.pretrain_path = Path(args.wave1_root) / "ar/pretrain-best.pt"
        self.wave1_joint_path = Path(args.wave1_root) / "ar/joint-best.pt"
        self.initializer_path = (
            Path(args.wave1_root) / "initializer/initializer-best.pt"
        )
        self.climatology_path = Path(args.wave1_root) / "initializer/climatology.pt"

    def make_model(self, checkpoint, arm="joint"):
        model = AdaptiveForecast(
            copy.deepcopy(self.initializer),
            Evolution(self.channels, self.args.widths, "ar").to(self.device),
            arm,
        )
        model.load_state_dict(
            torch.load(checkpoint, map_location=self.device, weights_only=False)[
                "model"
            ]
        )
        return model

    def append_records(self, filename, rows):
        if not rows:
            return
        path = self.out / filename
        exists = path.exists()
        with path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def origin_records(
        self, errors, prediction, target, ids, dataset, mode, region, evaluation
    ):
        """Preserve paired origin-level errors for uncertainty and common-reference comparisons."""
        records = []
        means = errors.detach().cpu().numpy()
        p2 = prediction.detach().cpu().numpy()
        t2 = target.detach().cpu().numpy()
        std2 = self.std.square().cpu().numpy()
        for sample, index in enumerate(ids):
            origin = str(dataset.sources[0].time.values[index + 5])
            for lead in range(means.shape[1]):
                for variable in VARIABLES:
                    channels = variable_indices(self.names, variable)
                    records.append(
                        {
                            "job_id": os.environ.get("SLURM_JOB_ID", "local"),
                            "evaluation": evaluation,
                            "mode": mode,
                            "region": region,
                            "origin_index": index,
                            "origin_time": origin,
                            "lead_days": (lead + 1) * 5,
                            "variable": variable,
                            "normalized_mse": float(
                                means[sample, lead, channels].mean()
                            ),
                            "physical_mse": float(
                                (means[sample, lead, channels] * std2[channels]).mean()
                            ),
                            "prediction_second_moment": float(
                                p2[sample, lead, channels].mean()
                            ),
                            "target_second_moment": float(
                                t2[sample, lead, channels].mean()
                            ),
                        }
                    )
        return records

    def moments(self, prediction, labels, weights):
        mean = self.mean[None, None, :, None, None]
        std = self.std[None, None, :, None, None]
        denom = weights.sum((-2, -1)).clamp_min(1e-12)
        return tuple(
            ((value.float() * std + mean).square() * weights).sum((-2, -1)) / denom
            for value in (prediction, labels)
        )

    def validate_adaptation(self, model):
        self.sync_buffers(model)
        model.eval()
        dataset = self.dataset(self.bundle.val_sources[0])
        indices = list(range(0, len(dataset), 6))[: self.args.val_origins]
        sampler = batches(indices[self.rank :: self.world], self.args.batch_size)
        # T/S selection, full-state forecast loss, reconstruction T/S, reconstruction full.
        totals = torch.zeros(5, device=self.device)
        records = []
        self.validation_id += 1
        with torch.no_grad():
            for ids, batch in zip(sampler, self.loader(dataset, sampler), strict=True):
                history, forcing, context, labels = self.inputs(batch, dataset, ids)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, initial = model(
                        history,
                        forcing,
                        context,
                        self.mask,
                        "inferred",
                        list(range(1, 7)),
                    )
                errors = channel_mse(prediction, labels, self.weights)
                initial_truth = history.reshape(
                    len(ids), 6, self.channels, *history.shape[-2:]
                )[:, -2:]
                reconstruction = channel_mse(initial, initial_truth, self.weights)
                totals += totals.new_tensor(
                    [
                        float(thermohaline_mse(errors, self.names).mean()) * len(ids),
                        float(
                            balanced_loss(prediction, labels, self.weights, self.names)
                        )
                        * len(ids),
                        float(thermohaline_mse(reconstruction, self.names).mean())
                        * len(ids),
                        float(
                            balanced_loss(
                                initial, initial_truth, self.weights, self.names, True
                            )
                        )
                        * len(ids),
                        len(ids),
                    ]
                )
                p2, t2 = self.moments(prediction, labels, self.weights)
                records.extend(
                    self.origin_records(
                        errors,
                        p2,
                        t2,
                        ids,
                        dataset,
                        "inferred",
                        "global",
                        self.validation_id,
                    )
                )
        self.reduce(totals)
        values = totals[:-1] / totals[-1]
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite validation diagnostics")
        self.append_records(f"validation-origins-rank{self.rank}.csv", records)
        result = dict(
            zip(
                (
                    "ts_mse",
                    "forecast_full_mse",
                    "reconstruction_ts_mse",
                    "reconstruction_full_mse",
                ),
                values.tolist(),
                strict=True,
            )
        )
        result["origins"] = int(totals[-1])
        model.train()
        return result

    def fit(self, model):
        last = self.out / "adapt-last.pt"
        best = self.out / "adapt-best.pt"
        state: AdaptationState = dict(
            epoch=0,
            cursor=0,
            step=0,
            elapsed=0.0,
            best=float("inf"),
            bad_checks=0,
            complete=False,
        )
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.args.learning_rate,
            weight_decay=0.01,
        )
        if last.exists():
            saved = torch.load(last, map_location=self.device, weights_only=False)
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
        if state["complete"]:
            model.load_state_dict(
                torch.load(best, map_location=self.device, weights_only=False)["model"]
            )
            return
        frozen = model.frozen_module()
        frozen_before = state_fingerprint(frozen) if frozen is not None else None
        metrics = self.validate_adaptation(model)
        if not math.isfinite(state["best"]):
            state["best"] = metrics["ts_mse"]
            if self.rank == 0:
                atomic_save({"model": model.state_dict(), "state": state.copy()}, best)
        self.emit(
            {
                "phase": "adapt",
                "event": "start",
                **metrics,
                "best_ts_mse": state["best"],
            }
        )
        wrapped = (
            DDP(model, device_ids=[self.device.index], broadcast_buffers=False)
            if self.world > 1
            else model
        )
        dataset = self.dataset(self.source)
        started = time.monotonic()
        prior = state["elapsed"]
        last_save = started
        done = False
        while not done:
            schedule = training_batches(
                len(dataset),
                self.args.batch_size,
                self.world,
                self.rank,
                self.args.seed,
                state["epoch"],
            )
            remaining = schedule[state["cursor"] :]
            if not remaining:
                state["epoch"] += 1
                state["cursor"] = 0
                continue
            for ids, batch in zip(
                remaining, self.loader(dataset, remaining), strict=True
            ):
                history, forcing, context, labels = self.inputs(batch, dataset, ids)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, initial = wrapped(
                        history,
                        forcing,
                        context,
                        self.mask,
                        "inferred",
                        list(range(1, 7)),
                    )
                    forecast_loss = balanced_loss(
                        prediction, labels, self.weights, self.names
                    )
                    initial_truth = history.reshape(
                        len(ids), 6, self.channels, *history.shape[-2:]
                    )[:, -2:]
                    reconstruction_loss = balanced_loss(
                        initial, initial_truth, self.weights, self.names, True
                    )
                    coefficient = (
                        self.args.reconstruction_weight
                        if self.args.arm != "evolution"
                        else 0.0
                    )
                    loss = forecast_loss + coefficient * reconstruction_loss
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(norm):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
                state["step"] += 1
                state["cursor"] += 1
                now = time.monotonic()
                state["elapsed"] = prior + now - started
                flags = torch.tensor(
                    [
                        state["elapsed"] >= self.args.hours * 3600
                        or bool(
                            self.args.max_steps and state["step"] >= self.args.max_steps
                        ),
                        now - last_save >= self.args.checkpoint_seconds,
                    ],
                    device=self.device,
                    dtype=torch.int32,
                )
                if self.world > 1:
                    dist.broadcast(flags, 0)
                done, checkpoint = map(bool, flags.tolist())
                if state["step"] % 20 == 0:
                    losses = torch.stack(
                        [
                            loss.detach(),
                            forecast_loss.detach(),
                            reconstruction_loss.detach(),
                        ]
                    )
                    self.reduce(losses)
                    self.emit(
                        {
                            "phase": "adapt",
                            "step": state["step"],
                            "elapsed_seconds": state["elapsed"],
                            "loss": float(losses[0] / self.world),
                            "forecast_loss": float(losses[1] / self.world),
                            "reconstruction_loss": float(losses[2] / self.world),
                            "learning_rate": self.args.learning_rate,
                        }
                    )
                if checkpoint or done:
                    metrics = self.validate_adaptation(model)
                    improved = metrics["ts_mse"] < state["best"]
                    state["bad_checks"] = 0 if improved else state["bad_checks"] + 1
                    if improved:
                        state["best"] = metrics["ts_mse"]
                        if self.rank == 0:
                            atomic_save(
                                {"model": model.state_dict(), "state": state.copy()},
                                best,
                            )
                    state["elapsed"] = prior + time.monotonic() - started
                    early = (
                        state["bad_checks"] >= self.args.patience
                        and state["elapsed"] >= self.args.min_hours * 3600
                    )
                    decision = torch.tensor(
                        [done or early, early], device=self.device, dtype=torch.int32
                    )
                    if self.world > 1:
                        dist.broadcast(decision, 0)
                    done, early = map(bool, decision.tolist())
                    state["complete"] = done
                    if (
                        frozen is not None
                        and state_fingerprint(frozen) != frozen_before
                    ):
                        raise RuntimeError(
                            "Frozen weights or batch-normalization buffers changed"
                        )
                    if self.rank == 0:
                        atomic_save(
                            {
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "state": state.copy(),
                            },
                            last,
                        )
                    self.emit(
                        {
                            "phase": "adapt",
                            "step": state["step"],
                            "elapsed_seconds": state["elapsed"],
                            **metrics,
                            "best_ts_mse": state["best"],
                            "bad_checks": state["bad_checks"],
                            "phase_complete": done,
                            "early_stopped": early,
                            "frozen_state_verified": frozen is not None,
                        }
                    )
                    self.barrier()
                    last_save = time.monotonic()
                if done:
                    break
        del wrapped
        model.load_state_dict(
            torch.load(best, map_location=self.device, weights_only=False)["model"]
        )
        if frozen is not None and state_fingerprint(frozen) != frozen_before:
            raise RuntimeError("Best checkpoint changed frozen component")

    def evaluate_adaptation(self, model):
        origin_file = self.out / f"heldout-origins-rank{self.rank}.csv"
        temporary = origin_file.with_suffix(".csv.tmp")
        temporary.unlink(missing_ok=True)
        self.frame_caches.clear()
        torch.cuda.empty_cache()
        self.sync_buffers(model)
        model.eval()
        references = {
            "frozen_pretrain": self.make_model(self.pretrain_path).eval(),
            "wave1_joint": self.make_model(self.wave1_joint_path).eval(),
        }
        references["frozen_pretrain"].initializer.load_state_dict(
            torch.load(
                self.initializer_path, map_location=self.device, weights_only=False
            )["model"]
        )
        for reference in references.values():
            reference.requires_grad_(False)
        climatology = torch.load(
            self.climatology_path, map_location=self.device, weights_only=False
        )["monthly"]
        source = native_om4_source(
            self.bundle.inference_source,
            LocalLocation(path=Path(self.args.data_root) / "OM4.zarr"),
            create_rust_io_runtime(self.args.readers),
        )
        dataset = self.dataset(source)
        indices = list(range(0, len(dataset), 6))
        if self.args.max_steps:
            indices = indices[: self.world * 2]
        sampler = batches(indices[self.rank :: self.world], self.args.batch_size)
        regions = {
            "global": torch.ones_like(self.lat, dtype=torch.bool),
            "tropics": self.lat.abs() <= 20,
            "extratropics": self.lat.abs() > 20,
        }
        modes = ("inferred", "true", "inferred_persistence", "climatology", *references)
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
                    estimates = {
                        "inferred": inferred,
                        "true": truth,
                        "inferred_persistence": initial[:, -1:].expand_as(inferred),
                    }
                    for key, reference in references.items():
                        estimates[key] = reference(
                            history,
                            forcing,
                            context,
                            self.mask,
                            "inferred",
                            list(range(1, 7)),
                        )[0]
                months = [
                    [source.time.values[i + 5 + lead].month - 1 for lead in range(1, 7)]
                    for i in ids
                ]
                estimates["climatology"] = climatology[
                    torch.tensor(months, device=self.device)
                ]
                records = []
                for mi, mode in enumerate(modes):
                    for ri, (region, selection) in enumerate(regions.items()):
                        weights = self.weights * selection[None, :, None]
                        errors = channel_mse(estimates[mode], labels, weights)
                        p2, t2 = self.moments(estimates[mode], labels, weights)
                        sums[mi, ri] += torch.stack((errors, p2, t2), -1).sum(0)
                        records.extend(
                            self.origin_records(
                                errors, p2, t2, ids, dataset, mode, region, "heldout"
                            )
                        )
                self.append_records(temporary.name, records)
                count += len(ids)
        self.reduce(sums)
        self.reduce(count)
        if self.rank == 0:
            values = (sums / count).cpu().numpy()
            if not np.isfinite(values).all():
                raise FloatingPointError("Non-finite final metrics")
            with (self.out / "heldout_metrics.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
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
                std = self.std.cpu().numpy()
                for mi, mode in enumerate(modes):
                    for ri, region in enumerate(regions):
                        for lead in range(6):
                            for ci, channel in enumerate(self.names):
                                mse, p2, t2 = values[mi, ri, lead, ci]
                                rmse = math.sqrt(float(mse))
                                writer.writerow(
                                    [
                                        mode,
                                        region,
                                        (lead + 1) * 5,
                                        channel,
                                        rmse,
                                        rmse * std[ci],
                                        p2,
                                        t2,
                                        int(count.item()),
                                    ]
                                )
        temporary.replace(origin_file)
        self.emit(
            {"event": "heldout_complete", "origins": int(count.item()), "modes": modes}
        )
        self.barrier()

    def execute(self, *, close_group=True):
        model = self.make_model(self.pretrain_path, self.args.arm)
        # Explicitly restore the shared initializer as specified for every arm.
        model.initializer.load_state_dict(
            torch.load(
                self.initializer_path, map_location=self.device, weights_only=False
            )["model"]
        )
        if self.rank == 0:
            provenance = {
                "initial_model_fingerprint": state_fingerprint(model),
                "initial_initializer_fingerprint": state_fingerprint(model.initializer),
                "initial_evolution_fingerprint": state_fingerprint(model.evolution),
                "arm": self.args.arm,
                "data_order_seed": self.args.seed,
                "checkpoint_selection": "mean normalized MSE, equal subsurface T/S and leads 5-30 days",
            }
            for label, path in (
                ("pretrain", self.pretrain_path),
                ("wave1_joint", self.wave1_joint_path),
                ("shared_initializer", self.initializer_path),
            ):
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    while chunk := handle.read(4 * 1024 * 1024):
                        digest.update(chunk)
                provenance[label] = {"path": str(path), "sha256": digest.hexdigest()}
            provenance["protocol"] = {
                "learning_rate": self.args.learning_rate,
                "reconstruction_weight": self.args.reconstruction_weight,
                "batch_size": self.args.batch_size,
                "world_size": self.world,
                "max_steps": self.args.max_steps,
                "val_origins": self.args.val_origins,
            }
            provenance_path = self.out / "initialization.json"
            if (
                provenance_path.exists()
                and json.loads(provenance_path.read_text()) != provenance
            ):
                raise ValueError(
                    "Resume protocol/checkpoint inputs differ from the original arm"
                )
            provenance_path.write_text(json.dumps(provenance, indent=2))
        self.barrier()
        self.fit(model)
        metrics = self.validate_adaptation(model)
        self.emit({"event": "selected_checkpoint_validation", **metrics})
        self.evaluate_adaptation(model)
        if self.rank == 0:
            (self.out / "COMPLETE.json").write_text(
                json.dumps(
                    {
                        "arm": self.args.arm,
                        "seed": self.args.seed,
                        "time_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                        "selected_ts_mse": metrics["ts_mse"],
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
        "--arm", choices=("initializer", "evolution", "joint"), required=True
    )
    parser.add_argument("--wave1-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--data-root", default="/scratch/jr7309/data/om4_onedeg_v3")
    parser.add_argument("--hours", type=float, default=3.5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--widths", type=int, nargs="+", default=[128, 192, 256, 384])
    parser.add_argument("--readers", type=int, default=8)
    parser.add_argument("--device-cache", action="store_true")
    parser.add_argument("--device-cache-reserve-gib", type=float, default=24)
    parser.add_argument("--val-origins", type=int, default=12)
    parser.add_argument("--checkpoint-seconds", type=float, default=1200)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-hours", type=float, default=1)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Qualification only; reduces evaluation origins",
    )
    parser.add_argument(
        "--wandb-mode", choices=("online", "disabled"), default="online"
    )
    args = parser.parse_args()
    if args.hours <= 0 or args.learning_rate <= 0 or args.patience < 1:
        parser.error("hours, learning rate and patience must be positive")
    # Persist rank-local failures even when Slurm/W&B stream capture is incomplete.
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    job = os.environ.get("SLURM_JOB_ID", "local")
    rank = os.environ.get("RANK", "0")
    with (directory / f"error-{job}-rank{rank}.log").open("a") as errors:
        faulthandler.enable(file=errors, all_threads=True)
        try:
            Adaptation(args).execute()
        except Exception:
            traceback.print_exc(file=errors)
            errors.flush()
            raise
        finally:
            faulthandler.disable()


if __name__ == "__main__":
    main()
