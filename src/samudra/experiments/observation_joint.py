# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""One optimizer, exact task exposure, and resumable OM4/observation scheduling."""

import copy
import datetime
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.initializer_wave import InitializerWave, Pair
from samudra.experiments.initializer_wave import build_parser as om4_parser
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_pilot import (
    Pilot,
    atomic_json,
    atomic_torch,
    build_parser,
    digest,
)
from samudra.experiments.surface_state import advance_season, balanced_loss
from samudra.experiments.task_schedule import TaskSchedule, sample_indices


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def assert_exact_state(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError("Serialized state keys differ")
        for key in expected:
            assert_exact_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise ValueError("Serialized state length differs")
        for a, b in zip(actual, expected, strict=True):
            assert_exact_state(a, b)
    elif actual != expected:
        raise ValueError("Serialized state value differs")


def parameter_difference(actual, expected):
    squared, count, largest = 0.0, 0, 0.0
    for name, value in actual.items():
        delta = value.detach().cpu().float() - expected[name].float()
        if not torch.isfinite(delta).all():
            raise FloatingPointError("Nonfinite replay difference")
        squared += float(delta.square().sum(dtype=torch.float64))
        count += delta.numel()
        largest = max(largest, float(delta.abs().max()))
    return {"rmse": math.sqrt(squared / count), "max_absolute": largest}


def om4_objective(model, data, ids, reconstruction_weight):
    surface, past, context, truth, forcing, labels = data.model_sample(
        data.trainset, ids
    )
    # OM4 uses its original forcings directly. The ERA5 adapter is used only by
    # observational samples; no fake adapter updates on the OM4 task.
    initial = model.call(model.initializer, surface, past, context, data.mask)
    states, predictions = initial, []
    for lead in range(1, 7):
        predicted = model.call(
            model.evolution,
            states,
            forcing,
            advance_season(context, (lead - 1) * 5),
            data.mask,
            lead,
        )
        predictions.append(predicted)
        states = torch.stack((states[:, -1], predicted), 1)
    return balanced_loss(
        torch.stack(predictions, 1), labels, data.weights, data.names
    ) + reconstruction_weight * balanced_loss(
        initial, truth, data.weights, data.names, True
    )


def qualification_contract(args):
    return {
        "code_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
        "data_manifest_sha256": digest(Path(args.data) / "SHA256SUMS"),
        "om4_data": args.om4_data,
        "selection_reference_sha256": digest(args.selection_reference),
        "normalization": args.normalization,
        "evolution_architecture": args.evolution_architecture,
        "reconstruction_weight": args.reconstruction_weight,
        "accumulate": args.accumulate,
        "core_lr": args.core_lr,
        "om4_lr": args.om4_lr,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
    }


def load_om4(args):
    options = om4_parser().parse_args(
        [
            "--arm",
            "D",
            "--phase",
            "joint",
            "--fresh-evolution",
            "--normalization",
            args.normalization,
            "--evolution-architecture",
            args.evolution_architecture,
            "--observation-normalization-root",
            args.data,
            "--data-root",
            args.om4_data,
            "--output",
            str(Path(args.output) / "om4-loader"),
            "--name",
            args.name + "-loader",
            "--wandb-mode",
            "disabled",
            "--batch-size",
            "1",
            "--val-origins",
            str(args.om4_validation_origins),
            "--seed",
            str(args.seed),
            "--readers",
            "4",
        ]
    )
    data = InitializerWave(options)
    if data.run:
        data.run.finish()
    return data


class JointPilot(Pilot):
    def __init__(self, args):
        self.schedule = TaskSchedule(args.om4_updates, args.joint_steps, args.ordering)
        self.deadline = datetime.datetime.fromisoformat(
            args.deadline.replace("Z", "+00:00")
        )
        if datetime.datetime.now(datetime.UTC) >= self.deadline:
            raise ValueError("Training deadline passed")
        if not args.joint_probe:
            if not args.joint_qualification:
                raise ValueError(
                    "Production needs a completed single-loop qualification"
                )
            qualified = json.loads(Path(args.joint_qualification).read_text())
            if qualified["contract"] != qualification_contract(
                args
            ) or not qualified.get("resume_verified"):
                raise ValueError(
                    "Single-loop qualification does not match producer/config"
                )
        # Construct loader before Pilot so every arm starts from exactly the same
        # seed/model; the loader's temporary model is then discarded.
        self.om4 = load_om4(args)
        super().__init__(args)
        if list(self.om4.names) != list(self.data.grid["names"]):
            raise ValueError("Task channel order differs")
        torch.testing.assert_close(self.om4.mean, self.data.mean[0, 0, :, 0, 0])
        torch.testing.assert_close(self.om4.std, self.data.std[0, 0, :, 0, 0])
        if self.om4.mask.shape != self.data.mask.shape:
            raise ValueError("Task grid shapes differ")
        # Physical domains can use distinct wet masks; coordinate arrays must
        # still coincide. Shape alone would not establish grid alignment.
        np.testing.assert_allclose(
            self.om4.lat.cpu().numpy(), self.data.grid["lat"], atol=1e-5, rtol=0
        )
        longitude = self.om4.source.resolution[1].cpu().numpy()
        np.testing.assert_allclose(longitude, self.data.grid["lon"], atol=1e-5, rtol=0)
        self.om4.initializer = self.model.initializer
        self.om4.model = Pair(self.model.initializer, self.model.evolution, joint=True)
        torch.cuda.empty_cache()
        self.model.set_phase("joint")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.core_lr, weight_decay=0.01
        )
        self.completed = 0
        self.elapsed_seconds = 0.0
        self.resume = self.out / "joint-last.pt"
        if self.resume.exists():
            saved = torch.load(
                self.resume, map_location=self.device, weights_only=False
            )
            if saved["manifest"] != self.manifest:
                raise ValueError("Resume contract differs")
            self.model.load_state_dict(saved["model"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.completed = saved["global_step"]
            if saved["task_counts"] != self.schedule.counts(self.completed):
                raise ValueError("Resume task counts differ")
            torch.set_rng_state(saved["rng_cpu"].cpu())
            torch.cuda.set_rng_state(saved["rng_cuda"].cpu())
            np.random.set_state(saved["rng_numpy"])
            self.elapsed_seconds = saved["elapsed_seconds"]
        if self.schedule.om4_updates:
            self.om4.prepare(self.om4.trainset)
        self.om4.prepare(self.om4.valset)
        self.started = time.monotonic()

    def checkpoint(self, path):
        atomic_torch(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "manifest": self.manifest,
                "global_step": self.completed,
                "step": self.schedule.counts(self.completed)["observation"],
                "phase": "joint",
                "task_counts": self.schedule.counts(self.completed),
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "elapsed_seconds": self.elapsed_seconds
                + time.monotonic()
                - self.started,
            },
            path,
        )

    def validate(self):
        metrics = self.evaluate(self.validation)
        score = selection_score(metrics, self.control, self.spectral_keys)
        counts = self.schedule.counts(self.completed)
        if score < self.global_best:
            self.save_best(score, "joint", counts["observation"], metrics)
        retention = self.om4.score(self.om4.valset, self.om4.val_ids, forecast=True)
        self.emit(
            {
                "event": "joint_validation",
                "global_step": self.completed,
                **counts,
                "validation/obs_score": score,
                "om4_retention": retention,
            }
        )
        atomic_json(
            {
                "metrics": metrics,
                "om4_retention": retention,
                "task_counts": counts,
                "global_step": self.completed,
                "score": score,
            },
            self.out / f"validation-{self.completed:05d}.json",
        )
        self.model.train()

    def train_update(self):
        task = self.schedule.task(self.completed)
        count = self.schedule.counts(self.completed)[task]
        seed = self.args.seed + (0 if task == "observation" else 100000)
        size = len(self.training) if task == "observation" else len(self.om4.trainset)
        ids = sample_indices(size, seed, count, self.args.accumulate)
        lr = self.args.core_lr if task == "observation" else self.args.om4_lr
        warmup = self.args.warmup_steps
        lr *= min(1.0, (count + 1) / warmup) if warmup else 1.0
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for index in ids:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if task == "observation":
                    loss = self.objective(self.data.load(self.training[index]), "joint")
                else:
                    loss = om4_objective(
                        self.model, self.om4, [index], self.args.reconstruction_weight
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite {task} loss")
                (loss / self.args.accumulate).backward()
            loss_sum += float(loss.detach()) / self.args.accumulate
        reached = {}
        for name in ("initializer", "evolution", "adapter"):
            reached[name] = any(
                p.grad is not None and bool(p.grad.count_nonzero())
                for p in getattr(self.model, name).parameters()
            )
        if (
            not reached["initializer"]
            or not reached["evolution"]
            or (task == "observation" and not reached["adapter"])
        ):
            raise ValueError(f"Missing {task} gradients: {reached}")
        if task == "om4" and reached["adapter"]:
            raise ValueError("OM4 unexpectedly trained ERA5 adapter")
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        if not torch.isfinite(norm) or not math.isfinite(loss_sum):
            raise FloatingPointError("Nonfinite joint gradient")
        self.optimizer.step()
        self.completed += 1
        self.emit(
            {
                "event": "joint_train",
                "global_step": self.completed,
                **self.schedule.counts(self.completed),
                "task": task,
                "loss": loss_sum,
                "lr": lr,
                "gradient_norm": float(norm),
                "elapsed_seconds": self.elapsed_seconds
                + time.monotonic()
                - self.started,
            }
        )

    def verify_resume_update(self):
        serialized = torch.load(self.resume, map_location="cpu", weights_only=False)
        native = {
            "model": cpu_copy(self.model.state_dict()),
            "optimizer": cpu_copy(self.optimizer.state_dict()),
            "global_step": self.completed,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "rng_numpy": np.random.get_state(),
        }
        # Check serialization and restoration exactly. GPU interpolation backward
        # can be nondeterministic, so separately measure native replay variation.
        for key, value in native.items():
            assert_exact_state(value, serialized[key])
        expected = None
        differences = {}
        for label, snapshot in (
            ("native", native),
            ("native_repeat", native),
            ("serialized", serialized),
        ):
            saved = cpu_copy(snapshot)
            self.model.load_state_dict(saved["model"], strict=True)
            self.optimizer.load_state_dict(saved["optimizer"])
            self.completed = saved["global_step"]
            torch.set_rng_state(saved["rng_cpu"])
            torch.cuda.set_rng_state(saved["rng_cuda"])
            np.random.set_state(saved["rng_numpy"])
            assert_exact_state(self.model.state_dict(), snapshot["model"])
            assert_exact_state(self.optimizer.state_dict(), snapshot["optimizer"])
            assert_exact_state(torch.get_rng_state(), snapshot["rng_cpu"])
            assert_exact_state(torch.cuda.get_rng_state(), snapshot["rng_cuda"])
            self.train_update()
            if expected is None:
                expected = cpu_copy(self.model.state_dict())
            else:
                differences[label] = parameter_difference(
                    self.model.state_dict(), expected
                )
        for metric, floor in (("rmse", 1e-8), ("max_absolute", 1e-7)):
            limit = max(floor, 5 * differences["native_repeat"][metric])
            if differences["serialized"][metric] > limit:
                raise ValueError(
                    f"Serialized replay exceeds measured native variation: {differences}"
                )
        self.resume_evidence = {
            "serialization_exact": True,
            "restoration_exact": True,
            "native_and_serialized_replay": differences,
            "acceptance": "Serialized replay difference <= 5x native replay difference, with RMSE floor 1e-8 and maximum absolute floor 1e-7",
        }
        self.emit(
            {
                "event": "resume_qualified",
                "global_step": self.completed,
                **self.resume_evidence,
            }
        )

    def save_best(self, value, phase, step, metrics):
        self.global_best = value
        counts = self.schedule.counts(self.completed)
        metadata = {
            "score": value,
            "phase": phase,
            "step": step,
            "global_step": self.completed,
            "task_counts": counts,
        }
        atomic_torch(
            {"model": self.model.state_dict(), **metadata}, self.out / "best.pt"
        )
        atomic_json(
            {
                **metadata,
                "metrics": metrics,
                "checkpoint_sha256": digest(self.out / "best.pt"),
            },
            self.out / "best.json",
        )

    def run_joint(self):
        try:
            self.qualify_selection()
            if not self.resume.exists():
                self.checkpoint(self.out / "joint-00000.pt")
            while self.completed < self.schedule.total:
                if (
                    datetime.datetime.now(datetime.UTC) >= self.deadline
                    or self.elapsed_seconds + time.monotonic() - self.started
                    >= self.args.joint_hours * 3600
                ):
                    self.checkpoint(self.resume)
                    atomic_json(
                        {
                            "global_step": self.completed,
                            "task_counts": self.schedule.counts(self.completed),
                            "reason": "wall-time limit",
                        },
                        self.out / "TRAIN_PARTIAL.json",
                    )
                    return
                task = self.schedule.task(self.completed)
                self.train_update()
                obs_count = self.schedule.counts(self.completed)["observation"]
                milestone = (
                    task == "observation" and obs_count in self.args.milestone_steps
                )
                if (
                    self.completed % self.args.validate_every == 0
                    or milestone
                    or self.completed == self.schedule.total
                ):
                    self.validate()
                    self.checkpoint(self.resume)
                if milestone:
                    self.checkpoint(self.out / f"joint-{obs_count:05d}.pt")
                if self.args.joint_probe and self.completed == 4:
                    self.checkpoint(self.resume)
                    # In-process disk round-trip tests the same model/optimizer
                    # serialization used on restart. Compare the next full update
                    # from uninterrupted and restored states, with optimizer moments restored.
                    self.verify_resume_update()
            if self.args.joint_probe:
                atomic_json(
                    {
                        "contract": qualification_contract(self.args),
                        "resume_verified": True,
                        "resume_evidence": self.resume_evidence,
                        "task_counts": self.schedule.counts(self.completed),
                        "gpu": torch.cuda.get_device_name(),
                        "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                    },
                    self.out / "JOINT_QUALIFIED.json",
                )
            atomic_json(
                {
                    "global_step": self.completed,
                    "task_counts": self.schedule.counts(self.completed),
                    "best_score": self.global_best,
                    "best_checkpoint_sha256": digest(self.out / "best.pt"),
                    "completed_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                },
                self.out / "TRAIN_COMPLETE.json",
            )
        finally:
            if self.wandb:
                self.wandb.finish()


def main():
    parser = build_parser()
    parser.add_argument(
        "--ordering", choices=["scratch", "sequential", "mixed"], required=True
    )
    parser.add_argument("--om4-updates", type=int, required=True)
    parser.add_argument("--om4-lr", type=float, required=True)
    parser.add_argument("--om4-data", default="/scratch/jr7309/data/om4_onedeg_v3")
    parser.add_argument("--om4-validation-origins", type=int, default=12)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--joint-probe", action="store_true")
    parser.add_argument("--joint-qualification")
    parser.set_defaults(
        from_scratch=True,
        observation_normalization=True,
        normalization="instance",
        evolution_architecture="samudra2",
        reconstruction_steps=0,
        adapter_steps=0,
        fixed_updates=True,
    )
    args = parser.parse_args()
    if (
        not args.from_scratch
        or args.normalization != "instance"
        or args.fit_probe
        or args.accumulate < 1
        or args.validate_every < 1
        or args.joint_hours <= 0
        or args.core_lr is None
        or args.core_lr <= 0
        or args.om4_lr <= 0
        or args.reconstruction_steps
        or args.adapter_steps
        or args.adapter_only
    ):
        parser.error(
            "Single-loop training needs qualified fresh InstanceNorm, positive rates/counts and no separate phases"
        )
    if args.joint_probe and (
        args.ordering != "mixed" or args.om4_updates != 6 or args.joint_steps != 4
    ):
        parser.error(
            "Single-loop probe uses exactly six OM4 and four observation updates"
        )
    if int(os.environ.get("WORLD_SIZE", 1)) != 1:
        parser.error(
            "This qualified deterministic sampler currently supports one GPU per arm"
        )
    JointPilot(args).run_joint()


if __name__ == "__main__":
    main()
