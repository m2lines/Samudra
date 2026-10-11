# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Native-quarter regional tasks within the existing observation joint loop."""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import torch

from samudra.experiments.extent_data import PatchSamples
from samudra.experiments.missingness import completion_loss, structured_visibility
from samudra.experiments.observation_joint import JointPilot, om4_objective
from samudra.experiments.observation_pilot import atomic_json
from samudra.experiments.surface_state import advance_season, balanced_loss


def patch_objective(
    model,
    sample,
    names,
    reconstruction_weight,
    completion_weight,
    seed,
    *,
    mode="shared",
    leads=6,
):
    if mode not in {"shared", "truth", "detach", "forecast"} or leads not in {1, 6}:
        raise ValueError(
            "Patch objective requires shared/truth/detach/forecast initialization and 1/6 leads"
        )
    surface, mask = sample["surface"], sample["mask"]
    available = mask[model.initializer.surface].expand_as(surface)
    visible = structured_visibility(available, seed)
    if mode == "truth":
        initial = sample["truth"].detach()
    else:
        with torch.no_grad() if mode == "detach" else nullcontext():
            initial = model.call(
                model.initializer,
                torch.where(visible, surface, 0),
                sample["past"],
                sample["context"],
                mask,
                visible,
                "om4-patch",
            )
    states, predictions = initial, []
    for lead in range(1, leads + 1):
        value = model.call(
            model.evolution,
            states,
            sample["forcing"],
            advance_season(sample["context"], (lead - 1) * 5),
            mask,
            lead,
            "om4-patch",
        )
        predictions.append(value)
        states = torch.stack((states[:, -1], value), 1)
    loss = balanced_loss(
        torch.stack(predictions, 1),
        sample["labels"][:, :leads],
        sample["weights"],
        names,
    )
    if mode in {"truth", "detach", "forecast"}:
        return loss
    loss = loss + reconstruction_weight * balanced_loss(
        initial, sample["truth"], sample["weights"], names, True
    )
    hidden = (
        available[:, -2:] & ~visible[:, -2:] & sample["core"][model.initializer.surface]
    )
    loss = loss + completion_weight * completion_loss(
        initial[:, :, model.initializer.surface], surface[:, -2:], hidden, sample["lat"]
    )
    return loss


class ExtentPilot(JointPilot):
    def verify_resume_update(self):
        if self.accessory is None:
            return super().verify_resume_update()
        from samudra.experiments.early_fine_training import deterministic_replay

        with deterministic_replay():
            super().verify_resume_update()
        differences = self.resume_evidence["native_and_serialized_replay"]
        if not isinstance(differences, dict):
            raise ValueError("Missing replay difference records")
        if any(value != 0 for part in differences.values() for value in part.values()):
            raise ValueError("Auxiliary deterministic reference replay was not exact")
        self.resume_evidence["deterministic_reference_bitwise_exact"] = True

    def __init__(self, args):
        super().__init__(args)
        self.patches = (
            PatchSamples(args.patch_cache, self.om4) if args.patch_training else None
        )
        self.update_seconds = []
        self.accessory = None
        if getattr(args, "auxiliary_cache", None):
            from samudra.experiments.extent_accessory import VarianceTargets

            self.accessory = VarianceTargets(
                args.auxiliary_cache,
                self.om4,
                args.auxiliary_weight,
                getattr(args, "auxiliary_target_mode", "aligned"),
            )
            if not args.joint_probe:
                qualified_targets = json.loads(
                    Path(args.joint_qualification)
                    .with_name("ACCESSORY_TARGETS.json")
                    .read_text()
                )
                if self.accessory.provenance != qualified_targets:
                    raise ValueError(
                        "Production accessory targets differ from qualification"
                    )
            atomic_json(
                self.accessory.provenance, Path(args.output) / "ACCESSORY_TARGETS.json"
            )
            # Same byte-equivalent observation preparation used by the matched
            # U-global RMSE reproductions; only CPU I/O/preparation is cached.
            self.data.cache_prepared = True
            if args.joint_probe:
                for path in [self.training[0], self.validation[0]]:
                    self.data.cache_prepared = False
                    reference = self.data.load(path)
                    self.data.cache_prepared = True
                    cached = self.data.load(path)
                    for name, value in reference.items():
                        if isinstance(value, torch.Tensor):
                            torch.testing.assert_close(
                                cached[name], value, rtol=0, atol=0, equal_nan=True
                            )
                    torch.testing.assert_close(
                        self.data.load_coverage(path),
                        reference["validity"][:, :19],
                        rtol=0,
                        atol=0,
                    )
                atomic_json(
                    dict(
                        observation_prepared_and_coverage_exact=True,
                        producer=self.manifest["code_commit"],
                    ),
                    self.out / "IO_EQUIVALENT.json",
                )
            self.observation_warm_pool = ThreadPoolExecutor(max_workers=2)
            self.observation_warm = [
                self.observation_warm_pool.submit(self.data.prepared_sample, path)
                for path in [*self.training, *self.validation]
            ]
        counts = {
            name: sum(p.numel() for p in getattr(self.model, name).parameters())
            for name in ("initializer", "evolution", "adapter")
        }
        self.emit(
            dict(
                event="extent_ready",
                parameters=counts,
                total_parameters=sum(counts.values()),
                patch_training=args.patch_training,
                state_channels=77,
                global_shape=[180, 360],
                patch_shape=[128, 128],
                scored_patch_shape=[64, 64],
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
        if self.patches is None or count % 2 == 0:
            return om4_objective(
                model,
                data,
                ids,
                reconstruction_weight,
                mask_seed,
                coverage,
                completion_weight,
                self.accessory,
            )
        seed = mask_seed
        if seed is None:
            raise ValueError("Patch tasks need deterministic sample seeds")
        sample = self.patches.sample(seed)
        return getattr(self.args, "patch_loss_scale", 1.0) * patch_objective(
            self.model,
            sample,
            self.om4.names,
            self.args.reconstruction_weight,
            self.args.completion_weight,
            seed,
            mode=getattr(self.args, "patch_mode", "shared"),
            leads=getattr(self.args, "patch_leads", 6),
        )

    def is_patch_slot(self):
        return (
            self.args.patch_training
            and self.schedule.task(self.completed) == "om4"
            and self.schedule.counts(self.completed)["om4"] % 2 == 1
        )

    def task_learning_rate(self, task):
        lr = super().task_learning_rate(task)
        return (
            lr * getattr(self.args, "patch_lr_scale", 1.0)
            if self.is_patch_slot()
            else lr
        )

    def required_gradient_components(self, task):
        if self.is_patch_slot() and getattr(self.args, "patch_mode", "shared") in {
            "truth",
            "detach",
        }:
            return {"evolution"}
        return super().required_gradient_components(task)

    def train_update(self):
        accessory = getattr(self, "accessory", None)
        if accessory is not None:
            for future in self.observation_warm:
                if future.done():
                    future.result()
            accessory.loss_records.clear()
        if (
            self.is_patch_slot()
            and getattr(self.args, "patch_mode", "shared") == "omit"
        ):
            # No forward, RNG draw, optimizer, momentum, decay or learning-rate change.
            self.completed += 1
            self.emit(
                dict(
                    event="joint_skip",
                    global_step=self.completed,
                    **self.schedule.counts(self.completed),
                    task="om4",
                    reason="omitted auxiliary patch slot",
                    elapsed_seconds=self.elapsed_seconds
                    + time.monotonic()
                    - self.started,
                )
            )
            return

        start = time.monotonic()
        super().train_update()
        elapsed = time.monotonic() - start
        self.update_seconds.append(elapsed)
        if accessory is not None and self.completed % 10 == 0:
            self.checkpoint(self.resume)

    def observation_coverage(self, path):
        if self.accessory is not None:
            return self.data.load_coverage(path)
        return super().observation_coverage(path)

    def emit(self, record):
        if record.get("event") in {"joint_train", "joint_skip"}:
            record = dict(record)
            accessory = getattr(self, "accessory", None)
            if accessory is not None and accessory.loss_records:
                extra = float(torch.stack(accessory.loss_records).mean())
                record["accessory_mse"] = extra
                record["physical_objective"] = (
                    record["loss"] - accessory.coefficient * extra
                )
            if self.args.patch_training:
                record["om4_global"] = (record["om4"] + 1) // 2
                auxiliary = record["om4"] // 2
                omitted = getattr(self.args, "patch_mode", "shared") == "omit"
                record["om4_patch"] = 0 if omitted else auxiliary
                record["om4_omitted"] = auxiliary if omitted else 0
            else:
                record["om4_global"], record["om4_patch"] = record["om4"], 0
            record["optimizer_updates"] = record["global_step"] - record.get(
                "om4_omitted", 0
            )
            if record["task"] == "om4":
                record["source_extent"] = (
                    "patch"
                    if self.args.patch_training and record["om4"] % 2 == 0
                    else "global"
                )
                if (
                    record["source_extent"] == "patch"
                    and record["event"] == "joint_train"
                ):
                    record["patch_mode"] = getattr(self.args, "patch_mode", "shared")
                    scale = getattr(self.args, "patch_loss_scale", 1.0)
                    record["patch_loss_scale"] = scale
                    record["unscaled_patch_loss"] = record["loss"] / scale
        super().emit(record)

    def run_joint(self):
        try:
            super().run_joint()
        finally:
            if self.accessory is not None:
                self.observation_warm_pool.shutdown(wait=True, cancel_futures=True)
        counts = self.schedule.counts(self.completed)
        auxiliary = counts["om4"] // 2 if self.args.patch_training else 0
        omitted = (
            auxiliary if getattr(self.args, "patch_mode", "shared") == "omit" else 0
        )
        atomic_json(
            dict(
                schedule_slots=self.completed,
                optimizer_updates=self.completed - omitted,
                om4_global=counts["om4"] - auxiliary,
                om4_patch=auxiliary - omitted,
                omitted_patch_slots=omitted,
                observation=counts["observation"],
            ),
            self.out / "EXPOSURE.json",
        )
        if self.args.joint_probe:
            if not (self.out / "JOINT_QUALIFIED.json").exists():
                raise RuntimeError("Probe stopped without qualification")
            # Timings include real sampling, backward, optimizer and replay updates.
            seconds = sum(self.update_seconds) / len(self.update_seconds)
            atomic_json(
                dict(
                    mean_update_seconds=seconds,
                    updates=len(self.update_seconds),
                    training_hours_for_4000_updates=seconds * 4000 / 3600,
                    note="Excludes validation and initial dataset preparation",
                ),
                self.out / "THROUGHPUT.json",
            )
