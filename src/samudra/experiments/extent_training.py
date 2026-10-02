# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Native-quarter regional tasks within the existing observation joint loop."""

import time

import torch

from samudra.experiments.extent_data import PatchSamples
from samudra.experiments.missingness import completion_loss, structured_visibility
from samudra.experiments.observation_joint import JointPilot
from samudra.experiments.observation_pilot import atomic_json
from samudra.experiments.surface_state import advance_season, balanced_loss


def patch_objective(
    model, sample, names, reconstruction_weight, completion_weight, seed
):
    surface, mask = sample["surface"], sample["mask"]
    available = mask[model.initializer.surface].expand_as(surface)
    visible = structured_visibility(available, seed)
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
    for lead in range(1, 7):
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
        torch.stack(predictions, 1), sample["labels"], sample["weights"], names
    )
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
    def __init__(self, args):
        super().__init__(args)
        self.patches = (
            PatchSamples(args.patch_cache, self.om4) if args.patch_training else None
        )
        self.update_seconds = []
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

    def om4_objective(self, *args):
        count = self.schedule.counts(self.completed)["om4"]
        if self.patches is None or count % 2 == 0:
            return super().om4_objective(*args)
        seed = args[4]
        if seed is None:
            raise ValueError("Patch tasks need deterministic sample seeds")
        sample = self.patches.sample(seed)
        return patch_objective(
            self.model,
            sample,
            self.om4.names,
            self.args.reconstruction_weight,
            self.args.completion_weight,
            seed,
        )

    def train_update(self):
        start = time.monotonic()
        super().train_update()
        elapsed = time.monotonic() - start
        self.update_seconds.append(elapsed)

    def emit(self, record):
        if record.get("event") == "joint_train":
            record = dict(record)
            if self.args.patch_training:
                record["om4_global"] = (record["om4"] + 1) // 2
                record["om4_patch"] = record["om4"] // 2
            else:
                record["om4_global"], record["om4_patch"] = record["om4"], 0
            if record["task"] == "om4":
                record["source_extent"] = (
                    "patch"
                    if self.args.patch_training and record["om4"] % 2 == 0
                    else "global"
                )
        super().emit(record)

    def run_joint(self):
        super().run_joint()
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
