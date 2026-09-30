# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed 8k OM4 + 8k observation reproduction harness."""

import argparse
import hashlib
import json
import math
import os
import signal
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from samudra.config import DataConfig
from samudra.config_base import resolve_config_path
from samudra.models.surface_initialized import SurfaceInitializedConfig, advance_season
from samudra.observations.annual import evaluate_year
from samudra.observations.archive import SPLITS
from samudra.observations.evaluate import Evaluator
from samudra.observations.losses import (
    balanced_loss,
    channel_mse,
    completion_loss,
    corrupt_sample,
    om4_objective,
)
from samudra.observations.metrics import PROTOCOL, selection_score
from samudra.observations.om4 import Om4Samples
from samudra.observations.samples import Samples
from samudra.utils.wandb import WandBLogger

# The scientific plan is deliberately plain code, not a configurable task framework.
SEED, TOTAL_UPDATES, ACCUMULATE, VALIDATE_EVERY = 1729, 16000, 8, 100
LEARNING_RATE, RECONSTRUCTION_WEIGHT, COMPLETION_WEIGHT = 1e-4, 0.1, 0.1


def task_counts(completed):
    if not 0 <= completed <= TOTAL_UPDATES:
        raise ValueError("Update outside the fixed plan")
    observation = (
        8000 * completed * (TOTAL_UPDATES + 3 * completed) // (4 * TOTAL_UPDATES**2)
    )
    return {"om4": completed - observation, "observation": observation}


@lru_cache(maxsize=8)
def permutation(size, seed, epoch):
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(
        size
    )


def sample_indices(size, seed, count):
    return [
        int(permutation(size, seed, p // size)[p % size])
        for p in range(count * ACCUMULATE, (count + 1) * ACCUMULATE)
    ]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(value, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def flat_metrics(value, prefix=""):
    """Nested scalar metrics retain full names; spectral arrays stay in JSON artifacts."""
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flat_metrics(item, name))
        elif isinstance(item, (int, float, np.number)) and not isinstance(item, bool):
            if not np.isfinite(item):
                raise ValueError(f"Nonfinite metric: {name}")
            result[name] = float(item)
    return result


def observation_objective(model, data, sample, mask_seed):
    sample = corrupt_sample(sample, mask_seed)
    arguments = (
        sample["surface"],
        sample["atmosphere"],
        sample["contexts"],
        data.mask,
        sample["validity"],
    )
    predicted, initial = model.forecast(*arguments)
    components = data.forecast_losses(predicted, sample)
    components["completion"] = completion_loss(
        initial[:, :, model.initializer.surface],
        sample["completion_target"],
        sample["completion_valid"],
        data.grid["lat"],
    )
    reconstructed = model.reconstruct_month(*arguments, sample["month_weights"])
    components["reconstruction"] = data.interior_loss(reconstructed, sample)
    loss = (
        0.8 * components["monthly_ts"]
        + 0.1 * components["sst"]
        + 0.1 * components["ssh"]
        + COMPLETION_WEIGHT * components["completion"]
        + RECONSTRUCTION_WEIGHT * components["reconstruction"]
    )
    return loss, components


@torch.no_grad()
def retention(model, om4):
    model.eval()
    totals = []
    for index in om4.val_ids:
        surface, past, context, truth, forcing, labels = om4.model_sample(
            om4.valset, [index]
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            initial = model.initializer(surface, past, context, om4.mask, task="om4")
            states = initial
            outputs = []
            for lead in range(1, 7):
                predicted = model.evolution(
                    states,
                    forcing,
                    advance_season(context, (lead - 1) * 5),
                    om4.mask,
                    lead,
                    "om4",
                )
                outputs.append(predicted)
                states = torch.stack((states[:, -1], predicted), 1)
            predicted = torch.stack(outputs, 1)

        def ts(pred, target):
            error = channel_mse(pred, target, om4.weights)
            return torch.stack(
                [
                    error[
                        ...,
                        [
                            i
                            for i, n in enumerate(om4.names)
                            if n.startswith(v + "_") and n != "thetao_0"
                        ],
                    ].mean()
                    for v in ["thetao", "so"]
                ]
            ).mean()

        totals.append(
            [
                float(ts(predicted, labels)),
                float(ts(initial, truth)),
                float(balanced_loss(predicted, labels, om4.weights, om4.names)),
            ]
        )
    values = np.mean(totals, axis=0)
    return dict(
        zip(
            ["ts_mse", "reconstruction_ts_mse", "full_mse"],
            map(float, values),
            strict=True,
        ),
        origins=len(totals),
    )


class Harness:
    def __init__(self, args):
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("This harness is single-GPU")
        self.args = args
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda", 0)
        torch.cuda.set_device(self.device)
        torch.set_num_threads(1)
        self.data = Samples(args.observations, self.device)
        self.data.use_observation_normalization()
        self.training = self.data.paths("train")
        self.validation = self.data.paths("validation")
        for split in SPLITS:
            expected = [str(m) for m in pd.period_range(*SPLITS[split], freq="M")]
            if [p.stem for p in self.data.paths(split)] != expected:
                raise ValueError(f"Incomplete {split} cohort")
        model_config = SurfaceInitializedConfig.model_validate(
            yaml.safe_load(resolve_config_path(args.model).read_text())
        )
        om4_config = DataConfig.model_validate(
            yaml.safe_load(resolve_config_path(args.om4_config).read_text())
        )
        self.om4 = (
            Om4Samples(args.om4, om4_config, self.data, self.device, args.cache_device)
            if args.command == "train"
            else None
        )
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        self.model = model_config.build(self.data.grid["names"].tolist()).to(
            self.device
        )
        self.evaluator = Evaluator(self.model, self.data)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=0.01
        )
        self.completed = 0
        self.best = math.inf
        self.stop = False
        manifest = {
            "model": model_config.model_dump(mode="json"),
            "om4": om4_config.model_dump(mode="json"),
            "observation_files": {
                str(p.relative_to(args.observations)): digest(p)
                for p in sorted(args.observations.rglob("*.npz"))
            },
            "protocol": PROTOCOL,
            "plan": {
                "seed": SEED,
                "updates": TOTAL_UPDATES,
                "accumulate": ACCUMULATE,
                "learning_rate": LEARNING_RATE,
                "reconstruction": RECONSTRUCTION_WEIGHT,
                "completion": COMPLETION_WEIGHT,
            },
            "source_sha256": {
                str(p.relative_to(Path(__file__).parents[1])): digest(p)
                for p in [
                    Path(__file__),
                    Path(__file__).parents[1] / "models/surface_initialized.py",
                    *sorted((Path(__file__).parents[1] / "observations").glob("*.py")),
                ]
            },
            "om4_root": str(args.om4.resolve()),
            "om4_metadata": {
                str(p.relative_to(args.om4)): digest(p)
                for p in sorted(args.om4.glob("*/.zmetadata"))
            },
        }
        self.manifest = manifest
        if (self.out / "manifest.json").exists() and json.loads(
            (self.out / "manifest.json").read_text()
        ) != manifest:
            raise ValueError("Run contract differs; choose a new output directory")
        atomic_json(manifest, self.out / "manifest.json")
        self.log = WandBLogger()
        self.log.configure(args.wandb_mode != "disabled", True)
        self.log.init(
            entity=args.entity,
            project=args.project,
            name=args.name,
            id=hashlib.sha256(str(self.out.resolve()).encode()).hexdigest()[:12],
            resume="allow",
            dir=str(self.out),
            mode=args.wandb_mode,
            config=manifest,
        )
        if args.wandb_mode != "disabled" and not self.log.enabled:
            raise RuntimeError("Requested W&B initialization failed")
        if self.log.enabled:
            assert self.log.run is not None
            self.log.run.define_metric("updates/total")
            self.log.run.define_metric("*", step_metric="updates/total")
        self.reference_path = self.out / "selection-reference.json"
        self.control = None
        self.spectral_keys = []

    def emit(self, event, values):
        record = {
            "event": event,
            "updates": {"total": self.completed, **task_counts(self.completed)},
            **values,
        }
        with (self.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        self.log.log(flat_metrics(record), step=None)

    def checkpoint(self, path):
        atomic_torch(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "manifest": self.manifest,
                "global_step": self.completed,
                "task_counts": task_counts(self.completed),
                "best_score": self.best,
                "rng_cpu": torch.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
                "rng_numpy": np.random.get_state(),
                "selection_reference_sha256": digest(self.reference_path)
                if getattr(self, "reference_path", None)
                and self.reference_path.exists()
                else None,
            },
            path,
        )

    def restore(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved["manifest"] != self.manifest:
            raise ValueError("Checkpoint contract differs")
        expected_reference = saved.get("selection_reference_sha256")
        if expected_reference is not None and (
            not self.reference_path.exists()
            or digest(self.reference_path) != expected_reference
        ):
            raise ValueError("Selection reference differs from checkpoint")
        self.completed = saved["global_step"]
        if saved["task_counts"] != task_counts(self.completed):
            raise ValueError("Checkpoint task counts differ")
        self.model.load_state_dict(saved["model"], strict=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        self.best = saved["best_score"]
        torch.set_rng_state(saved["rng_cpu"].cpu())
        torch.cuda.set_rng_state(saved["rng_cuda"].cpu())
        np.random.set_state(saved["rng_numpy"])

    def reference(self):
        if self.reference_path.exists():
            ref = json.loads(self.reference_path.read_text())
            self.control, self.spectral_keys = ref["control"], ref["spectral_keys"]
        else:
            baseline = self.evaluator.evaluate(self.validation)
            self.control = self.evaluator.evaluate(self.validation, climatology=True)
            self.spectral_keys = sorted(
                set(baseline["spectra"]) & set(self.control["spectra"])
            )
            if any(
                not any(k.startswith(v + "/") for k in self.spectral_keys)
                for v in ["sst", "adt", "eke"]
            ):
                raise ValueError("Incomplete selection spectra")
            atomic_json(
                {
                    "control": self.control,
                    "spectral_keys": self.spectral_keys,
                    "protocol": PROTOCOL,
                },
                self.reference_path,
            )

    def validate(self):
        result = self.evaluator.evaluate(self.validation)
        score = selection_score(result, self.control, self.spectral_keys)
        values = {
            "validation": {"composite": score, **result},
            "om4_retention": retention(self.model, self.om4),
        }
        atomic_json(values, self.out / f"validation-{self.completed:05d}.json")
        self.emit("validation", values)
        if score < self.best:
            self.best = score
            self.checkpoint(self.out / "best.pt")
        self.checkpoint(self.out / "last.pt")

    def train(self):
        assert self.om4 is not None
        if (self.out / "last.pt").exists():
            self.restore(self.out / "last.pt")
        self.reference()
        if not self.completed:
            self.validate()
        for sig in (signal.SIGTERM, signal.SIGUSR1):
            signal.signal(sig, lambda *_: setattr(self, "stop", True))
        while self.completed < TOTAL_UPDATES and not self.stop:
            counts = task_counts(self.completed)
            after = task_counts(self.completed + 1)
            task = (
                "observation" if after["observation"] > counts["observation"] else "om4"
            )
            count = counts[task]
            seed = SEED + (0 if task == "observation" else 100000)
            size = (
                len(self.training) if task == "observation" else len(self.om4.trainset)
            )
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            metrics: dict[str, float] = {}
            for micro, index in enumerate(sample_indices(size, seed, count)):
                mask_seed = seed + 1000000 + count * ACCUMULATE + micro
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if task == "observation":
                        loss, parts = observation_objective(
                            self.model,
                            self.data,
                            self.data.load(self.training[index]),
                            mask_seed,
                        )
                    else:
                        coverage = self.data.load(
                            self.training[
                                (count * ACCUMULATE + micro) % len(self.training)
                            ]
                        )["validity"][:, :19]
                        loss, parts = om4_objective(
                            self.model,
                            self.om4,
                            [index],
                            RECONSTRUCTION_WEIGHT,
                            mask_seed,
                            coverage,
                            COMPLETION_WEIGHT,
                        )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / ACCUMULATE).backward()
                for k, v in {"loss": loss, **parts}.items():
                    metrics[k] = metrics.get(k, 0) + float(v.detach()) / ACCUMULATE
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 1, error_if_nonfinite=True
            )
            self.optimizer.step()
            self.completed += 1
            self.emit(
                "train",
                {
                    "train": {
                        task: metrics,
                        "gradient_norm": float(grad),
                        "learning_rate": LEARNING_RATE,
                    }
                },
            )
            if self.completed % VALIDATE_EVERY == 0:
                self.validate()
            if self.args.stop_after and self.completed >= self.args.stop_after:
                self.stop = True
        self.checkpoint(self.out / "last.pt")
        if self.completed == TOTAL_UPDATES:
            self.checkpoint(self.out / "endpoint.pt")
            atomic_json(
                {
                    "updates": self.completed,
                    "task_counts": task_counts(self.completed),
                    "endpoint_sha256": digest(self.out / "endpoint.pt"),
                },
                self.out / "COMPLETE.json",
            )

    def annual(self):
        self.restore(self.out / self.args.checkpoint)
        if not self.args.annual_data:
            raise ValueError("Annual evaluation requires --annual-data")
        for origin in ["2015-01-01", "2018-01-01", "2021-01-01"]:
            result = evaluate_year(
                self.model,
                self.data,
                self.args.annual_data / origin,
                self.out / f"annual-{self.args.checkpoint}-{origin}.npz",
            )
            atomic_json(
                result, self.out / f"annual-{self.args.checkpoint}-{origin}.json"
            )
            self.emit("annual", {"annual": {origin: result}})

    def evaluate(self):
        self.restore(self.out / self.args.checkpoint)
        self.reference()
        paths = self.data.paths("test")
        control = self.evaluator.evaluate(paths, climatology=True)
        for label, options in [
            ("forecast", {}),
            ("initialized-persistence", {"persistence": True}),
            ("climatology", {"climatology": True}),
        ]:
            metrics = self.evaluator.evaluate(paths, **options)
            result = {
                "composite": selection_score(metrics, control, self.spectral_keys),
                **metrics,
            }
            atomic_json(result, self.out / f"test-{self.args.checkpoint}-{label}.json")
            self.emit("test", {"test": {label: result}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate", "annual"])
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--om4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annual-data", type=Path)
    parser.add_argument("--model", default="observation/model.yaml")
    parser.add_argument("--om4-config", default="observation/om4.yaml")
    parser.add_argument("--cache-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--checkpoint", choices=["best.pt", "endpoint.pt"], default="best.pt"
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        help="Operational partial-run limit; does not change the fixed schedule",
    )
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--entity", default=None)
    parser.add_argument("--project", default="observational-transfer")
    parser.add_argument("--name", default="global-physical-only")
    args = parser.parse_args()
    harness = Harness(args)
    try:
        if args.command == "train":
            harness.train()
        elif args.command == "evaluate":
            harness.evaluate()
        else:
            harness.annual()
    finally:
        harness.log.finish()


if __name__ == "__main__":
    main()
