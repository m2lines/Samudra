# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded, resumable deterministic-D observational fine-tuning pilot."""

import argparse
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.observation_metrics import PROTOCOL, score, selection_score
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_training import Samples


def atomic_torch(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def atomic_json(value, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class Pilot:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output)
        self.out.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        ready = Path(args.data) / "DATA_READY.json"
        if not ready.exists():
            raise ValueError(
                "Dataset must pass full source-hash verification before training"
            )
        readiness = json.loads(ready.read_text())
        if (
            readiness.get("verification")
            != "all 350 NPZ files checked against source SHA256SUMS"
        ):
            raise ValueError("Unknown data verification protocol")
        self.data = Samples(args.data, self.device)
        self.model = ObservationTransfer(
            self.data.grid["names"].tolist(), getattr(args, "normalization", "batch")
        )
        if args.from_scratch:
            self.data.use_observation_normalization()
            self.model.update_batchnorm = True
        else:
            contract = json.loads(Path(args.source_contract).read_text())
            if digest(args.checkpoint) != contract["checkpoint_sha256"]:
                raise ValueError(
                    "Source checkpoint differs from the qualified D artifact"
                )
            saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            self.model.load_core(saved["model"])
            del saved
        if not args.fit_probe:
            if not args.qualification:
                raise ValueError(
                    "Production requires a completed fitting qualification"
                )
            qualification = json.loads(Path(args.qualification).read_text())
            if (
                qualification["code_commit"] != os.environ.get("SAMUDRA_CODE_COMMIT")
                or qualification["data_manifest_sha256"]
                != digest(Path(args.data) / "SHA256SUMS")
                or not args.selection_reference
                or qualification["selection_reference_sha256"]
                != digest(args.selection_reference)
                or qualification.get("normalization", "batch")
                != getattr(args, "normalization", "batch")
                or not all(qualification["gradient_reached"].values())
                or qualification["losses"][-1] >= qualification["losses"][0]
            ):
                raise ValueError(
                    "Fitting qualification does not match this producer/data"
                )
        self.model.to(self.device)
        self.training = self.data.paths("train")
        self.validation = self.data.paths("validation")
        if len(self.training) != 243 or len(self.validation) != 9:
            raise ValueError("Incomplete fixed training/validation cohort")
        self.manifest = {
            "arguments": vars(args),
            "source_checkpoint_sha256": None
            if args.from_scratch
            else digest(args.checkpoint),
            "normalization_mode": "observation-only"
            if args.from_scratch
            else "source-model",
            "effective_mean": self.data.grid["mean"].tolist(),
            "effective_std": self.data.grid["std"].tolist(),
            "data_manifest_sha256": digest(Path(args.data) / "SHA256SUMS"),
            "grid_sha256": digest(Path(args.data) / "grid.npz"),
            "statistics_sha256": digest(Path(args.data) / "statistics.npz"),
            "protocol": PROTOCOL,
            "code_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
            "training_months": [p.stem for p in self.training],
            "validation_months": [p.stem for p in self.validation],
        }
        manifest_file = self.out / "manifest.json"
        if (
            manifest_file.exists()
            and json.loads(manifest_file.read_text()) != self.manifest
        ):
            raise ValueError("Resume manifest differs from frozen run")
        atomic_json(self.manifest, manifest_file)
        self.wandb = None
        if args.wandb_mode != "disabled":
            import wandb

            self.wandb = wandb.init(
                entity="ocean_emulators",
                project="observational-transfer",
                name=args.name,
                id=hashlib.sha256(str(self.out).encode()).hexdigest()[:12],
                resume="allow",
                dir=str(self.out),
                mode=args.wandb_mode,
                config=self.manifest,
            )
        self.control = None
        self.spectral_keys = None
        self.global_best = math.inf
        self.started = time.monotonic()

    def emit(self, value):
        record = {"time_utc": datetime.datetime.now(datetime.UTC).isoformat(), **value}
        with (self.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, allow_nan=False), flush=True)
        if self.wandb:
            numeric = {k: v for k, v in record.items() if isinstance(v, (int, float))}
            self.wandb.log(numeric)

    def arguments(
        self, sample
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            sample["surface"],
            sample["atmosphere"],
            sample["contexts"],
            self.data.mask,
            sample["validity"],
        )

    @torch.no_grad()
    def evaluate(
        self, paths, persistence=False, export=None, climatology=False, anomaly=False
    ):
        self.model.eval()
        predictions, references, ohc, reference_ohc = [], [], [], []
        interior_error, interior_bias, interior_count = (
            np.zeros(28),
            np.zeros(28),
            np.zeros(28),
        )
        for path in paths:
            sample = self.data.load(path)
            if climatology:
                prediction = self.data.climatology_prediction(sample)
            elif persistence or anomaly:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    initial = self.model.initialize(
                        sample["surface"][:, :19],
                        sample["atmosphere"][:, :19],
                        sample["contexts"][:, 18],
                        self.data.mask,
                        sample["validity"][:, :19],
                    )
                state = (
                    self.data.persistence_anomaly(initial, sample)
                    if anomaly
                    else initial[:, -1]
                )
                prediction = state[:, None].expand(
                    -1, len(sample["month_weights"]), -1, -1, -1
                )
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction, _ = self.model.forecast(*self.arguments(sample))
            monthly = (
                prediction * sample["month_weights"][None, :, None, None, None]
            ).sum(1)
            surface = self.data.physical(prediction)[:, :6, [38, 76]].cpu().numpy()[0]
            truth = np.concatenate(
                [sample["raw"]["surface"][19:25], sample["raw"]["velocity"][19:25]],
                axis=1,
            )
            predicted_ohc = self.data.ohc(monthly)[0]
            observed_ohc = np.where(
                np.isfinite(predicted_ohc), sample["raw"]["ohc"], np.nan
            )
            predictions.append(surface)
            references.append(truth)
            ohc.append(predicted_ohc)
            reference_ohc.append(observed_ohc)
            if export:
                print(
                    json.dumps(
                        {
                            "event": "evaluation_origin",
                            "method": Path(export).stem,
                            "origin": sample["name"],
                            "completed": len(predictions),
                            "total": len(paths),
                        }
                    ),
                    flush=True,
                )
            physical_ts = self.data.physical(monthly[:, None])[
                :, 0, self.data.ts_indices
            ]
            target = sample["interior"]
            valid = torch.isfinite(target) & self.data.ts_mask.bool()
            weights = valid * self.data.area
            error = torch.where(valid, physical_ts - target, 0)
            interior_error += (error.square() * weights).sum((0, 2, 3)).cpu().numpy()
            interior_bias += (error * weights).sum((0, 2, 3)).cpu().numpy()
            interior_count += weights.sum((0, 2, 3)).cpu().numpy()
        arrays = dict(
            prediction=np.array(predictions),
            reference=np.array(references),
            predicted_ohc=np.array(ohc),
            reference_ohc=np.array(reference_ohc),
        )
        result = score(
            **arrays,
            lat=self.data.grid["lat"],
            lon=self.data.grid["lon"],
            mask=self.data.grid["mask"][0],
        )
        supported = interior_count > 0
        result["thermohaline"] = {
            "channels": np.array(self.data.grid["names"])[self.data.ts_indices][
                supported
            ].tolist(),
            "rmse": np.sqrt(
                interior_error[supported] / interior_count[supported]
            ).tolist(),
            "bias": (interior_bias[supported] / interior_count[supported]).tolist(),
        }
        result["origins"] = [p.stem for p in paths]
        if export:
            temporary = Path(export).with_suffix(".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **arrays, origins=result["origins"])
            temporary.replace(export)
        return result

    def qualify_selection(self):
        frozen = self.out / "selection-reference.json"
        if frozen.exists():
            record = json.loads(frozen.read_text())
            self.control, self.spectral_keys = (
                record["control"],
                record["spectral_keys"],
            )
            self.global_best = float(
                json.loads((self.out / "best.json").read_text())["score"]
            )
            return
        baseline = self.evaluate(self.validation)
        if self.args.selection_reference:
            reference = json.loads(Path(self.args.selection_reference).read_text())
            if (
                reference["protocol"] != PROTOCOL
                or reference["data_manifest_sha256"]
                != self.manifest["data_manifest_sha256"]
                or reference["control"]["origins"] != [p.stem for p in self.validation]
            ):
                raise ValueError(
                    "Shared selection reference has different data or protocol"
                )
            control, keys = reference["control"], reference["spectral_keys"]
            if not set(keys).issubset(baseline["spectra"]):
                raise ValueError("Candidate cannot supply all shared selection spectra")
        else:
            control = self.evaluate(self.validation, climatology=True)
            keys = sorted(set(control["spectra"]) & set(baseline["spectra"]))
        missing_groups = [
            name
            for name in ("sst", "adt", "eke")
            if not any(key.startswith(name + "/") for key in keys)
        ]
        if missing_groups:
            raise ValueError(
                f"No qualified spectra for {missing_groups}; refuse incomplete selection"
            )
        self.control, self.spectral_keys = control, keys
        initial_score = selection_score(baseline, control, keys)
        self.global_best = initial_score
        atomic_json(
            {
                "control": control,
                "spectral_keys": keys,
                "protocol": PROTOCOL,
                "data_manifest_sha256": self.manifest["data_manifest_sha256"],
            },
            frozen,
        )
        atomic_json(baseline, self.out / "baseline-validation.json")
        self.save_best(initial_score, "source", 0, baseline)
        self.emit(
            {
                "event": "selection_qualified",
                "validation/obs_score": initial_score,
                "spectral_components": len(keys),
            }
        )

    def save_best(self, value, phase, step, metrics):
        self.global_best = value
        atomic_torch(
            {
                "model": self.model.state_dict(),
                "score": value,
                "phase": phase,
                "step": step,
            },
            self.out / "best.pt",
        )
        atomic_json(
            {
                "score": value,
                "phase": phase,
                "step": step,
                "metrics": metrics,
                "checkpoint_sha256": digest(self.out / "best.pt"),
            },
            self.out / "best.json",
        )

    def objective(self, sample, phase):
        arguments = self.arguments(sample)
        if phase == "joint":
            prediction, _ = self.model.forecast(*arguments)
            loss = self.data.forecast_loss(prediction, sample)
            if self.args.reconstruction_weight:
                reconstructed = self.model.reconstruct_month(
                    *arguments, sample["month_weights"]
                )
                loss = loss + self.args.reconstruction_weight * self.data.interior_loss(
                    reconstructed, sample
                )
        else:
            reconstructed = self.model.reconstruct_month(
                *arguments, sample["month_weights"]
            )
            loss = self.data.interior_loss(reconstructed, sample)
        return loss

    def phase(self, name, max_steps, hours):
        complete = self.out / (name + "-complete.json")
        if complete.exists():
            self.model.load_state_dict(
                torch.load(
                    self.out / (name + "-best.pt"),
                    map_location=self.device,
                    weights_only=False,
                )["model"]
            )
            return
        train_phase = "adapter" if self.args.adapter_only or name == "adapter" else name
        self.model.set_phase(train_phase)
        core = list(self.model.initializer.parameters()) + list(
            self.model.evolution.parameters()
        )
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in core if p.requires_grad],
                    "lr": getattr(self.args, "core_lr", None)
                    or (1e-4 if self.args.from_scratch else 1e-5),
                },
                {
                    "params": self.model.adapter.parameters(),
                    "lr": 1e-4 if name == "joint" else 1e-3,
                },
            ],
            weight_decay=0.01,
        )
        state = {"step": 0, "elapsed": 0.0, "best": math.inf, "bad_checks": 0}
        last = self.out / (name + "-last.pt")
        best = self.out / (name + "-best.pt")
        if last.exists():
            saved = torch.load(last, map_location=self.device, weights_only=False)
            self.model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            state = saved["state"]
            torch.set_rng_state(saved["torch_rng"].cpu())
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        else:
            metrics = self.evaluate(self.validation)
            state["best"] = selection_score(metrics, self.control, self.spectral_keys)
            atomic_torch(
                {"model": self.model.state_dict(), "score": state["best"]}, best
            )
        self.model.set_phase(train_phase)
        phase_start, prior, last_save = (
            time.monotonic(),
            state["elapsed"],
            time.monotonic(),
        )
        base_rates = [group["lr"] for group in optimizer.param_groups]
        # Adam's saved group rate may be in warm-up; use explicit configured peaks.
        base_rates[0] = getattr(self.args, "core_lr", None) or (
            1e-4 if self.args.from_scratch else 1e-5
        )
        base_rates[1] = 1e-4 if name == "joint" else 1e-3
        while state["step"] < max_steps and state["elapsed"] < hours * 3600:
            warmup = getattr(self.args, "warmup_steps", 0)
            factor = min(1.0, (state["step"] + 1) / warmup) if warmup else 1.0
            for group, rate in zip(optimizer.param_groups, base_rates, strict=True):
                group["lr"] = rate * factor
            optimizer.zero_grad(set_to_none=True)
            indices = np.random.default_rng(self.args.seed + state["step"]).choice(
                len(self.training), self.args.accumulate, replace=False
            )
            loss_sum = 0.0
            for index in indices:
                sample = self.data.load(self.training[index])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.objective(sample, name)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
                (loss / self.args.accumulate).backward()
                loss_sum += float(loss.detach()) / self.args.accumulate
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite training gradient")
            optimizer.step()
            state["step"] += 1
            state["elapsed"] = prior + time.monotonic() - phase_start
            done = state["step"] >= max_steps or state["elapsed"] >= hours * 3600
            self.emit(
                {
                    "event": "training",
                    "phase": name,
                    "step": state["step"],
                    "train/normalized_loss": loss_sum,
                    "train/normalized_rms": math.sqrt(loss_sum),
                    "train/gradient_norm": float(norm),
                    "phase_seconds": state["elapsed"],
                }
            )
            validate = state["step"] % self.args.validate_every == 0 or done
            if validate:
                metrics = self.evaluate(self.validation)
                value = selection_score(metrics, self.control, self.spectral_keys)
                atomic_json(
                    metrics, self.out / f"{name}-validation-{state['step']:05d}.json"
                )
                improved = value < state["best"]
                state["bad_checks"] = 0 if improved else state["bad_checks"] + 1
                if improved:
                    state["best"] = value
                    atomic_torch(
                        {"model": self.model.state_dict(), "score": value}, best
                    )
                if value < self.global_best:
                    self.save_best(value, name, state["step"], metrics)
                if not getattr(self.args, "fixed_updates", False):
                    done |= state["bad_checks"] >= self.args.patience
                self.emit(
                    {
                        "event": "validation",
                        "phase": name,
                        "step": state["step"],
                        "validation/obs_score": value,
                        "validation/best_obs_score": self.global_best,
                        **{f"obs/{k}": v for k, v in metrics["metrics"].items()},
                    }
                )
                milestones = getattr(self.args, "milestone_steps", [])
                if name == "joint" and state["step"] in milestones:
                    import shutil

                    prefix = self.out / f"joint-{state['step']:05d}"
                    atomic_torch(
                        {
                            "model": self.model.state_dict(),
                            "score": value,
                            "step": state["step"],
                        },
                        prefix.with_suffix(".pt"),
                    )
                    shutil.copyfile(self.out / "best.pt", str(prefix) + "-best.pt")
                    shutil.copyfile(self.out / "best.json", str(prefix) + "-best.json")
                self.model.set_phase(train_phase)
            if validate or done or time.monotonic() - last_save >= 180:
                state["elapsed"] = prior + time.monotonic() - phase_start
                atomic_torch(
                    {
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "state": state,
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(),
                    },
                    last,
                )
                last_save = time.monotonic()
            if done:
                break
        if getattr(self.args, "fixed_updates", False) and state["step"] < max_steps:
            raise RuntimeError(
                f"{name} hit its time cap before its fixed update budget"
            )
        self.model.load_state_dict(
            torch.load(best, map_location=self.device, weights_only=False)["model"]
        )
        atomic_json(state, complete)

    def fit_probe(self):
        """Small training-only fitting check, separate from scientific comparisons."""
        self.model.set_phase("joint")
        core = list(self.model.initializer.parameters()) + list(
            self.model.evolution.parameters()
        )
        optimizer = torch.optim.AdamW(
            [
                {"params": core, "lr": 1e-4 if self.args.from_scratch else 1e-5},
                {"params": self.model.adapter.parameters(), "lr": 1e-4},
            ],
            weight_decay=0.01,
        )
        sample = self.data.load(self.training[0])
        losses, reached = (
            [],
            {"initializer": False, "evolution": False, "adapter": False},
        )
        for step in range(11):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.objective(sample, "joint")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite fitting-probe loss")
            losses.append(float(loss.detach()))
            self.emit({"event": "fit_probe", "step": step, "loss": losses[-1]})
            if step == 10:
                break
            loss.backward()
            for name in reached:
                module = getattr(self.model, name)
                norm = (
                    sum(
                        float(p.grad.float().square().sum())
                        for p in module.parameters()
                        if p.grad is not None
                    )
                    ** 0.5
                )
                if not math.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite {name} gradient")
                reached[name] |= norm > 0
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite fitting-probe gradient")
            optimizer.step()
        if not all(reached.values()) or not losses[-1] < losses[0]:
            raise ValueError(
                f"Training-only fitting qualification failed: {losses}, {reached}"
            )
        atomic_json(
            {
                "normalization": getattr(self.args, "normalization", "batch"),
                "losses": losses,
                "gradient_reached": reached,
                "training_origin": self.training[0].stem,
                "code_commit": self.manifest["code_commit"],
                "data_manifest_sha256": self.manifest["data_manifest_sha256"],
                "selection_reference_sha256": digest(
                    self.out / "selection-reference.json"
                ),
                "gpu": torch.cuda.get_device_name(),
                "max_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
                "scope": "training-only ten-update fitting and integrated-validation qualification, not held-out skill",
            },
            self.out / "QUALIFIED.json",
        )

    def run(self):
        try:
            self.qualify_selection()
            if self.args.fit_probe:
                self.fit_probe()
                return
            if not self.args.from_scratch and self.args.adapter_steps:
                self.phase("adapter", self.args.adapter_steps, self.args.adapter_hours)
            self.phase(
                "reconstruction",
                self.args.reconstruction_steps,
                self.args.reconstruction_hours,
            )
            self.phase("joint", self.args.joint_steps, self.args.joint_hours)
            atomic_json(
                {
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--normalization", choices=["batch", "instance"], default="batch"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-contract", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--qualification")
    parser.add_argument("--fit-probe", action="store_true")
    parser.add_argument("--selection-reference")
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--adapter-steps", type=int, default=200)
    parser.add_argument("--adapter-hours", type=float, default=0.5)
    parser.add_argument("--reconstruction-steps", type=int, default=1000)
    parser.add_argument("--reconstruction-hours", type=float, default=2)
    parser.add_argument("--joint-steps", type=int, default=1000)
    parser.add_argument("--joint-hours", type=float, default=4)
    parser.add_argument("--reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--validate-every", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--core-lr", type=float)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--fixed-updates", action="store_true")
    parser.add_argument("--milestone-steps", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    args = parser.parse_args()
    for value in (
        args.accumulate,
        args.validate_every,
        args.reconstruction_steps,
        args.joint_steps,
    ):
        if value < 1:
            parser.error("Update counts and accumulation must be positive")
    if args.adapter_steps < 0 or args.warmup_steps < 0:
        parser.error("Adapter/warm-up counts must be nonnegative")
    if args.core_lr is not None and args.core_lr <= 0:
        parser.error("Core learning rate must be positive")
    if any(
        x <= 0 or x > args.joint_steps or x % args.validate_every
        for x in args.milestone_steps
    ):
        parser.error(
            "Milestones must be positive validation steps within the joint budget"
        )
    if args.from_scratch and args.adapter_only:
        parser.error("Scratch control must train its core")
    Pilot(args).run()


if __name__ == "__main__":
    main()
