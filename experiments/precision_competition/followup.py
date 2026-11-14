# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Learning-rate qualification and exact-state forks with observational weight EMA."""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import shutil
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from .benchmark import (
    Config,
    Fields,
    Model,
    approximate_step_flops,
    atomic_json,
    evaluate,
    execution,
    spectrum_probe,
)
from .benchmark import losses as physical_losses


@dataclass(frozen=True)
class FollowupConfig(Config):
    stage: str = "qualification"
    schedule: str = "constant"
    lr_end: float = 3e-7
    decay_steps: int = 3000
    ema_decays: list[float] = field(default_factory=lambda: [0.99, 0.999])
    main_loss_scale: float = 1.0
    aux_loss_scale: float = 1.0
    parent_checkpoint: str = ""
    parent_sha256: str = ""
    pair_id: str = ""

    def __post_init__(self):
        super().__post_init__()
        if self.schedule not in {"constant", "cosine"} or self.decay_steps <= 100:
            raise ValueError("Invalid LR schedule")
        if min(self.main_loss_scale, self.aux_loss_scale) <= 0:
            raise ValueError("Loss scales must be positive")
        if any(not 0 < decay < 1 for decay in self.ema_decays):
            raise ValueError("EMA decay must be inside (0,1)")
        if bool(self.parent_checkpoint) != bool(self.parent_sha256):
            raise ValueError("Forks require both checkpoint path and SHA256")


def scheduled_lr(config, step):
    if step <= 100:
        return config.lr * step / 100
    if config.schedule == "constant":
        return config.lr
    progress = min(1.0, (step - 100) / (config.decay_steps - 100))
    return (
        config.lr_end
        + (config.lr - config.lr_end) * (1 + math.cos(math.pi * progress)) / 2
    )


def losses(prediction, auxiliary, batch, config):
    main, aux = physical_losses(prediction, auxiliary, batch, config)
    return main / config.main_loss_scale, aux / config.aux_loss_scale


def weighted_gradient_probe(model, batch, config):
    params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith(("head.", "aux_head."))
    ]
    main, aux = model(batch)
    lm, la = losses(main, aux, batch, config)
    gm = torch.autograd.grad(
        config.main_weight * lm, params, retain_graph=True, allow_unused=True
    )
    ga = torch.autograd.grad(config.auxiliary_weight * la, params, allow_unused=True)
    m = sum(g.double().square().sum().item() for g in gm if g is not None)
    a = sum(g.double().square().sum().item() for g in ga if g is not None)
    dot = sum(
        (x.double() * y.double()).sum().item()
        for x, y in zip(gm, ga, strict=True)
        if x is not None and y is not None
    )
    return {
        "weighted_main_gradient_norm": math.sqrt(m),
        "weighted_aux_gradient_norm": math.sqrt(a),
        "weighted_gradient_cosine": dot / math.sqrt(m * a) if m * a else None,
    }


def train(config, destination, device="cuda", wandb_mode="disabled", stop_after=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / "config.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != asdict(
        config
    ):
        raise ValueError("Cannot resume with a changed configuration")
    atomic_json(metadata_path, asdict(config))
    if (destination / "result.json").exists():
        return json.loads((destination / "result.json").read_text())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(config.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)
    source = Fields(config, device)
    model_dtype = torch.float64 if config.precision == "fp64" else torch.float32
    model = Model(config, source.positions).to(device=device, dtype=model_dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.0)
    ema_models = {
        str(decay): copy.deepcopy(model).requires_grad_(False)
        for decay in config.ema_decays
    }
    generator = source.generator(config.seed + 100000)
    last_step, training_seconds, history = 0, 0.0, []
    checkpoint_path = destination / "latest.pt"
    initial_path = (
        checkpoint_path
        if checkpoint_path.exists()
        else (Path(config.parent_checkpoint) if config.parent_checkpoint else None)
    )
    parent_digest = None
    if initial_path is not None:
        if initial_path != checkpoint_path:
            parent_digest = hashlib.sha256(initial_path.read_bytes()).hexdigest()
            if parent_digest != config.parent_sha256:
                raise ValueError("Parent checkpoint hash mismatch")
        saved = torch.load(initial_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["data_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        for key, ema in ema_models.items():
            ema.load_state_dict(saved.get("ema", {}).get(key, saved["model"]))
        last_step, training_seconds, history = (
            saved["step"],
            saved["training_seconds"],
            saved["history"],
        )
        if initial_path != checkpoint_path:
            # Preserve weights, Adam moments, EMA and RNG; stage counters restart.
            last_step, training_seconds, history = 0, 0.0, []
    milestones = set()
    if checkpoint_path.exists():
        milestones = {path.stem for path in destination.glob("bits-*.pt")}
    provenance = {
        "parent_sha256": config.parent_sha256,
        "parent_checkpoint": config.parent_checkpoint,
        "config_id": config.run_id,
        "source_commit": os.environ.get("SOURCE_COMMIT", "uncommitted"),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": os.environ.get("MANIFEST_SHA256"),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "restart_count": os.environ.get("SLURM_RESTART_COUNT", "0"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
        "parameters": sum(p.numel() for p in model.parameters()),
        "main_head_parameters": sum(p.numel() for p in model.head.parameters()),
        "approximate_training_flops_per_update": approximate_step_flops(config),
        "tf32": False,
        "wandb_mode": wandb_mode,
        "resumed_step": last_step,
    }
    atomic_json(destination / f"provenance-{time.time_ns()}.json", provenance)
    run = None
    if wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project="samudra",
            group="precision-competition-followup-20261108",
            id="pc2-" + config.run_id,
            resume="allow",
            mode=wandb_mode,
            name=f"{config.stage}-{config.main_weight}-{config.auxiliary}-{config.schedule}-lr{config.lr}-s{config.seed}-{config.run_id[:6]}",
            config={**asdict(config), **provenance},
            dir=str(destination),
        )
    stopping = False

    def request_stop(signum, frame):
        nonlocal stopping
        stopping = True

    previous_signals = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGTERM, signal.SIGUSR1)
    }

    def save(step):
        value = {
            "config": asdict(config),
            "milestones": sorted(milestones),
            "ema": {key: value.state_dict() for key, value in ema_models.items()},
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "training_seconds": training_seconds,
            "history": history,
            "data_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if device.startswith("cuda")
            else [],
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(value, temporary)
        os.replace(temporary, checkpoint_path)
        atomic_json(destination / "history.json", history)
        if config.stage == "qualification":
            tags = []
            if step in (3000, config.steps):
                tags.append(f"step-{step}")
            if config.main_weight and not config.auxiliary_weight and len(history) >= 3:
                for bits in (4, 6, 8):
                    tag = f"bits-{bits}"
                    if tag not in milestones and all(
                        row["bits"] is not None and row["bits"] >= bits
                        for row in history[-3:]
                    ):
                        tags.append(tag)
                        milestones.add(tag)
            for tag in tags:
                if not (destination / f"{tag}.pt").exists():
                    target = destination / f"{tag}.pt"
                    temporary_copy = target.with_suffix(".tmp")
                    shutil.copyfile(checkpoint_path, temporary_copy)
                    os.replace(temporary_copy, target)

    def validate(step):
        metrics = evaluate(model, source, config, config.validation_size, 880001)
        metrics["ema"] = {
            key: evaluate(ema, source, config, config.validation_size, 880001)
            for key, ema in ema_models.items()
        }
        probe = source.draw(config.batch, source.generator(880002))
        metrics.update(weighted_gradient_probe(model, probe, config))
        metrics.update(
            step=step,
            examples=step * config.batch,
            training_seconds=training_seconds,
            approximate_training_flops=step * approximate_step_flops(config),
        )
        history.append(metrics)
        print(
            json.dumps({"event": "validation", "id": config.run_id, **metrics}),
            flush=True,
        )
        if run:
            run.log(
                {
                    "validation/" + k: v
                    for k, v in metrics.items()
                    if not isinstance(v, dict)
                },
                step=step,
            )

    # Use math attention for FP64 and CPU; production FP32/BF16 may dispatch to
    # fused kernels. Same-checkpoint precision comparisons are a later diagnostic.
    backend = (
        sdpa_kernel(SDPBackend.MATH)
        if config.precision == "fp64" or device == "cpu"
        else contextlib.nullcontext()
    )
    try:
        with backend:
            if not history:
                validate(0)
                save(0)
            reason = "steps"
            step = last_step
            model.train()
            for next_step in range(last_step + 1, config.steps + 1):
                if training_seconds >= config.train_seconds:
                    reason = "training_time"
                    break
                step = next_step
                if config.path == "wire":
                    step = 0
                    reason = "exact_wire"
                    break
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                started = time.monotonic()
                batch = source.draw(config.batch, generator)
                optimizer.zero_grad(set_to_none=True)
                learning_rate = scheduled_lr(config, step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                with execution(config, device):
                    main, auxiliary = model(batch)
                    lm, la = losses(main, auxiliary, batch, config)
                    weight = (
                        config.auxiliary_weight
                        if step > config.auxiliary_start
                        else 0.0
                    )
                    loss = config.main_weight * lm + weight * la
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    for key, ema in ema_models.items():
                        for target, current in zip(
                            ema.parameters(), model.parameters(), strict=True
                        ):
                            target.lerp_(current, 1 - float(key))
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                training_seconds += time.monotonic() - started
                if step == 1 or step % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "train",
                                "step": step,
                                "main_loss": lm.item(),
                                "auxiliary_loss": la.item(),
                                "auxiliary_weight": weight,
                                "training_seconds": training_seconds,
                            }
                        ),
                        flush=True,
                    )
                final = step == config.steps or training_seconds >= config.train_seconds
                if step % config.eval_every == 0 or final:
                    validate(step)
                    save(step)
                if stopping or (stop_after is not None and step >= stop_after):
                    save(step)
                    return {"interrupted": True, "step": step}
                if final:
                    reason = (
                        "training_time"
                        if training_seconds >= config.train_seconds
                        else "steps"
                    )
                    break
            if not history or history[-1]["step"] != step:
                validate(step)
            save(step)
            # Test set is used only at the fixed final/budget-censored endpoint.
            test = evaluate(model, source, config, config.test_size, 990001)
            spectral = spectrum_probe(model, source, config)
            ema_test = {
                key: evaluate(ema, source, config, config.test_size, 990001)
                for key, ema in ema_models.items()
            }
            # Fit only the additive correction on validation; never fit on test.
            correction = history[-1]["bias"]
            test["validation_bias_corrected_rmse"] = math.sqrt(
                max(
                    0.0,
                    test["rmse"] ** 2 - 2 * correction * test["bias"] + correction**2,
                )
            )
            for key, metrics in ema_test.items():
                correction = history[-1]["ema"][key]["bias"]
                metrics["validation_bias_corrected_rmse"] = math.sqrt(
                    max(
                        0.0,
                        metrics["rmse"] ** 2
                        - 2 * correction * metrics["bias"]
                        + correction**2,
                    )
                )
            result = {
                "config": asdict(config),
                "id": config.run_id,
                "step": step,
                "stop_reason": reason,
                "training_seconds": training_seconds,
                "test": test,
                "ema_test": ema_test,
                "spectrum": spectral,
                "provenance": provenance,
            }
            atomic_json(destination / "result.json", result)
            if run:
                run.log(
                    {
                        "test/" + k: v
                        for k, v in test.items()
                        if not isinstance(v, dict)
                    },
                    step=step,
                )
            return result
    finally:
        if run:
            run.finish()
        for sig, previous in previous_signals.items():
            signal.signal(sig, previous)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--wandb-mode", default="online", choices=["online", "offline", "disabled"]
    )
    args = parser.parse_args()
    config = FollowupConfig(
        **json.loads(args.manifest.read_text())[args.index]["config"]
    )
    result = train(config, args.output / config.run_id, args.device, args.wandb_mode)
    if result.get("interrupted"):
        raise SystemExit(75)


if __name__ == "__main__":
    main()
