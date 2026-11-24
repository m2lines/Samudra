# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Auxiliary perturbations and copy-output displacement measurements."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from .benchmark import Config, Fields, Model, atomic_json

SETTINGS = [(3e-4, 64), (1e-4, 64), (3e-5, 64), (1e-5, 64), (3e-4, 256), (3e-4, 1024)]
CONDITIONS = [
    ("unlearned", 0.0),
    ("learned", 0.0),
    ("learned", 0.1),
    ("learned", 0.3),
    ("learned", 1.0),
    ("main", 0.0),
    ("offset", 0.0),
]


def optimizer(model, lr, kind="adam"):
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)


def draw(source, count, seed):
    generator = source.generator(seed)
    return [source.draw(min(64, count - i), generator) for i in range(0, count, 64)]


def target(batch):
    rows = torch.arange(len(batch["index"]), device=batch["index"].device)
    return 0.1 - batch["values"][rows, batch["index"], 1]


def predictions(model, batches):
    with torch.no_grad():
        values = [model(batch) for batch in batches]
    return tuple(torch.cat([v[j].double() for v in values]) for j in (0, 1))


def metrics(model, batches, scale):
    main, aux = predictions(model, batches)
    truth = torch.cat([b["main"] for b in batches])
    aux_truth = torch.cat([target(b) for b in batches])
    error = main - truth
    rmse = error.square().mean().sqrt().item()
    return {
        "bits": -math.log2(max(rmse / scale, 1e-30)),
        "rmse": rmse,
        "bias": error.mean().item(),
        "p99_abs": error.abs().quantile(0.99).item(),
        "aux_clean_nrmse": (aux - aux_truth).square().mean().sqrt().item() / scale,
    }


def step(
    model,
    opt,
    source,
    batch_size,
    data_rng,
    noise_rng,
    sigma,
    mode,
    offset=0.0,
    dead_gradient=False,
):
    opt.zero_grad(set_to_none=True)
    totals = [0.0, 0.0]
    for _ in range(batch_size // 64):
        batch = source.draw(64, data_rng)
        # Always draw noise, even for clean controls, to preserve pairing.
        epsilon = torch.randn(64, device=source.device, generator=noise_rng)
        main, aux = model(batch)
        lm = ((main - batch["main"].float()) / source.scale).square().mean()
        la = (
            ((aux - target(batch).float()) / source.scale - sigma * epsilon)
            .square()
            .mean()
        )
        loss = lm if mode == "main" else la if mode == "aux" else lm + la
        if dead_gradient:
            loss = loss * 0
        ((loss + offset) / (batch_size // 64)).backward()
        # The additive constant enters the differentiated scalar; its derivative is zero.
        totals[0] += lm.item() / (batch_size // 64)
        totals[1] += (la.item() + offset) / (batch_size // 64)
    opt.step()
    return totals


def atomic_save(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    os.replace(temp, path)


def model_from(saved, source, learned=False):
    model = Model(Config(**saved["base_config"]), source.positions).to(
        device=source.device, dtype=torch.float32
    )
    model.load_state_dict(saved["model"])
    if learned:
        model.aux_head.load_state_dict(saved["learned_head"])
    return model


def prepare(entry, output, device):
    path = Path(entry["parent"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError("Parent SHA256 mismatch")
    old = torch.load(path, map_location=device, weights_only=False)
    config = {
        k: v for k, v in old["config"].items() if k in Config.__dataclass_fields__
    }
    c = Config(**config)
    source = Fields(c, device)
    model = Model(c, source.positions).to(device=device, dtype=torch.float32)
    model.load_state_dict(old["model"])
    features = []
    hook = model.aux_head.register_forward_pre_hook(
        lambda module, args: features.append(args[0].detach().double().cpu())
    )
    calibration = draw(source, 4096, 663001)
    predictions(model, calibration)
    hook.remove()
    x = torch.cat(features)
    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1)
    y = torch.cat([target(b).cpu() for b in calibration])
    # Minimum-norm least squares, fitted only on a separate calibration stream.
    solution = torch.linalg.lstsq(x, y, driver="gelsd", rcond=1e-8).solution
    before = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        model.aux_head.weight.copy_(solution[:-1][None].to(device))
        model.aux_head.bias.copy_(solution[-1:].to(device))
    val = draw(source, 1024, 883001)
    result = metrics(model, val, source.scale)
    smoke = copy.deepcopy(model)
    step(
        smoke,
        optimizer(smoke, 1e-5),
        source,
        1024,
        source.generator(556001),
        source.generator(556002),
        0.3,
        "aux",
    )
    if not all(torch.isfinite(p).all() for p in smoke.parameters()):
        raise FloatingPointError("GPU accumulated noisy-update qualification failed")
    saved = {
        "base_config": config,
        "model": before,
        "learned_head": copy.deepcopy(model.aux_head.state_dict()),
        "inherited_optimizer": old["optimizer"],
        "parent_step": old["step"],
        "parent_sha256": digest,
        "validation": result,
        "qualified": result["aux_clean_nrmse"] < 0.2,
        "seed": c.seed,
    }
    atomic_save(output / "prepared.pt", saved)
    return {
        "entry": entry,
        "validation": result,
        "qualified": saved["qualified"],
        "parent_step": old["step"],
        "parent_sha256": digest,
    }


def probes(saved, source, lr, batch_size, condition, sigma, repeats=16):
    heldout = draw(source, 64, 883002)
    learned = condition != "unlearned"
    model = model_from(saved, source, learned)
    initial = copy.deepcopy(model.state_dict())
    start, _ = predictions(model, heldout)
    truth = torch.cat([b["main"] for b in heldout])
    err = start - truth
    rows = []
    for kind in ("sgd", "adam", "inherited_adam"):
        displacements = []
        for index in range(repeats):
            model.load_state_dict(initial)
            opt = optimizer(model, lr, "sgd" if kind == "sgd" else "adam")
            if kind == "inherited_adam":
                opt.load_state_dict(copy.deepcopy(saved["inherited_optimizer"]))
                for group in opt.param_groups:
                    group["lr"] = lr
            step(
                model,
                opt,
                source,
                batch_size,
                source.generator(773000 + index),
                source.generator(774000 + index),
                sigma,
                "main" if condition == "main" else "aux",
                dead_gradient=condition == "coast",
            )
            pred, _ = predictions(model, heldout)
            displacements.append((pred - start).cpu())
        displacement = torch.stack(displacements)
        mean = displacement.mean(0)
        variance = displacement.var(0, unbiased=True).mean().item()
        alignment = (2 * mean * err.cpu()).mean().item()
        rows.append(
            {
                "optimizer": kind,
                "repeats": repeats,
                "mean_displacement_rms": mean.square().mean().sqrt().item(),
                "displacement_variance": variance,
                "error_alignment_term": alignment,
                "mean_mse_change": alignment + displacement.square().mean().item(),
                "displacements": displacement.tolist(),
            }
        )
    return rows


def trajectory(
    saved,
    source,
    folder,
    lr,
    batch_size,
    condition,
    sigma,
    mode,
    steps,
    recovery_steps,
    stop_after=None,
):
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / "result.json"
    if final.exists():
        return json.loads(final.read_text())
    model = model_from(saved, source, condition != "unlearned")
    opt = optimizer(model, lr)
    emas = {str(d): copy.deepcopy(model).requires_grad_(False) for d in (0.99, 0.999)}
    data_rng, noise_rng = (
        source.generator(773100 + saved["seed"]),
        source.generator(774100 + saved["seed"]),
    )
    val = draw(source, 256, 883003)
    checkpoint = folder / "latest.pt"
    history = []
    done = 0
    elapsed = 0.0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=source.device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        for key, ema in emas.items():
            ema.load_state_dict(state["ema"][key])
        data_rng.set_state(state["data_rng"].cpu())
        noise_rng.set_state(state["noise_rng"].cpu())
        history, done, elapsed = state["history"], state["step"], state["seconds"]

    def evaluate(index):
        row = {
            "step": index,
            "examples": index * batch_size,
            "raw": metrics(model, val, source.scale),
            "ema": {key: metrics(ema, val, source.scale) for key, ema in emas.items()},
        }
        history.append(row)
        print(json.dumps({"condition": condition, "sigma": sigma, **row}), flush=True)

    if not history:
        evaluate(0)
    for index in range(done + 1, steps + recovery_steps + 1):
        started = time.monotonic()
        active_mode = "main" if condition == "main" or index > steps else mode
        losses = step(
            model,
            opt,
            source,
            batch_size,
            data_rng,
            noise_rng,
            sigma,
            active_mode,
            offset=1.0 if condition == "offset" else 0.0,
        )
        with torch.no_grad():
            for key, ema in emas.items():
                for a, b in zip(ema.parameters(), model.parameters(), strict=True):
                    a.lerp_(b, 1 - float(key))
        if source.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.monotonic() - started
        relative = index if index <= steps else index - steps
        if (
            relative in (1, 4, 16, 32, 64, 128, 256, 512, 1024, 2048)
            or relative % 100 == 0
            or index in (steps, steps + recovery_steps)
        ):
            evaluate(index)
            history[-1]["training_main_loss"] = losses[0]
            history[-1]["training_aux_loss"] = losses[1]
            atomic_save(
                checkpoint,
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "ema": {key: ema.state_dict() for key, ema in emas.items()},
                    "data_rng": data_rng.get_state(),
                    "noise_rng": noise_rng.get_state(),
                    "step": index,
                    "history": history,
                    "seconds": elapsed,
                },
            )
        if stop_after is not None and index >= stop_after:
            return {"interrupted": True}
        if not all(math.isfinite(v) for v in losses):
            raise FloatingPointError(losses)
    test = draw(source, 8192, 993001)
    result = {
        "condition": condition,
        "sigma": sigma,
        "mode": mode,
        "lr": lr,
        "batch": batch_size,
        "steps": steps,
        "recovery_steps": recovery_steps,
        "training_seconds": elapsed,
        "history": history,
        "test": metrics(model, test, source.scale),
        "ema_test": {
            key: metrics(ema, test, source.scale) for key, ema in emas.items()
        },
    }
    atomic_json(final, result)
    return result


def run(entry, output, device):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    output.mkdir(parents=True, exist_ok=True)
    entry_path = output / "entry.json"
    if entry_path.exists() and json.loads(entry_path.read_text()) != entry:
        raise ValueError("Resume entry mismatch")
    if (output / "result.json").exists():
        return
    atomic_json(entry_path, entry)
    tracking = None
    if os.environ.get("WANDB_MODE", "disabled") != "disabled":
        import wandb

        tracking = wandb.init(
            project="samudra",
            group="precision-noise-20261109",
            id="pn-" + entry["stage"] + "-" + entry["id"],
            resume="allow",
            config=entry,
            dir=str(output),
            mode="offline" if os.environ["WANDB_MODE"] == "offline" else "online",
        )
    if entry["stage"] == "prepare":
        result = prepare(entry, output, device)
    else:
        path = Path(entry["prepared"])
        saved = torch.load(path, map_location=device, weights_only=False)
        source = Fields(Config(**saved["base_config"]), device)
        lr, batch_size = entry["lr"], entry["batch"]
        result = {
            "entry": entry,
            "prepared_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "qualified": saved["qualified"],
            "trajectories": [],
            "probes": [],
        }
        conditions = (
            CONDITIONS
            if entry["stage"] == "perturb"
            else [("main", 0.0), ("learned", 0.0), ("learned", 1.0)]
        )
        for condition, sigma in conditions:
            if condition not in ("unlearned", "main") and not saved["qualified"]:
                continue
            label = f"{condition}-{sigma}"
            probe_file = output / f"probe-{label}.json"
            if entry["stage"] == "perturb":
                if not probe_file.exists():
                    atomic_json(
                        probe_file,
                        probes(saved, source, lr, batch_size, condition, sigma),
                    )
                result["probes"].append(
                    {
                        "condition": condition,
                        "sigma": sigma,
                        "measurements": json.loads(probe_file.read_text()),
                    }
                )
            result["trajectories"].append(
                trajectory(
                    saved,
                    source,
                    output / label,
                    lr,
                    batch_size,
                    condition,
                    sigma,
                    "aux" if entry["stage"] == "perturb" else "joint",
                    256 if entry["stage"] == "perturb" else 3000,
                    256 if entry["stage"] == "perturb" else 0,
                )
            )
        if entry["stage"] == "perturb":
            coast = output / "probe-coast.json"
            if not coast.exists():
                atomic_json(coast, probes(saved, source, lr, batch_size, "coast", 0.0))
            result["coasting_probes"] = json.loads(coast.read_text())
    result["provenance"] = {
        "source": os.environ.get("SOURCE_COMMIT"),
        "job": os.environ.get("SLURM_JOB_ID"),
        "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    atomic_json(output / "result.json", result)
    if tracking:
        for record in result.get("trajectories", []):
            label = f"{record['condition']}-{record['sigma']}"
            for row in record["history"]:
                tracking.log(
                    {
                        label + "/step": row["step"],
                        **{label + "/" + k: v for k, v in row["raw"].items()},
                    }
                )
        tracking.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    entry = json.loads(args.manifest.read_text())[args.index]
    run(entry, args.output / entry["id"], args.device)


if __name__ == "__main__":
    main()
