# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Auditable single-GPU qualification/training; no autoregressive feedback."""

import argparse
import hashlib
import json
import math
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import DEPTHS, OceanSource, audit_support
from .model import ModelConfig, PointTransformer, Vocabulary


def emit(record, path=None):
    line = json.dumps(record, allow_nan=False)
    print(line, flush=True)
    if path:
        with Path(path).open("a") as f:
            f.write(line + "\n")


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


class FrameStream:
    """One current and one prefetched global window; resume preserves sampling."""

    def __init__(self, source, seed, state=None):
        self.source = source
        self.rng = np.random.default_rng(seed)
        self.pool = ThreadPoolExecutor(max_workers=1)
        if state is None:
            self.order = self.rng.permutation(source.origins["train"]).tolist()
            self.cursor = 0
            self.current_origin, self.next_origin = self.pop(), self.pop()
            self.used = 0
        else:
            self.rng.bit_generator.state = state["rng"]
            self.order, self.cursor = state["order"], state["cursor"]
            self.current_origin, self.next_origin = (
                state["current_origin"],
                state["next_origin"],
            )
            self.used = state["used"]
        self.frame = source.read(self.current_origin)
        self.future = self.pool.submit(source.read, self.next_origin)

    def pop(self):
        if self.cursor == len(self.order):
            self.order = self.rng.permutation(self.source.origins["train"]).tolist()
            self.cursor = 0
        value = self.order[self.cursor]
        self.cursor += 1
        return value

    def advance(self):
        self.frame = self.future.result()  # Read failures propagate to the trainer.
        self.current_origin, self.used = self.next_origin, 0
        self.next_origin = self.pop()
        self.future = self.pool.submit(self.source.read, self.next_origin)

    def state(self):
        return {
            "rng": self.rng.bit_generator.state,
            "order": self.order,
            "cursor": self.cursor,
            "current_origin": self.current_origin,
            "next_origin": self.next_origin,
            "used": self.used,
        }

    def close(self):
        # result() prevents a late background read failure from being swallowed.
        self.future.result()
        self.pool.shutdown(wait=True)


@torch.no_grad()
def evaluate(model, vocab, source, config, device, output, step):
    model.eval()
    origins = source.origins["validation"]
    limit = config["validation_dates"]
    if limit:
        origins = origins[
            np.linspace(0, len(origins) - 1, min(limit, len(origins)), dtype=int)
        ]
    rng = np.random.default_rng(config["validation_seed"])
    pit_rng = np.random.default_rng(config["validation_seed"] + 1)
    # Columns: count, error squared in C, persistence squared in C, normalized
    # error squared, bias in C, NLL, finite density NLL, outside count,
    # quantization error squared in C, density count.
    sums = np.zeros((19, 10), dtype=np.float64)
    pit = np.zeros((19, 20), dtype=np.int64)
    baseline_sse = np.zeros(19, dtype=np.float64)
    baseline_nmse = np.zeros(19, dtype=np.float64)
    baseline_root = (
        Path(config["baseline_cache"]) if config.get("baseline_cache") else None
    )
    if baseline_root:
        baseline_receipt = json.loads((baseline_root / "manifest.json").read_text())
        for key in ("metadata_sha256", "mean", "std"):
            if baseline_receipt["source"][key] != source.receipt()[key]:
                raise ValueError(f"Baseline cache mismatch: {key}")
    target_std = torch.tensor(source.std[38:57], device=device)
    sample_records = []
    for origin in origins:
        frame = source.read(int(origin))
        baseline = (
            np.load(baseline_root / f"temperature-{origin:04d}.npy", mmap_mode="r")
            if baseline_root
            else None
        )
        depths = np.tile(np.arange(19), config["validation_points_per_depth"])
        for start in range(0, len(depths), config["microbatch"]):
            selected = depths[start : start + config["microbatch"]]
            batch = to_device(
                source.sample(frame, rng, len(selected), config["crop"], selected),
                device,
            )
            with torch.autocast(
                "cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = model(batch)
            target = batch["target"]
            residual = model.config.objective == "residual_ce"
            objective_target = target - batch["current"] if residual else target
            depth = batch["depth"]
            scale = target_std[depth]
            categorical = model.config.objective != "regression"
            if categorical:
                ids = vocab.encode(objective_target, depth)
                pred = vocab.decode(logits, depth) + (
                    batch["current"] if residual else 0
                )
                nll = F.cross_entropy(logits, ids, reduction="none")
                log_density = vocab.log_density(
                    logits, objective_target, depth, target_std
                )
                outside = vocab.outside(objective_target, depth)
                oracle = vocab.centers[depth, ids] + (
                    batch["current"] if residual else 0
                )
                probabilities = logits.softmax(-1)
                cdf = probabilities.cumsum(-1)
                mass = probabilities.gather(1, ids[:, None]).squeeze(1)
                cdf_at = cdf.gather(1, ids[:, None]).squeeze(1)
                u = torch.tensor(pit_rng.uniform(size=len(selected)), device=device)
                randomized_pit = (cdf_at - mass + u * mass).cpu().numpy()
                for z, p in zip(selected, randomized_pit, strict=True):
                    pit[z, min(19, max(0, int(p * 20)))] += 1
            else:
                pred = logits.squeeze(-1)
                nll = torch.zeros_like(target)
                log_density = torch.zeros_like(target)
                outside = torch.zeros_like(target, dtype=torch.bool)
                oracle = target
            error = (pred - target) * scale
            persist = (batch["current"] - target) * scale
            baseline_values, baseline_log_density = None, None
            if baseline is not None:
                baseline_values = torch.tensor(
                    [
                        float(baseline[z, int(batch["row"][i]), int(batch["col"][i])])
                        for i, z in enumerate(selected)
                    ],
                    device=device,
                )
                baseline_errors = (baseline_values - target).square().cpu().numpy()
                for z, squared_error in zip(selected, baseline_errors, strict=True):
                    baseline_nmse[z] += squared_error
                    baseline_sse[z] += squared_error * source.std[38 + z] ** 2
                if categorical:
                    candidate = (
                        baseline_values - batch["current"]
                        if residual
                        else baseline_values
                    )
                    baseline_log_density = vocab.log_density(
                        logits, candidate, depth, target_std
                    )
            rows = (
                torch.stack(
                    (
                        torch.ones_like(target),
                        error.square(),
                        persist.square(),
                        (pred - target).square(),
                        error,
                        nll,
                        torch.where(outside, 0, -log_density),
                        outside.float(),
                        ((oracle - target) * scale).square(),
                        (~outside).float(),
                    ),
                    -1,
                )
                .cpu()
                .numpy()
            )
            for z, row in zip(selected, rows, strict=True):
                sums[z] += row
            for i, z in enumerate(selected):
                sample_records.append(
                    {
                        "origin": int(origin),
                        "row": int(batch["row"][i]),
                        "col": int(batch["col"][i]),
                        "depth": int(z),
                        "truth_c": float(target[i] * scale[i] + source.mean[38 + z]),
                        "prediction_c": float(pred[i] * scale[i] + source.mean[38 + z]),
                        "nll": float(nll[i]) if categorical else None,
                        "log_density": float(log_density[i])
                        if categorical and not outside[i]
                        else None,
                        "samudra_c": float(
                            baseline_values[i] * scale[i] + source.mean[38 + z]
                        )
                        if baseline_values is not None
                        else None,
                        "samudra_log_density": float(baseline_log_density[i])
                        if baseline_log_density is not None
                        and torch.isfinite(baseline_log_density[i])
                        else None,
                    }
                )
    metrics = []
    for z, row in enumerate(sums):
        n = row[0]
        if not n:
            raise ValueError(f"No validation samples at depth {z}")
        metrics.append(
            {
                "depth": z,
                "depth_m": float(DEPTHS[z]),
                "count": int(n),
                "rmse_c": math.sqrt(row[1] / n),
                "persistence_rmse_c": math.sqrt(row[2] / n),
                "nmse": row[3] / n,
                "bias_c": row[4] / n,
                "bin_nll": row[5] / n if categorical else None,
                "in_support_density_nll": row[6] / row[9]
                if categorical and row[9]
                else None,
                "outside_support": int(row[7]) if categorical else None,
                "quantization_rmse_c": math.sqrt(row[8] / n) if categorical else None,
                "samudra_rmse_c": math.sqrt(baseline_sse[z] / n)
                if baseline_root
                else None,
                "samudra_nmse": baseline_nmse[z] / n if baseline_root else None,
            }
        )
    result = {
        "step": step,
        "origins": origins.tolist(),
        "per_depth": metrics,
        "mean_depth_nmse": float(np.mean([m["nmse"] for m in metrics])),
        "mean_depth_bin_nll": float(np.mean([m["bin_nll"] for m in metrics]))
        if categorical
        else None,
        "randomized_pit_counts": pit.tolist() if categorical else None,
        "scope": "sampled wet points, area-proportional within each depth; true histories; +5 days",
    }
    (output / f"validation-{step:07d}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    with (output / f"predictions-{step:07d}.jsonl").open("w") as f:
        for record in sample_records:
            f.write(json.dumps(record, allow_nan=False) + "\n")
    model.train()
    return result


def save_atomic(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def train(config, source, output):
    output.mkdir(parents=True, exist_ok=True)
    if config["steps"] < 1 or config["accumulation"] < 1 or config["microbatch"] < 1:
        raise ValueError("Positive steps and batch sizes required")
    if config["samples_per_frame"] % config["microbatch"]:
        raise ValueError("samples_per_frame must be divisible by microbatch")
    support = json.loads(Path(config["support"]).read_text())
    if support["source"]["metadata_sha256"] != source.metadata_sha256:
        raise ValueError("Support and data release differ")
    if (
        support["source"]["mean"] != source.mean.tolist()
        or support["source"]["std"] != source.std.tolist()
    ):
        raise ValueError("Support audit and training normalization differ")
    if not support["complete_training_scan"] and not config["qualification"]:
        raise ValueError(
            "Scientific training requires the complete training-only support audit"
        )
    if config["qualification"] and config["steps"] > 100:
        raise ValueError("Qualification is bounded to 100 optimizer updates")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "Cluster trainer requires a GPU; use unit tests for CPU checks"
        )
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    model_config = ModelConfig(**config["model"])
    model = PointTransformer(model_config).to(device)
    bounds = support[
        "residual_bounds"
        if model_config.objective == "residual_ce"
        else "absolute_bounds"
    ]
    vocab = Vocabulary(bounds, model_config.bins).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        betas=(0.9, 0.95),
        fused=True,
    )
    start_step, best_nll, best_nmse, stream_state = 0, math.inf, math.inf, None
    latest = output / "latest.pt"
    if latest.exists():
        if not config["resume"]:
            raise FileExistsError(
                f"Checkpoint exists; explicit resume required: {latest}"
            )
        state = torch.load(latest, map_location="cpu", weights_only=False)
        # Budgets are fixed: extending a schedule is a new explicit experiment.
        saved_config = dict(state["config"])
        current_config = dict(config)
        saved_config.pop("resume", None)
        current_config.pop("resume", None)
        if saved_config != current_config:
            raise ValueError("Resume configuration changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_step, best_nll, best_nmse = (
            state["step"],
            state["best_nll"],
            state["best_nmse"],
        )
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        stream_state = state["stream"]
        del state
    elif config["resume"]:
        raise FileNotFoundError("Resume requested without latest.pt")
    provenance = {
        "config": config,
        "model": asdict(model_config),
        "source": source.receipt(),
        "support": support,
        "parameters": sum(p.numel() for p in model.parameters()),
        "effective_batch": config["microbatch"] * config["accumulation"],
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "source_commit": os.environ.get("SKYLINE_COMMIT"),
        "source_archive_sha256": os.environ.get("SKYLINE_ARCHIVE_SHA256"),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    wandb_run = None
    if config["wandb"]:
        import wandb

        wandb_run = wandb.init(
            project="temperature-skyline",
            entity="ocean_emulators",
            name=output.name,
            config=provenance,
            id=hashlib.sha256(str(output).encode()).hexdigest()[:12],
            resume="allow" if config["resume"] else "never",
        )
    emit(
        {
            "event": "start",
            "parameters": provenance["parameters"],
            "effective_batch": provenance["effective_batch"],
            "gpu": provenance["gpu"],
            "qualification": config["qualification"],
        }
    )
    stream = FrameStream(source, seed + 100, stream_state)
    torch.cuda.reset_peak_memory_stats()
    model.train()
    for step in range(start_step + 1, config["steps"] + 1):
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        warmup = config["warmup"]
        fraction = max(0, (step - warmup) / max(1, config["steps"] - warmup))
        factor = (
            step / max(1, warmup)
            if step <= warmup
            else 0.1 + 0.9 * (1 + math.cos(math.pi * fraction)) / 2
        )
        lr = config["learning_rate"] * factor
        for group in optimizer.param_groups:
            group["lr"] = lr
        losses, waiting, outside_count = [], 0.0, 0
        for _ in range(config["accumulation"]):
            load_start = time.monotonic()
            if stream.used >= config["samples_per_frame"]:
                stream.advance()
            batch = to_device(
                source.sample(stream.frame, rng, config["microbatch"], config["crop"]),
                device,
            )
            stream.used += config["microbatch"]
            waiting += time.monotonic() - load_start
            target = batch["target"]
            if model_config.objective == "residual_ce":
                target = target - batch["current"]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(batch)
            if model_config.objective == "regression":
                loss = F.mse_loss(logits.squeeze(-1), target)
            else:
                outside_count += int(vocab.outside(target, batch["depth"]).sum())
                if outside_count and not config["qualification"]:
                    raise ValueError("Training target outside audited support")
                loss = F.cross_entropy(logits, vocab.encode(target, batch["depth"]))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            (loss / config["accumulation"]).backward()
            losses.append(float(loss.detach()))
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        record = {
            "step": step,
            "loss": float(np.mean(losses)),
            "grad_norm": float(norm),
            "lr": lr,
            "seconds": time.monotonic() - begin,
            "data_wait_seconds": waiting,
            "frame_read_seconds": stream.frame["read_seconds"],
            "origin": stream.current_origin,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "outside_support": outside_count,
        }
        emit(record, output / "training.jsonl")
        if wandb_run:
            wandb_run.log({"train/" + k: v for k, v in record.items()}, step=step)
        if step % config["validate_every"] == 0 or step == config["steps"]:
            validation = evaluate(model, vocab, source, config, device, output, step)
            scores = {"validation/nmse": validation["mean_depth_nmse"]}
            nll = validation["mean_depth_bin_nll"]
            if nll is not None:
                scores["validation/nll"] = nll
            for m in validation["per_depth"]:
                scores[f"validation/thetao_{m['depth']}/rmse_c"] = m["rmse_c"]
            emit(
                {"event": "validation", "step": step, **scores},
                output / "training.jsonl",
            )
            if wandb_run:
                wandb_run.log(scores, step=step)
            weights = {
                "model": model.state_dict(),
                "vocabulary": vocab.state_dict(),
                "config": config,
                "step": step,
                "validation": validation,
            }
            if validation["mean_depth_nmse"] < best_nmse:
                best_nmse = validation["mean_depth_nmse"]
                save_atomic(weights, output / "best-error.pt")
            if nll is not None and nll < best_nll:
                best_nll = nll
                save_atomic(weights, output / "best-likelihood.pt")
        if step % config["save_every"] == 0 or step == config["steps"]:
            save_atomic(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "vocabulary": vocab.state_dict(),
                    "config": config,
                    "step": step,
                    "best_nll": best_nll,
                    "best_nmse": best_nmse,
                    "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                    "stream": stream.state(),
                },
                latest,
            )
    stream.close()
    if wandb_run:
        wandb_run.finish()
    (output / "COMPLETE.json").write_text(
        json.dumps({"step": config["steps"], "qualification": config["qualification"]})
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "precision", "train"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dates", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.set_num_threads(config.get("torch_threads", 4))
    source = OceanSource(
        config["source"], config["readers"], config.get("reference_root")
    )
    try:
        if args.mode == "audit":
            print(audit_support(source, args.output, args.audit_dates), flush=True)
        elif args.mode == "precision":
            from .precision import audit_precision

            audit_precision(source, config, args.output)
        else:
            train(config, source, args.output)
    finally:
        source.close()


if __name__ == "__main__":
    main()
