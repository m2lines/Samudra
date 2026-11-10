# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Scalar encodings with a single shared transformer and small refinement heads."""

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .benchmark import Config, Fields, Model, atomic_json
from .noise import atomic_save, optimizer

KINDS = [
    "mse",
    "asinh_mse",
    "radix",
    "parallel_nested",
    "ar_uniform",
    "ar_nested",
    "ar_overlap",
    "ar_twohot_hard",
    "ar_twohot_soft",
    "flat_twohot",
    "flat_gauss",
]
EXTENSION_KINDS = [
    "cont_shrink_mse",
    "cont_shrink_normalized",
    "cont_fixed_mse",
    "ar_overlap_scaled",
    "ar_twohot_hard_scaled",
    "ar_twohot_soft_scaled",
]
SUPPORT = 8.0
QUANTUM = 1 / 4096


def quantize(x):
    code = ((x + SUPPORT) / QUANTUM).round().long().clamp(0, 65535)
    return code.to(x.dtype) * QUANTUM - SUPPORT, code


class Source(Fields):
    def draw(self, count, generator, query_index=None):
        batch = super().draw(count, generator, query_index)
        normalized = batch["values"][..., 0] / self.scale
        if normalized.abs().max() >= SUPPORT:
            raise ValueError("Synthetic field escaped fixed support")
        value, code = quantize(normalized)
        batch["values"][..., 0] = value * self.scale
        rows = torch.arange(count, device=self.device)
        batch["main"] = value[rows, batch["index"]].float()
        batch["code"] = code[rows, batch["index"]]
        batch["aux"] = (
            (0.1 - batch["values"][rows, batch["index"], 1]) / self.scale
        ).float()
        return batch


def grid(count, reference, uniform=False):
    u = torch.linspace(-1, 1, count, device=reference.device, dtype=reference.dtype)
    return u if uniform else torch.sinh(u * math.asinh(3)) / 3


def twohot(value, centers):
    """Interpolate in scalar coordinates, clamping only outside current support."""
    hi = (value[:, None] >= centers).sum(-1).clamp(1, centers.shape[-1] - 1)
    lo = hi - 1
    left = centers.gather(1, lo[:, None]).squeeze(1)
    right = centers.gather(1, hi[:, None]).squeeze(1)
    weight = ((value - left) / (right - left)).clamp(0, 1)
    result = torch.zeros_like(centers)
    result.scatter_(1, lo[:, None], (1 - weight)[:, None])
    result.scatter_add_(1, hi[:, None], weight[:, None])
    return result


def geometry(center, radius, kind):
    overlap = "overlap" in kind or "twohot" in kind
    if overlap:
        candidates = center[:, None] + radius[:, None] * grid(16, center)
        return candidates, None
    edges = center[:, None] + radius[:, None] * grid(17, center, kind == "ar_uniform")
    return (edges[:, :-1] + edges[:, 1:]) / 2, edges


def labels(value, candidates, edges):
    if edges is None:
        return (value[:, None] - candidates).abs().argmin(-1)
    return (value[:, None] >= edges[:, 1:]).sum(-1).clamp(max=15)


def advance(center, radius, candidates, edges, index=None, probabilities=None):
    if probabilities is not None:
        new_center = (candidates * probabilities).sum(-1)
    else:
        new_center = candidates.gather(1, index[:, None]).squeeze(1)
    if edges is None:
        return new_center, radius / 8
    widths = (edges[:, 1:] - edges[:, :-1]) / 2
    return new_center, widths.gather(1, index[:, None]).squeeze(1)


class EncodedModel(nn.Module):
    def __init__(self, config, positions, kind):
        super().__init__()
        self.kind = kind
        self.backbone = Model(config, positions)
        self.backbone.add_module("head", nn.Identity())
        sequential = kind.startswith(("ar_", "cont_"))
        outputs = (
            1
            if kind in {"mse", "asinh_mse"} or kind.startswith("cont_")
            else 16
            if sequential
            else 64
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.width + (6 if sequential else 0), config.width),
            nn.SiLU(),
            nn.Linear(config.width, outputs),
        )

    def features(self, batch):
        return self.backbone(batch)

    def decode(self, h, value, teacher=False, oracle=False):
        if self.kind.startswith("cont_"):
            return self.decode_continuous(h, value, teacher, oracle)
        scaled = self.kind.endswith("_scaled")
        kind = self.kind.removesuffix("_scaled")
        count = len(h)
        if kind in {"mse", "asinh_mse"}:
            raw = self.decoder(h).squeeze(-1)
            target = value if kind == "mse" else value.asinh()
            prediction = raw if kind == "mse" else raw.sinh()
            return prediction, F.mse_loss(raw, target), {"stages": prediction[:, None]}
        if kind == "radix":
            logits = self.decoder(h).reshape(count, 4, 16)
            _, code = quantize(value)
            powers = torch.tensor([4096, 256, 16, 1], device=h.device)
            truth = (code[:, None] // powers) % 16
            chosen = truth if oracle else logits.argmax(-1)
            prediction = (chosen * powers).sum(-1) * QUANTUM - SUPPORT
            return (
                prediction,
                F.cross_entropy(logits.flatten(0, 1), truth.flatten()),
                {
                    "stages": prediction[:, None],
                    "correct": chosen == truth,
                },
            )
        if kind.startswith("flat_"):
            logits = self.decoder(h)
            centers = (SUPPORT * grid(64, h)).expand(count, -1)
            if kind == "flat_twohot":
                target = twohot(value, centers)
            else:
                # Gaussian histogram in transformed space; sigma = one bin width.
                edges = torch.linspace(
                    -math.asinh(SUPPORT),
                    math.asinh(SUPPORT),
                    65,
                    device=h.device,
                    dtype=h.dtype,
                )
                sigma = edges[1] - edges[0]
                cdf = 0.5 * (
                    1
                    + torch.erf(
                        (edges[None] - value.asinh()[:, None]) / (sigma * math.sqrt(2))
                    )
                )
                target = cdf[:, 1:] - cdf[:, :-1]
                target = target / target.sum(-1, keepdim=True)
                centers = ((edges[:-1] + edges[1:]) / 2).sinh().expand(count, -1)
            probabilities = target if oracle else logits.softmax(-1)
            prediction = (probabilities * centers).sum(-1)
            return (
                prediction,
                -(target * logits.log_softmax(-1)).sum(-1).mean(),
                {
                    "stages": prediction[:, None],
                },
            )
        parallel = kind == "parallel_nested"
        parallel_logits = self.decoder(h).reshape(count, 4, 16) if parallel else None
        center = torch.zeros_like(value)
        radius = torch.full_like(value, SUPPORT)
        # Parallel targets follow a canonical true path; decoding follows predictions.
        true_center, true_radius = center.clone(), radius.clone()
        losses, stages, correct, misses, margins, entropy = [], [], [], [], [], []
        for level in range(4):
            level_vector = F.one_hot(
                torch.full((count,), level, device=h.device), 4
            ).to(h.dtype)
            state = torch.cat(
                (
                    center[:, None] / SUPPORT,
                    (radius / SUPPORT).log2()[:, None] / 16,
                    level_vector,
                ),
                -1,
            )
            logits = (
                parallel_logits[:, level]
                if parallel_logits is not None
                else self.decoder(torch.cat((h, state.detach()), -1))
            )
            candidates, edges = geometry(center, radius, kind)
            tc, te = (
                geometry(true_center, true_radius, kind)
                if parallel
                else (candidates, edges)
            )
            target_index = labels(value, tc, te)
            target = (
                twohot(value, tc)
                if "twohot" in kind
                else F.one_hot(target_index, 16).to(h.dtype)
            )
            probabilities = logits.softmax(-1)
            chosen = logits.argmax(-1)
            correct.append(chosen == target_index)
            misses.append((value < center - radius) | (value > center + radius))
            top = probabilities.topk(2, -1).values
            margins.append(top[:, 0] - top[:, 1])
            entropy.append(-(probabilities * logits.log_softmax(-1)).sum(-1))
            stage_ce = -(target.detach() * logits.log_softmax(-1)).sum(-1)
            stage_weight = (
                (radius / SUPPORT).square() if scaled else torch.ones_like(radius)
            )
            losses.append((stage_weight.detach() * stage_ce).mean())
            if parallel:
                true_center, true_radius = advance(
                    true_center, true_radius, tc, te, target_index
                )
            if teacher or oracle:
                chosen = labels(value, candidates, edges)
                weights = (
                    twohot(value, candidates) if kind == "ar_twohot_soft" else None
                )
            else:
                weights = probabilities if kind == "ar_twohot_soft" else None
            center, radius = advance(center, radius, candidates, edges, chosen, weights)
            stages.append(center)
            center, radius = center.detach(), radius.detach()
        return (
            stages[-1],
            torch.stack(losses).mean(),
            {
                "stages": torch.stack(stages, -1),
                "correct": torch.stack(correct, -1),
                "miss": torch.stack(misses, -1),
                "margin": torch.stack(margins, -1),
                "entropy": torch.stack(entropy, -1),
            },
        )

    def decode_continuous(self, h, value, teacher=False, oracle=False):
        center = torch.zeros_like(value)
        radius = torch.full_like(value, SUPPORT)
        stages, losses, misses, radii, corrections = [], [], [], [], []
        normalized = self.kind == "cont_shrink_normalized"
        shrink = self.kind != "cont_fixed_mse"
        for level in range(4):
            indicator = F.one_hot(torch.full((len(h),), level, device=h.device), 4).to(
                h.dtype
            )
            state = torch.cat(
                (
                    center[:, None] / SUPPORT,
                    (radius / SUPPORT).log2()[:, None] / 16,
                    indicator,
                ),
                -1,
            )
            correction = (
                self.decoder(torch.cat((h, state.detach()), -1)).squeeze(-1).tanh()
            )
            target = (value - center) / radius
            prediction = center + radius * correction
            error = correction - target.detach() if normalized else prediction - value
            losses.append(error.square().mean())
            misses.append(target.abs() > 1)
            radii.append(radius)
            corrections.append(correction)
            if teacher or oracle:
                prediction = center + radius * target.clamp(-1, 1)
            stages.append(prediction)
            center = prediction.detach()
            if shrink:
                radius = radius / 8
        return (
            stages[-1],
            torch.stack(losses).mean(),
            {
                "stages": torch.stack(stages, -1),
                "miss": torch.stack(misses, -1),
                "radius": torch.stack(radii, -1),
                "correction": torch.stack(corrections, -1),
            },
        )

    def forward(self, batch, teacher=False):
        h, auxiliary = self.features(batch)
        prediction, loss, details = self.decode(h, batch["main"], teacher)
        return prediction, auxiliary, loss, details


def batches(source, count, seed):
    rng = source.generator(seed)
    return [source.draw(min(64, count - i), rng) for i in range(0, count, 64)]


@torch.no_grad()
def evaluate(model, data):
    preds, errors, aux_errors, details = [], [], [], []
    for batch in data:
        pred, auxiliary, _, detail = model(batch)
        preds.append(pred)
        errors.append(pred.double() - batch["main"].double())
        aux_errors.append(auxiliary.double() - batch["aux"].double())
        details.append({k: v.double() for k, v in detail.items()})
    error = torch.cat(errors)
    rmse = error.square().mean().sqrt().item()
    result: dict[str, Any] = {
        "rmse": rmse,
        "bits": -math.log2(max(rmse, 1e-30)),
        "bias": error.mean().item(),
        "p99_abs": error.abs().quantile(0.99).item(),
        "max_abs": error.abs().max().item(),
        "aux_nrmse": torch.cat(aux_errors).square().mean().sqrt().item(),
        "exact_value": (error == 0).double().mean().item(),
    }
    truth = torch.cat([b["main"] for b in data])
    p = torch.cat(preds)
    result["rounded_word_accuracy"] = (
        (quantize(p)[1] == quantize(truth)[1]).float().mean().item()
    )
    for key in details[0]:
        values = torch.cat([d[key] for d in details])
        if key == "stages":
            result["stage_rmse"] = (
                (values - truth[:, None]).square().mean(0).sqrt().tolist()
            )
        else:
            result[key] = values.mean(0).tolist()
    if "correct" in details[0] and len(result["stage_rmse"]) == 4:
        first_wrong = ~torch.cat([d["correct"] for d in details])[:, 0].bool()
        stage_values = torch.cat([d["stages"] for d in details])
        result["first_wrong_fraction"] = first_wrong.double().mean().item()
        result["stage_rmse_first_wrong"] = (
            (stage_values[first_wrong] - truth[first_wrong, None])
            .square()
            .mean(0)
            .sqrt()
            .tolist()
            if first_wrong.any()
            else None
        )
    return result


def calibrate(model, source):
    """Match decoder CE gradient RMS to a fixed normalized-MSE reference."""
    shared = [
        p
        for name, p in model.backbone.named_parameters()
        if not name.startswith("aux_head")
    ]
    # Same reference initialization per seed for all encodings.
    with torch.random.fork_rng(
        devices=[source.device.index or 0] if source.device.type == "cuda" else []
    ):
        torch.manual_seed(source.c.seed + 91000)
        reference = nn.Linear(source.c.width, 1).to(source.device)
    norms = [0.0, 0.0]
    for batch in batches(source, 256, 77001):
        h, _ = model.features(batch)
        _, loss, _ = model.decode(h, batch["main"], teacher=True)
        base = F.mse_loss(reference(h).squeeze(-1), batch["main"])
        for i, objective in enumerate([loss, base]):
            grad = torch.autograd.grad(
                objective, shared, retain_graph=True, allow_unused=True
            )
            norms[i] += sum(
                g.double().square().sum().item() for g in grad if g is not None
            )
    return math.sqrt(norms[1] / max(norms[0], 1e-30)), norms


def fit_aux(model, source):
    xs, ys = [], []
    with torch.no_grad():
        hook = model.backbone.aux_head.register_forward_pre_hook(
            lambda m, args: xs.append(args[0].double().cpu())
        )
        for batch in batches(source, 4096, 663001):
            model.features(batch)
            ys.append(batch["aux"].double().cpu())
        hook.remove()
        x = torch.cat(xs)
        x = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), -1)
        solution = torch.linalg.lstsq(
            x, torch.cat(ys), driver="gelsd", rcond=1e-8
        ).solution
        model.backbone.aux_head.weight.copy_(solution[:-1][None])
        model.backbone.aux_head.bias.copy_(solution[-1:])
    return {
        "weight_norm": model.backbone.aux_head.weight.norm().item(),
        "bias": model.backbone.aux_head.bias.item(),
    }


def gradient_diagnostics(model, source, weight):
    batch = batches(source, 64, 77881)[0]
    _, aux, main_loss, _ = model(batch)
    noise = torch.randn(64, device=source.device, generator=source.generator(77882))
    objectives = [
        weight * main_loss,
        (aux - batch["aux"]).square().mean(),
        (aux - batch["aux"] - noise).square().mean(),
    ]
    parameters = [
        p for n, p in model.backbone.named_parameters() if not n.startswith("aux_head")
    ]
    gradients = []
    for objective in objectives:
        values = torch.autograd.grad(
            objective, parameters, retain_graph=True, allow_unused=True
        )
        gradients.append(
            torch.cat([v.detach().flatten() for v in values if v is not None])
        )
    norms = [g.norm().item() for g in gradients]
    return {
        "shared_gradient_norms_main_clean_noisy": norms,
        "shared_cosines_main_clean_noisy": [
            (gradients[0] @ g).item() / max(norms[0] * norm, 1e-30)
            for g, norm in zip(gradients[1:], norms[1:], strict=True)
        ],
    }


def train(model, source, folder, steps, lr, weight, mode, sigma, warmup, tracking=None):
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "latest.pt"
    opt = optimizer(model, lr)
    ema = copy.deepcopy(model)
    data_rng, noise_rng = source.generator(44000), source.generator(55000)
    validation = batches(source, source.c.validation_size, 88001)
    start, history, elapsed = 0, [], 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location=source.device, weights_only=False)
        model.load_state_dict(saved["model"])
        ema.load_state_dict(saved["ema"])
        opt.load_state_dict(saved["optimizer"])
        data_rng.set_state(saved["data_rng"].cpu())
        noise_rng.set_state(saved["noise_rng"].cpu())
        start, history, elapsed = saved["step"], saved["history"], saved["elapsed"]
    began = time.monotonic()
    for step in range(start, steps + 1):
        if step > start:
            batch = source.draw(source.c.batch, data_rng)
            epsilon = torch.randn(
                source.c.batch, device=source.device, generator=noise_rng
            )
            opt.zero_grad(set_to_none=True)
            _, aux, main_loss, _ = model(batch, teacher=step <= warmup)
            aux_loss = (aux - batch["aux"] - sigma * epsilon).square().mean()
            loss = (
                weight * main_loss
                if mode == "main"
                else aux_loss
                if mode == "aux"
                else weight * main_loss + aux_loss
            )
            loss.backward()
            opt.step()
            with torch.no_grad():
                for e, p in zip(ema.parameters(), model.parameters(), strict=True):
                    e.lerp_(p, 0.01)
        if (
            step == start
            or step % 100 == 0
            or step == steps
            or (mode == "aux" and step in {1, 4, 16, 64})
        ):
            row = {
                "step": step,
                "raw": evaluate(model, validation),
                "ema": evaluate(ema, validation),
            }
            if step == 0 or step % 1000 == 0 or step == steps:
                row["gradients"] = gradient_diagnostics(model, source, weight)
            if not math.isfinite(row["raw"]["rmse"]):
                raise FloatingPointError("Nonfinite copy prediction")
            if not history or history[-1]["step"] != step:
                history.append(row)
            saved = {
                "step": step,
                "model": model.state_dict(),
                "ema": ema.state_dict(),
                "optimizer": opt.state_dict(),
                "data_rng": data_rng.get_state(),
                "noise_rng": noise_rng.get_state(),
                "history": history,
                "elapsed": elapsed + time.monotonic() - began,
            }
            atomic_save(checkpoint, saved)
            atomic_json(folder / "curves.json", history)
            if mode == "main" and warmup > 0:
                for threshold in (4, 6):
                    path = folder / f"bits-{threshold}.pt"
                    if (
                        step > warmup
                        and len(history) >= 3
                        and all(r["raw"]["bits"] >= threshold for r in history[-3:])
                        and not path.exists()
                    ):
                        atomic_save(
                            path,
                            {
                                "model": model.state_dict(),
                                "step": step,
                                "bits": row["raw"]["bits"],
                            },
                        )
            print(json.dumps({"phase": folder.name, **row}), flush=True)
            if tracking:
                tracking.log(
                    {
                        f"{folder.name}/step": step,
                        **{
                            f"{folder.name}/{k}": v
                            for k, v in row["raw"].items()
                            if isinstance(v, (int, float))
                        },
                    }
                )
    test = batches(source, source.c.test_size, 99001)
    result = {
        "raw": evaluate(model, test),
        "ema": evaluate(ema, test),
        "elapsed": elapsed + time.monotonic() - began,
    }
    atomic_json(folder / "result.json", result)
    return result


def run(entry, output, device):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(entry["seed"])
    config = Config(
        seed=entry["seed"],
        lr=entry["lr"],
        validation_size=64 if entry.get("smoke") else 512,
        test_size=128 if entry.get("smoke") else 8192,
    )
    source = Source(config, device)
    model = EncodedModel(config, source.positions, entry["kind"]).to(
        device=device, dtype=torch.float32
    )
    tracking = None
    if os.environ.get("WANDB_MODE") != "disabled":
        import wandb

        tracking = wandb.init(
            project="samudra",
            entity="ocean_emulators",
            group=os.environ.get("WANDB_RUN_GROUP", "precision-encoding"),
            id=entry["id"],
            name=entry["id"],
            resume="allow",
            config=entry,
        )
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text())
    else:
        weight, gradient_norms = calibrate(model, source)
        meta = {
            "entry": entry,
            "weight": weight,
            "calibration_squared_norms": gradient_norms,
            "parameters": sum(p.numel() for p in model.parameters()),
            "decoder_parameters": sum(p.numel() for p in model.decoder.parameters()),
            "source": os.environ.get("SOURCE_COMMIT"),
            "job": os.environ.get("SLURM_JOB_ID"),
            "device": torch.cuda.get_device_name() if device == "cuda" else "cpu",
            "torch": torch.__version__,
        }
        atomic_json(metadata_path, meta)
    solo = train(
        model,
        source,
        output / "solo",
        entry["solo_steps"],
        entry["lr"],
        meta["weight"],
        "main",
        0,
        entry["warmup"],
        tracking,
    )
    final_state = copy.deepcopy(model.state_dict())
    fit_path = output / "prepared.pt"
    if fit_path.exists():
        fit = torch.load(fit_path, map_location=device, weights_only=False)
    else:
        conditioning = fit_aux(model, source)
        fit_metrics = evaluate(model, batches(source, 1024, 883001))
        fit = {
            "model": copy.deepcopy(model.state_dict()),
            "metrics": fit_metrics,
            "conditioning": conditioning,
            "qualified": fit_metrics["aux_nrmse"] < 0.2,
        }
        atomic_save(fit_path, fit)
    result = {
        "metadata": meta,
        "solo": solo,
        "aux_fit": {k: v for k, v in fit.items() if k != "model"},
        "forks": {},
    }
    # Copy-learning failure is reported, rather than relabeled as interference.
    solo_history = json.loads((output / "solo" / "curves.json").read_text())
    qualified = (
        len(solo_history) >= 3
        and all(row["raw"]["bits"] >= 4 for row in solo_history[-3:])
        and fit["qualified"]
    )
    result["qualified"] = qualified
    if qualified or entry.get("smoke", False):
        for name, mode, sigma in [
            ("main", "main", 0),
            ("clean", "joint", 0),
            ("noisy", "joint", 1),
        ]:
            model.load_state_dict(fit["model"])
            result["forks"][name] = train(
                model,
                source,
                output / name,
                entry["joint_steps"],
                entry["lr"],
                meta["weight"],
                mode,
                sigma,
                0,
                tracking,
            )
    # Direct probes use sustained-precision checkpoints, with separately qualified heads.
    result["probes"] = {}
    for level in (4, 6):
        milestone = output / "solo" / f"bits-{level}.pt"
        if not milestone.exists() or not entry.get("probes", True):
            continue
        saved = torch.load(milestone, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        conditioning = fit_aux(model, source)
        before = evaluate(model, batches(source, 1024, 883001))
        if before["aux_nrmse"] >= 0.2:
            result["probes"][str(level)] = {"qualified": False, "before": before}
            continue
        prepared = copy.deepcopy(model.state_dict())
        record = {
            "before": before,
            "parent_step": saved["step"],
            "conditioning": conditioning,
        }
        for name, mode, sigma in [
            ("main", "main", 0),
            ("clean", "aux", 0),
            ("noisy", "aux", 1),
        ]:
            model.load_state_dict(prepared)
            record[name] = train(
                model,
                source,
                output / f"probe{level}-{name}",
                entry.get("probe_steps", 256),
                entry["lr"],
                meta["weight"],
                mode,
                sigma,
                0,
                tracking,
            )
        result["probes"][str(level)] = record
    model.load_state_dict(final_state)
    atomic_json(output / "result.json", result)
    if tracking:
        tracking.finish()


def audit(output):
    """Exhaustive fixed-point target audit using float64 oracle decoding."""
    torch.set_num_threads(4)
    source = Source(Config(), "cpu")
    values = torch.arange(65536, dtype=torch.float64) * QUANTUM - SUPPORT
    results = {}
    for kind in KINDS:
        if kind in {"mse", "asinh_mse"}:
            continue
        model = EncodedModel(Config(), source.positions, kind).double()
        predictions = []
        with torch.no_grad():
            for v in values.split(1024):
                h = torch.zeros(len(v), 128, dtype=v.dtype)
                predictions.append(model.decode(h, v, oracle=True)[0])
        error = torch.cat(predictions) - values
        results[kind] = {
            "rmse": error.square().mean().sqrt().item(),
            "max_abs": error.abs().max().item(),
            "rounded_word_accuracy": (
                quantize(torch.cat(predictions))[1] == quantize(values)[1]
            )
            .double()
            .mean()
            .item(),
        }
    atomic_json(output, results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--make-manifest", action="store_true")
    parser.add_argument("--extension", action="store_true")
    args = parser.parse_args()
    if args.audit:
        audit(args.output)
    elif args.make_manifest:
        entries = []
        for kind in EXTENSION_KINDS if args.extension else KINDS:
            for lr in (1e-4, 1e-5):
                for seed in (11, 22, 33):
                    entry = {
                        "kind": kind,
                        "lr": lr,
                        "seed": seed,
                        "solo_steps": 12000,
                        "joint_steps": 3000,
                        "warmup": 1000,
                        "probes": True,
                        "probe_steps": 256,
                    }
                    entry["id"] = (
                        kind
                        + "-"
                        + hashlib.sha256(
                            json.dumps(entry, sort_keys=True).encode()
                        ).hexdigest()[:12]
                    )
                    entries.append(entry)
        atomic_json(args.output, entries)
    else:
        entry = json.loads(args.manifest.read_text())[args.index]
        run(entry, args.output / entry["id"], args.device)


if __name__ == "__main__":
    main()
