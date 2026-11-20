# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic synthetic fields, point-query transformer, and paired training."""

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import signal
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class Config:
    width: int = 128
    depth: int = 4
    heads: int = 2
    tokens: int = 128
    batch: int = 64
    steps: int = 3000
    eval_every: int = 100
    validation_size: int = 1024
    test_size: int = 8192
    train_seconds: float = 600.0
    seed: int = 11
    lr: float = 0.0003
    main: str = "copy"
    auxiliary: str = "transport"
    relation: str = "independent"
    main_weight: float = 1.0
    auxiliary_weight: float = 0.0
    auxiliary_start: int = 0
    alpha: float = 0.01
    velocity_fine: float = 1.0
    temperature_fine: float = 0.125
    filter: str = "after"
    cutoff: int = 4
    path: str = "embedded"
    objective: str = "mse"
    bins: int = 256
    support: float = 2.0
    precision: str = "fp32"
    final_norm: bool = True

    def __post_init__(self):
        if self.width % self.heads or self.depth < 0 or self.tokens < 64:
            raise ValueError("Invalid transformer dimensions")
        if self.precision not in {"fp64", "fp32", "bf16"}:
            raise ValueError(self.precision)
        if self.main not in {"copy", "affine", "nonlinear", "forecast"}:
            raise ValueError(self.main)
        if self.auxiliary not in {"transport", "evolution", "shift"}:
            raise ValueError(self.auxiliary)
        if self.relation not in {"independent", "dependent"}:
            raise ValueError(self.relation)
        if self.objective not in {"mse", "quantized_mse", "tolerance", "ce"}:
            raise ValueError(self.objective)
        if self.path not in {"embedded", "retrieval", "residual", "wire", "mlp"}:
            raise ValueError(self.path)
        if self.filter not in {"before", "after"} or not 1 <= self.cutoff <= 8:
            raise ValueError("Invalid filter")
        if self.bins < 2 or self.support < 1.1 or not 0 <= self.alpha <= 0.1:
            raise ValueError("Targets must fit the fixed categorical support")
        if min(self.batch, self.eval_every, self.validation_size, self.test_size) < 1:
            raise ValueError("Invalid batch or evaluation size")
        if self.path == "wire" and (self.main != "copy" or self.auxiliary_weight):
            raise ValueError("Exact wire is an untrained copy-only control")

    @property
    def run_id(self):
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


class Fields:
    """Band-limited fields and exact manufactured 1-D periodic flux targets.

    Every source has Fourier coefficient L1 norm <= 1. Multiplication is
    alias-free at N>=64 (largest product mode 30). All reference operations
    are FP64, independent of the model arithmetic. No ocean data are used.
    """

    def __init__(self, config, device):
        self.c = config
        self.device = torch.device(device)
        x = (
            2
            * math.pi
            * torch.arange(config.tokens, device=device, dtype=torch.float64)
            / config.tokens
        )
        self.modes = torch.tensor(
            [1, 2, 3, 4, 12, 13, 14, 15], device=device, dtype=torch.float64
        )
        phase = self.modes[:, None] * x
        self.basis = torch.stack((phase.cos(), phase.sin()), 1).flatten(0, 1)
        self.positions = torch.stack(
            (x.sin(), x.cos(), (2 * x).sin(), (2 * x).cos()), -1
        )
        self.k = torch.arange(
            config.tokens // 2 + 1, device=device, dtype=torch.float64
        )
        self.low = self.k <= config.cutoff
        fine = torch.tensor(
            [config.temperature_fine] * 2 + [config.velocity_fine] * 2,
            device=device,
            dtype=torch.float64,
        )
        amplitudes = torch.cat(
            (
                torch.ones(4, 4, device=device, dtype=torch.float64),
                fine[:, None].expand(4, 4),
            ),
            1,
        )
        self.amplitudes = amplitudes.repeat_interleave(2, -1) / (
            8 * (1 + fine[:, None])
        )
        # Analytic variance for independent uniform[-1,1] Fourier coefficients.
        self.scale = math.sqrt(
            (1 + config.temperature_fine**2) / (48 * (1 + config.temperature_fine) ** 2)
        )

    def generator(self, seed):
        return torch.Generator(device=self.device).manual_seed(seed)

    def lowpass(self, value):
        return torch.fft.irfft(torch.fft.rfft(value) * self.low, n=self.c.tokens)

    def flux(self, temperature, velocity):
        if self.c.filter == "before":
            velocity = self.lowpass(velocity)
        product = torch.fft.rfft(temperature * velocity)
        # Both variants have a smooth output. "before" additionally removes
        # high-frequency velocity before multiplication.
        return (
            -torch.fft.irfft(product * (1j * self.k) * self.low, n=self.c.tokens)
            / self.c.cutoff
        )

    def draw(self, count, generator, query_index=None):
        coefficients = (
            2
            * torch.rand(
                count,
                4,
                16,
                device=self.device,
                dtype=torch.float64,
                generator=generator,
            )
            - 1
        )
        values = (coefficients * self.amplitudes) @ self.basis
        temperature, other, velocity, other_velocity = values.unbind(1)
        if self.c.main == "copy":
            main = temperature
        elif self.c.main == "affine":
            main = -temperature + 0.1
        elif self.c.main == "nonlinear":
            main = torch.sin(math.pi * temperature) / math.pi
        else:
            main = temperature + self.c.alpha * self.flux(temperature, velocity)
        aux_temperature = other if self.c.relation == "independent" else temperature
        aux_velocity = other_velocity if self.c.relation == "independent" else velocity
        if self.c.auxiliary == "transport":
            auxiliary = self.flux(aux_temperature, velocity)
        elif self.c.auxiliary == "shift":
            auxiliary = aux_velocity.roll(-1, -1)
        else:
            auxiliary = aux_velocity.roll(-1, -1) + 0.1 * self.flux(
                aux_velocity, aux_velocity
            )
        index = torch.randint(
            self.c.tokens, (count,), device=self.device, generator=generator
        )
        if query_index is not None:
            index[:] = query_index
        rows = torch.arange(count, device=self.device)
        return {
            "values": values.transpose(1, 2),
            "index": index,
            "main": main[rows, index],
            "auxiliary": auxiliary[rows, index],
            "current": temperature[rows, index],
            "main_field": main,
            "auxiliary_field": auxiliary,
        }


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.norm2 = nn.RMSNorm(width)
        hidden = 8 * width // 3
        self.gate_up = nn.Linear(width, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)

    def forward(self, x):
        batch, tokens, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(
            batch, tokens, 3, self.heads, width // self.heads
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = F.scaled_dot_product_attention(q, k, v)
        x = x + self.out(attention.transpose(1, 2).reshape(batch, tokens, width))
        gate, up = self.gate_up(self.norm2(x)).chunk(2, -1)
        return x + self.down(F.silu(gate) * up)


class Model(nn.Module):
    positions: torch.Tensor

    def __init__(self, config, positions):
        super().__init__()
        self.c = config
        d = config.width
        self.input = nn.Sequential(nn.Linear(8, d), nn.SiLU(), nn.Linear(d, d))
        self.query = nn.Sequential(nn.Linear(8, d), nn.SiLU(), nn.Linear(d, d))
        self.role = nn.Parameter(torch.randn(2, d) * 0.02)
        self.blocks = nn.ModuleList(Block(d, config.heads) for _ in range(config.depth))
        self.norm = nn.RMSNorm(d) if config.final_norm else nn.Identity()
        # Construct shared/trunk and auxiliary parameters before the variable-size
        # main head so changing bin count preserves their initialization.
        self.aux_head = nn.Linear(d, 1)
        self.head = nn.Linear(d, config.bins if config.objective == "ce" else 1)
        self.register_buffer("positions", positions.clone())

    def forward(self, batch):
        dtype = self.input[0].weight.dtype
        values = batch["values"].to(dtype)
        b, n, _ = values.shape
        position = self.positions[None].expand(b, -1, -1)
        context = self.input(torch.cat((values, position), -1))
        rows = torch.arange(b, device=values.device)
        query_values = values[rows, batch["index"]].clone()
        # The current temperature is explicitly in the query except for the
        # retrieval control. Inputs are otherwise identical across task arms.
        if self.c.path == "retrieval":
            query_values[:, 0] = 0
        query_input = torch.cat((query_values, self.positions[batch["index"]]), -1)
        query = self.query(query_input)[:, None] + self.role[None]
        if self.c.path == "mlp":
            x = query
        else:
            x = torch.cat((context, query), 1)
            for block in self.blocks:
                x = block(x)
        result = self.norm(x[:, -2:])
        output_dtype = torch.float64 if dtype == torch.float64 else torch.float32
        main = self.head(result[:, 0]).to(output_dtype)
        auxiliary = self.aux_head(result[:, 1]).squeeze(-1).to(output_dtype)
        if self.c.objective != "ce":
            main = main.squeeze(-1)
            if self.c.path == "residual":
                main = main + batch["current"].to(output_dtype)
            elif self.c.path == "wire":
                main = batch["current"].to(output_dtype)
        return main, auxiliary


def bin_labels(target, config):
    if torch.any(target.abs() > config.support):
        raise ValueError("Target outside fixed bin support; refusing to clip it")
    return (
        ((target + config.support) * config.bins / (2 * config.support)).floor().long()
    ).clamp(max=config.bins - 1)


def centers(config, device, dtype):
    return (
        -config.support
        + (torch.arange(config.bins, device=device, dtype=dtype) + 0.5)
        * 2
        * config.support
        / config.bins
    )


def losses(prediction, auxiliary, batch, config):
    target = batch["main"].to(prediction.dtype)
    if config.objective == "ce":
        main_loss = F.cross_entropy(prediction, bin_labels(target, config))
    elif config.objective == "quantized_mse":
        quantized = centers(config, target.device, target.dtype)[
            bin_labels(target, config)
        ]
        main_loss = (prediction - quantized).square().mean()
    elif config.objective == "tolerance":
        main_loss = (
            (prediction.sub(target).abs() - config.support / config.bins)
            .clamp_min(0)
            .square()
            .mean()
        )
    else:
        main_loss = (prediction - target).square().mean()
    aux_loss = (auxiliary - batch["auxiliary"].to(auxiliary.dtype)).square().mean()
    return main_loss, aux_loss


def execution(config, device):
    if config.precision == "bf16":
        return torch.autocast(
            device_type=torch.device(device).type, dtype=torch.bfloat16
        )
    return contextlib.nullcontext()


def evaluate(model, source, config, size, seed):
    model.eval()
    generator = source.generator(seed)
    errors, aux_errors, persistence, targets, predictions, expected_errors = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    auxiliary_targets = []
    correct, nll, brier, oracle = 0, 0.0, 0.0, 0.0
    with torch.no_grad(), execution(config, source.device):
        for offset in range(0, size, config.batch):
            batch = source.draw(min(config.batch, size - offset), generator)
            prediction, auxiliary = model(batch)
            target = batch["main"]
            labels = bin_labels(target, config)
            grid = centers(config, source.device, torch.float64)
            oracle += (grid[labels] - target).square().sum().item()
            if config.objective == "ce":
                probabilities = prediction.double().softmax(-1)
                predicted_labels = prediction.argmax(-1)
                correct += (predicted_labels == labels).sum().item()
                nll += F.cross_entropy(
                    prediction.double(), labels, reduction="sum"
                ).item()
                selected = probabilities.gather(1, labels[:, None]).squeeze(1)
                brier += (
                    (probabilities.square().sum(-1) - 2 * selected + 1).sum().item()
                )
                expected_errors.append((probabilities @ grid - target).cpu())
                prediction = grid[predicted_labels]
            errors.append((prediction.double() - target).cpu())
            aux_errors.append((auxiliary.double() - batch["auxiliary"]).cpu())
            auxiliary_targets.append(batch["auxiliary"].cpu())
            persistence.append((batch["current"] - target).cpu())
            targets.append(target.cpu())
            predictions.append(prediction.double().cpu())
    error = torch.cat(errors)
    rmse = error.square().mean().sqrt().item()
    persistence_rmse = torch.cat(persistence).square().mean().sqrt().item()
    auxiliary_rmse = torch.cat(aux_errors).square().mean().sqrt().item()
    auxiliary_rms = torch.cat(auxiliary_targets).square().mean().sqrt().item()
    result = {
        "count": size,
        "rmse": rmse,
        "normalized_rmse": rmse / source.scale,
        "bits": -math.log2(rmse / source.scale) if rmse else None,
        "exact": rmse == 0,
        "bias": error.mean().item(),
        "p99_abs": error.abs().quantile(0.99).item(),
        "max_abs": error.abs().max().item(),
        "auxiliary_rmse": auxiliary_rmse,
        "auxiliary_target_rms": auxiliary_rms,
        "auxiliary_normalized_rmse": auxiliary_rmse / auxiliary_rms,
        "persistence_rmse": persistence_rmse,
        "persistence_ratio": rmse / persistence_rmse if persistence_rmse else None,
        "target_rms": torch.cat(targets).square().mean().sqrt().item(),
        "quantization_rmse": math.sqrt(oracle / size),
        "reference_scale": source.scale,
        "violations": {
            str(bits): (error.abs() > source.scale * 2**-bits).double().mean().item()
            for bits in (4, 8, 12, 16, 20)
        },
    }
    if config.objective == "ce":
        result.update(
            accuracy=correct / size,
            nll=nll / size,
            brier=brier / size,
            expectation_rmse=torch.cat(expected_errors).square().mean().sqrt().item(),
        )
    model.train()
    return result


def spectrum_probe(model, source, config):
    """Assemble point forecasts on complete fields; never infer spectra from random points."""
    generator = source.generator(770003)
    batch = source.draw(8, generator)
    predictions = []
    model.eval()
    with torch.no_grad(), execution(config, source.device):
        for index in range(config.tokens):
            point = dict(batch)
            point["index"] = torch.full_like(batch["index"], index)
            point["current"] = batch["values"][:, index, 0]
            pred, _ = model(point)
            if config.objective == "ce":
                pred = centers(config, source.device, torch.float64)[pred.argmax(-1)]
            predictions.append(pred.double())
    prediction = torch.stack(predictions, -1)
    error = prediction - batch["main_field"]
    high = source.k > config.cutoff
    # norm='forward' with factor two for conjugate pairs gives mean-square power.
    power = torch.fft.rfft(error, norm="forward").abs().square()
    weights = torch.full_like(source.k, 2)
    weights[0] = weights[-1] = 1
    model.train()
    return {
        "high_frequency_error_rms": (power[:, high] * weights[high])
        .sum(-1)
        .mean()
        .sqrt()
        .item(),
        "field_rmse": error.square().mean().sqrt().item(),
        "mean_error_power_by_mode": power.mean(0).cpu().tolist(),
        "fields": 8,
    }


def gradient_probe(model, batch, config):
    """Measure shared-parameter geometry without modifying parameters or optimizer."""
    parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith(("head.", "aux_head."))
    ]
    with execution(config, batch["values"].device):
        main, auxiliary = model(batch)
        lm, la = losses(main, auxiliary, batch, config)
    gm = torch.autograd.grad(lm, parameters, retain_graph=True, allow_unused=True)
    ga = torch.autograd.grad(la, parameters, allow_unused=True)
    main_norm = (
        torch.stack([g.double().square().sum() for g in gm if g is not None])
        .sum()
        .item()
    )
    aux_norm = (
        torch.stack([g.double().square().sum() for g in ga if g is not None])
        .sum()
        .item()
    )
    dot = (
        torch.stack(
            [
                a.double().mul(b.double()).sum()
                for a, b in zip(gm, ga, strict=True)
                if a is not None and b is not None
            ]
        )
        .sum()
        .item()
    )
    return {
        "main_gradient_norm": math.sqrt(main_norm),
        "aux_gradient_norm": math.sqrt(aux_norm),
        "gradient_cosine": dot / math.sqrt(main_norm * aux_norm)
        if main_norm * aux_norm
        else None,
    }


def update_probe(model, optimizer, batch, heldout, config, weight):
    """Compare actual Adam updates from identical weights and optimizer state."""
    weights = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    state = copy.deepcopy(optimizer.state_dict())

    def score():
        with torch.no_grad(), execution(config, batch["values"].device):
            prediction, auxiliary = model(heldout)
            lm, _ = losses(prediction, auxiliary, heldout, config)
            if config.objective == "ce":
                prediction = centers(config, prediction.device, torch.float64)[
                    prediction.argmax(-1)
                ]
            return lm.item(), (
                prediction.double() - heldout["main"]
            ).square().mean().item()

    before_loss, before_mse = score()
    scores = {}
    try:
        for label, coefficient in (("main_only", 0.0), ("joint", weight)):
            model.load_state_dict(weights)
            optimizer.load_state_dict(copy.deepcopy(state))
            optimizer.zero_grad(set_to_none=True)
            with execution(config, batch["values"].device):
                prediction, auxiliary = model(batch)
                lm, la = losses(prediction, auxiliary, batch, config)
                loss = config.main_weight * lm + coefficient * la
            loss.backward()
            optimizer.step()
            after_loss, after_mse = score()
            scores[label + "_heldout_loss_change"] = after_loss - before_loss
            scores[label + "_heldout_mse_change"] = after_mse - before_mse
    finally:
        model.load_state_dict(weights)
        optimizer.load_state_dict(state)
        optimizer.zero_grad(set_to_none=True)
    scores["auxiliary_added_mse_change"] = (
        scores["joint_heldout_mse_change"] - scores["main_only_heldout_mse_change"]
    )
    return scores


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def approximate_step_flops(config):
    n, d, depth = config.tokens + 2, config.width, config.depth
    h = 8 * d // 3
    # Multiply/add=2; backward approximated as twice forward. Excludes
    # softmax, normalization, data generation, validation, and optimizer.
    forward = (
        2
        * config.batch
        * (
            depth * (4 * n * d * d + 3 * n * d * h + 2 * n * n * d)
            + (n + 2) * (8 * d + d * d)
            + d * (config.bins if config.objective == "ce" else 1)
            + d
        )
    )
    return 3 * forward


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
    generator = source.generator(config.seed + 100000)
    last_step, training_seconds, history = 0, 0.0, []
    checkpoint_path = destination / "latest.pt"
    if checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["data_rng"].cpu())
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all([v.cpu() for v in saved["cuda_rng"]])
        last_step, training_seconds, history = (
            saved["step"],
            saved["training_seconds"],
            saved["history"],
        )
    provenance = {
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
            group="precision-competition-20261107",
            id="pc-" + config.run_id,
            resume="allow",
            mode=wandb_mode,
            name=f"{config.main}-{config.objective}-d{config.width}-l{config.depth}-{config.relation}-w{config.auxiliary_weight}-s{config.seed}",
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

    def validate(step):
        metrics = evaluate(model, source, config, config.validation_size, 880001)
        if config.path != "wire":
            probe = source.draw(config.batch, source.generator(880002))
            metrics.update(gradient_probe(model, probe, config))
            if step % 1000 == 0 or step == config.steps:
                heldout = source.draw(config.batch, source.generator(880003))
                weight = (
                    config.auxiliary_weight if step > config.auxiliary_start else 0.0
                )
                metrics.update(
                    update_probe(model, optimizer, probe, heldout, config, weight)
                )
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
                # Fixed warmup followed by constant LR: one curve can measure
                # threshold costs without moving an LR-decay endpoint.
                learning_rate = config.lr * min(1.0, step / 100)
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
            result = {
                "config": asdict(config),
                "id": config.run_id,
                "step": step,
                "stop_reason": reason,
                "training_seconds": training_seconds,
                "test": test,
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
        "--wandb-mode", choices=["online", "offline", "disabled"], default="online"
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    entry = json.loads(args.manifest.read_text())[args.index]
    config = Config(**entry["config"])
    if args.smoke:
        config = replace(
            config,
            steps=30,
            eval_every=30,
            validation_size=128,
            test_size=128,
            train_seconds=180,
        )
    result = train(config, args.output / config.run_id, args.device, args.wandb_mode)
    if result.get("interrupted"):
        raise SystemExit(75)
    print(
        json.dumps(
            {
                "event": "complete",
                "id": config.run_id,
                "step": result["step"],
                "test": result["test"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
