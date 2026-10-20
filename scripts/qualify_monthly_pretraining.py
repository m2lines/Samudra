#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Qualify model-only training and an exact observation-only cooldown restart."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import ModelMonths, model_loss, task_source
from samudra.experiments.monthly_pilot_data import observation_loss
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_training import Samples


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a.cpu(), b.cpu())
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            assert_exact(left, right)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--producer", required=True)
    p.add_argument("--arm", choices=list("ABCD"), required=True)
    p.add_argument("--qualification", type=Path, required=True)
    p.add_argument("--observations", required=True)
    p.add_argument("--om4", required=True)
    p.add_argument("--cm4", required=True)
    a = p.parse_args()
    proof = json.loads(a.qualification.read_text())
    if proof["producer"] != "1c9817d44" or not proof["production_qualified"]:
        raise ValueError("Original full data qualification missing")
    for path, expected in proof["data_contracts"].items():
        if digest(path) != expected:
            raise ValueError("Original qualified data changed: " + path)
    out = a.root / "qualification" / a.arm
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        a.observations,
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=32,
    )
    data.cache_prepared = True
    stores = {
        "om4": ModelMonths(
            Path(a.om4) / "OM4.zarr",
            "om4",
            data.grid,
            a.om4,
            packed=proof["packed_om4"],
        )
    }
    if a.arm in "CD":
        stores["cm4"] = ModelMonths(a.cm4, "cm4", data.grid, a.om4)
    model = MonthlyLatentOcean().cuda()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator(device="cuda").manual_seed(1729)
    rng = np.random.default_rng(1729)
    initial = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial.update(name.encode())
        initial.update(value.detach().cpu().numpy().tobytes())
    paths = data.paths("train")
    records = []
    for source, reader in stores.items():
        for task in ["surface", "variable", "gap"] if a.arm in "BD" else ["surface"]:
            started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for _ in range(8):
                batch = reader.sample(int(rng.integers(len(reader.windows))), "cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model_loss(model, batch, task, generator, data.area)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite model training loss")
                (loss / 8).backward()
                losses.append(float(loss.detach()))
            # Model-data training must not train the observation forcing adapter.
            if any(p.grad is not None for p in model.atmosphere_adapter.parameters()):
                raise ValueError("Model training unexpectedly used the ERA5 adapter")
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            record = dict(
                source=source,
                task=task,
                loss=float(np.mean(losses)),
                gradient_norm=float(norm),
                seconds=time.monotonic() - started,
            )
            records.append(record)
            print(json.dumps(record), flush=True)
    # Match the real transition: retain Adam moments; reset EMA; decay LR.
    ema.load_state_dict(model.state_dict())
    optimizer.param_groups[0]["lr"] = 1e-4 + (1e-6 - 1e-4) / 1000

    def state():
        return dict(
            model=model.state_dict(),
            ema=ema.state_dict(),
            optimizer=optimizer.state_dict(),
            generator=generator.get_state(),
            rng=rng.bit_generator.state,
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
        )

    torch.save(state(), out / "cooldown-source.pt")

    def observation_update():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(8):
            rng.random()
            sample = data.load(paths[int(rng.integers(len(paths)))])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = observation_loss(model, data, sample, "surface", generator)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite cooldown observation loss")
            (loss / 8).backward()
            losses.append(float(loss.detach()))
        if not all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.atmosphere_adapter.parameters()
        ):
            raise ValueError("Observation adapter has missing/nonfinite gradients")
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1, error_if_nonfinite=True
        )
        optimizer.step()
        with torch.no_grad():
            for target, current in zip(
                ema.parameters(), model.parameters(), strict=True
            ):
                target.lerp_(current, 1 - 2 ** (-1 / 100))
        return dict(loss=float(np.mean(losses)), gradient_norm=float(norm))

    native = observation_update()
    torch.save(state(), out / "cooldown-native.pt")
    loaded = torch.load(
        out / "cooldown-source.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(loaded["model"])
    ema.load_state_dict(loaded["ema"])
    optimizer.load_state_dict(loaded["optimizer"])
    generator.set_state(loaded["generator"].cpu())
    rng.bit_generator.state = loaded["rng"]
    torch.set_rng_state(loaded["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in loaded["cuda_rng"]])
    replay = observation_update()
    assert_exact(native, replay)
    assert_exact(
        torch.load(out / "cooldown-native.pt", map_location="cpu", weights_only=False),
        state(),
    )
    validation = data.paths("validation")
    control = evaluate(ema, data, validation, climatology=True)
    metrics = evaluate(ema, data, validation[:1])
    selection_score(metrics, control, [], metric="rmse-only")
    # Exhaustively verify the exact scheduler used by production.
    main_sources = [
        task_source(a.arm, i, model_only_pretraining=True) for i in range(2000)
    ]
    assert "obs" not in main_sources
    assert all(
        task_source(a.arm, i, cooldown=True, model_only_pretraining=True) == "obs"
        for i in range(2000, 3000)
    )
    (out / "QUALIFIED.json").write_text(
        json.dumps(
            dict(
                producer=a.producer,
                arm=a.arm,
                initial_backbone_sha256=initial.hexdigest(),
                qualification_sha256=digest(a.qualification),
                paths={key: getattr(a, key) for key in ("observations", "om4", "cm4")},
                records=records,
                cooldown=native,
                exact_observation_cooldown_resume=True,
                main_counts={
                    key: main_sources.count(key) for key in ("obs", "om4", "cm4")
                },
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
