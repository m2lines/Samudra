#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Initial real-observation fit and exact checkpoint replay; not production authorization."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_pilot_data import observation_loss
from samudra.experiments.observation_training import Samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--observations", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--producer", required=True)
    a = p.parse_args()
    out = Path(a.output)
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
        cache_size=2,
    )
    sample = data.load(sorted((Path(a.observations) / "train").glob("*.npz"))[0])
    model = MonthlyLatentOcean().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    gen = torch.Generator(device="cuda").manual_seed(1729)
    results = []
    start = time.monotonic()
    for task in ("surface", "variable", "gap"):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components = observation_loss(model, data, sample, task, gen)
        loss.backward()
        groups = {
            name: float(
                torch.stack(
                    [
                        p.grad.float().square().sum()
                        for p in part.parameters()
                        if p.grad is not None
                    ]
                )
                .sum()
                .sqrt()
            )
            for name, part in [
                ("initializer", model.initializer),
                ("processor", model.processor),
                ("decoder", model.decoder),
                ("atmosphere_adapter", model.atmosphere_adapter),
            ]
        }
        if not torch.isfinite(loss) or not all(
            v > 0 and v < float("inf") for v in groups.values()
        ):
            raise ValueError(("Invalid fit gradients", groups))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        record = dict(
            task=task,
            loss=float(loss.detach()),
            components=components,
            gradients=groups,
        )
        results.append(record)
        print(json.dumps(record), flush=True)
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    state = dict(
        model=model.state_dict(),
        optimizer=optimizer.state_dict(),
        generator=gen.get_state(),
        cpu_rng=torch.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state_all(),
        producer=a.producer,
    )
    torch.save(state, out / "diagnostic.pt")

    def trial():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = observation_loss(model, data, sample, "surface", gen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        return float(loss.detach()), {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    native = trial()
    loaded = torch.load(out / "diagnostic.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(loaded["model"])
    optimizer.load_state_dict(loaded["optimizer"])
    gen.set_state(loaded["generator"].cpu())
    torch.set_rng_state(loaded["cpu_rng"].cpu())
    torch.cuda.set_rng_state_all([v.cpu() for v in loaded["cuda_rng"]])
    replay = trial()
    if native[0] != replay[0] or any(
        not torch.equal(native[1][k], replay[1][k]) for k in native[1]
    ):
        raise ValueError("Serialization replay differs")
    result = dict(
        producer=a.producer,
        phase="initial-observation-only-qualification",
        production_qualified=False,
        results=results,
        parameters=sum(p.numel() for p in model.parameters()),
        seconds_per_microbatch=elapsed / 3,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        exact_replay=True,
        diagnostic_sha256=hashlib.sha256(
            (out / "diagnostic.pt").read_bytes()
        ).hexdigest(),
    )
    (out / "INITIAL_QUALIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
