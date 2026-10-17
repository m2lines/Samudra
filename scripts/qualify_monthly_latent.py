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
    p.add_argument("--om4")
    p.add_argument("--cm4")
    p.add_argument("--cm4-verification")
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
    if a.cm4:
        import numpy as np
        import zarr  # type: ignore[import-untyped]

        from samudra.experiments.monthly_model_data import ModelMonths, model_loss
        from samudra.experiments.monthly_pilot_evaluation import evaluate
        from samudra.experiments.observation_metrics import selection_score

        if not a.om4 or not a.cm4_verification:
            raise ValueError("Full qualification requires CM4 read-back verification")
        transfer = json.loads(Path(a.cm4_verification).read_text())
        if transfer.get("destination") != a.cm4:
            raise ValueError("CM4 destination differs from transfer proof")
        if transfer.get(
            "verification"
        ) != "all files compared by rclone check --download" or hashlib.sha256(
            (Path(a.cm4) / ".zmetadata").read_bytes()
        ).hexdigest() != transfer.get("metadata_sha256"):
            raise ValueError("CM4 verified metadata contract changed")
        contracts = [
            Path(a.observations) / name
            for name in ("grid.npz", "statistics.npz", "SHA256SUMS")
        ]
        contracts += [
            Path(a.om4) / "OM4.zarr" / ".zmetadata",
            Path(a.cm4) / ".zmetadata",
            Path(a.cm4_verification),
        ]
        for label, field in (("means", "mean"), ("stds", "std")):
            path = Path(a.om4) / f"OM4_{label}.zarr"
            store = zarr.open_consolidated(str(path), mode="r")
            values = np.array([float(store[name][...]) for name in data.grid["names"]])
            np.testing.assert_allclose(values, data.grid[field], rtol=1e-6, atol=1e-7)
            contracts.append(path / ".zmetadata")
        result["data_contracts"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in contracts
        }
        model_checks = []
        for kind, path in [("om4", Path(a.om4) / "OM4.zarr"), ("cm4", Path(a.cm4))]:
            reader = ModelMonths(path, kind, data.grid, a.om4)
            for task in ("surface", "variable", "gap"):
                beginning = time.monotonic()
                batch = reader.sample(0, "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model_loss(model, batch, task, gen, data.area)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                if not torch.isfinite(loss) or norm <= 0:
                    raise ValueError("Model-data fitting failed")
                optimizer.step()
                torch.cuda.synchronize()
                record = dict(
                    source=kind,
                    task=task,
                    loss=float(loss.detach()),
                    seconds=time.monotonic() - beginning,
                    windows=len(reader.windows),
                    gap_exclusions=reader.excluded_gap_months,
                )
                model_checks.append(record)
                print(json.dumps(record), flush=True)
        beginning = time.monotonic()
        paths = data.paths("validation")[:1]
        metrics = evaluate(model, data, paths)
        control = evaluate(model, data, paths, climatology=True)
        value = selection_score(metrics, control, [], metric="rmse-only")
        result.update(
            phase="full-pilot-qualification",
            production_qualified=True,
            model_checks=model_checks,
            evaluation_seconds_one_origin=time.monotonic() - beginning,
            validation_smoke_score=value,
            validation_smoke=metrics,
            cm4_transfer_proof=transfer,
        )
        (out / "QUALIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "INITIAL_QUALIFIED.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
