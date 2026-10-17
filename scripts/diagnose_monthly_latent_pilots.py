#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only pilot gradient probes and native-window map exports; never updates weights."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_model_data import ModelMonths, model_loss
from samudra.experiments.monthly_pilot_data import observation_batch, observation_loss
from samudra.experiments.observation_training import Samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.manual_seed(7921)
    torch.use_deterministic_algorithms(True)
    proof = json.loads(
        (a.root / "initial-qualification-214096/QUALIFIED.json").read_text()
    )
    for path, digest in proof["data_contracts"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
    protocol = json.loads((a.root / "A/protocol.json").read_text())
    data = Samples(
        protocol["observations"],
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=32,
    )
    data.cache_prepared = True
    om4 = ModelMonths(
        Path(protocol["om4"]) / "OM4.zarr",
        "om4",
        data.grid,
        protocol["om4"],
        packed=proof["packed_om4"],
    )
    model = MonthlyLatentOcean().cuda()
    train = data.paths("train")
    result: dict[str, Any] = {
        "gradient_probes": [],
        "checkpoints": {},
        "map_note": "Native five-day-bin means at leads 5 and 30; native OHC is an exploratory readout, not monthly OHC validation. Obs-only has no native interior supervision.",
    }
    for step in (2000, 3000):
        path = a.root / "A" / f"endpoint-{step:05d}.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["step"] == step
        model.load_state_dict(saved["model"])
        model.train()
        groups = {
            name: [p for n, p in model.named_parameters() if n.startswith(prefix)]
            for name, prefix in [
                ("initializer", "initializer."),
                ("processor", "processor."),
                ("decoder", "decoder."),
                ("adapter", "atmosphere_adapter."),
            ]
        }
        assert all(groups.values()), list(dict(model.named_parameters()))
        gradients: dict[str, list[torch.Tensor]] = {k: [] for k in groups}
        records = []
        for source in ("obs", "om4"):
            for repeat in range(4):
                model.zero_grad(set_to_none=True)
                gen = torch.Generator(device="cuda").manual_seed(83000 + repeat)
                rng = np.random.default_rng(41000 + repeat)
                losses = []
                components = []
                for _ in range(8):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if source == "obs":
                            loss, component = observation_loss(
                                model,
                                data,
                                data.load(train[int(rng.integers(len(train)))]),
                                "surface",
                                gen,
                            )
                            components.append(component)
                        else:
                            loss = model_loss(
                                model,
                                om4.sample(int(rng.integers(len(om4.windows))), "cuda"),
                                "surface",
                                gen,
                                data.area,
                            )
                    assert torch.isfinite(loss)
                    (loss / 8).backward()
                    losses.append(float(loss.detach()))
                record: dict[str, Any] = {
                    "source": source,
                    "repeat": repeat,
                    "loss": float(np.mean(losses)),
                    "components": components,
                    "norms": {},
                }
                for name, params in groups.items():
                    vec = torch.cat(
                        [
                            (torch.zeros_like(p) if p.grad is None else p.grad)
                            .detach()
                            .float()
                            .flatten()
                            .cpu()
                            for p in params
                        ]
                    )
                    gradients[name].append(vec)
                    record["norms"][name] = float(vec.norm())
                records.append(record)
        alignment = {}
        for name, values in gradients.items():
            # Four independent batch/noise replicates per source. Compare cross-source
            # alignment with within-source stochastic variation, before global clipping.
            matrix = torch.stack(values)
            unit = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-30)
            cosine = unit @ unit.T
            alignment[name] = {
                "cosine_matrix": cosine.tolist(),
                "obs_om4_mean": float(cosine[:4, 4:].mean()),
                "obs_obs_mean": float(
                    cosine[:4, :4][
                        torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)
                    ].mean()
                ),
                "om4_om4_mean": float(
                    cosine[4:, 4:][
                        torch.triu(torch.ones(4, 4, dtype=torch.bool), 1)
                    ].mean()
                ),
            }
        result["gradient_probes"].append(
            {"step": step, "weights": "raw", "records": records, "alignment": alignment}
        )
        (a.output / "diagnostics.json").write_text(json.dumps(result, indent=2))
    del gradients, matrix, values, saved
    model.zero_grad(set_to_none=True)
    for arm in "ABCDE":
        path = a.root / arm / "endpoint-03000.pt"
        result["checkpoints"][arm] = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = torch.load(path, map_location="cpu", weights_only=False)
        assert saved["step"] == 3000
        model.load_state_dict(saved["ema"])
        model.eval()
        for origin in (0, 5):
            sample = data.load(data.paths("validation")[origin])
            gen = torch.Generator(device="cuda").manual_seed(904001 + origin)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                batch = observation_batch(data, sample, "surface", gen)
                states = model.trajectory(
                    batch["fields"],
                    batch["available"],
                    model.adapt(sample["atmosphere"]),
                    sample["contexts"],
                )
                native_support = torch.zeros_like(data.mask, dtype=torch.bool)
                native_support[[38, 76]] = data.mask[[38, 76]].bool()
                interior_support = torch.zeros_like(native_support)
                interior_support[data.ts_indices] = data.ts_mask.bool()
                latent_norms = [float(s.float().square().mean().sqrt()) for s in states]
                for lead in (1, 6):
                    latent = states[lead].expand(8, -1, -1, -1, -1)
                    surface = model.decoder.sample(
                        latent,
                        native_support,
                        gen,
                        steps=32,
                        position_days=lead * 5 - 2.5,
                    ).mean(0, keepdim=True)
                    interior = model.decoder.sample(
                        latent,
                        interior_support,
                        gen,
                        steps=32,
                        position_days=lead * 5 - 2.5,
                    ).mean(0, keepdim=True)
                    interior[:, 38] = surface[:, 38]
                    physical = (
                        data.physical(surface[:, None])[0, 0, [38, 76]]
                        .float()
                        .cpu()
                        .numpy()
                    )
                    ohc = data.ohc(interior)[0]
                    np.savez_compressed(
                        a.output / f"{arm}-{sample['name']}-day{lead * 5}.npz",
                        surface=physical,
                        ohc=ohc,
                        reference_surface=sample["raw"]["surface"][18 + lead],
                        monthly_reference_ohc=sample["raw"]["ohc"],
                        lat=data.grid["lat"],
                        lon=data.grid["lon"],
                        latent_rms=latent_norms,
                    )
    (a.output / "diagnostics.json").write_text(json.dumps(result, indent=2))
    (a.output / "COMPLETE.json").write_text(
        json.dumps(
            {
                "arms": list("ABCDE"),
                "origins": 2,
                "map_files": 20,
                "training_updates": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
