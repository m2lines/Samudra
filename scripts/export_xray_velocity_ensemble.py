#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Eight fixed-seed initializer draws, three cases, four useful GPU workers."""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.observation_pilot import atomic_json, digest

CHANNELS = [0, 7, 19, 26]
SEEDS = list(range(905001, 905009))


def verify_trace(root, case, trace):
    manifest = json.loads((root / "COMPLETE.json").read_text())
    for name in ("grid.npz", f"{case}/{trace}/final.npz", f"{case}/{trace}/trace.json"):
        if digest(root / name) != manifest["files"][name]:
            raise ValueError(f"Saved X-ray file changed: {name}")
    return manifest["cases"][case]


@torch.no_grad()
def worker(root, rank):
    config = json.loads((root / "config.json").read_text())
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    directory = root / f"worker-{rank}"
    directory.mkdir()
    records = []
    for row in config["cases"]:
        bundle = Path(row["bundle"])
        source = verify_trace(bundle, row["case"], row["trace"])
        checkpoint = Path(source["checkpoint"])
        if digest(checkpoint) != source["checkpoint_sha256"]:
            raise ValueError("Checkpoint checksum changed")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["step"] != source["step"]:
            raise ValueError("Checkpoint step mismatch")
        model = (
            MonthlyLatentOcean(**saved["protocol"].get("architecture", {}))
            .cuda()
            .eval()
        )
        model.load_state_dict(saved["ema"], strict=True)
        trace_root = bundle / row["case"] / row["trace"]
        metadata = json.loads((trace_root / "trace.json").read_text())
        if metadata["seed"] != SEEDS[0] or metadata["steps"] != 32:
            raise ValueError("Unexpected original sampler contract")
        with np.load(trace_root / "final.npz") as saved_trace:
            # Initializer was autocast bf16; NPZ records its exact float32 values.
            latent = (
                torch.from_numpy(saved_trace["latent"])[None].cuda().to(torch.bfloat16)
            )
            if not np.array_equal(
                latent.float().cpu().numpy()[0], saved_trace["latent"]
            ):
                raise ValueError("Saved initializer latent is not exactly bf16")
            support = torch.from_numpy(saved_trace["support"])[None].cuda().bool()
            original = saved_trace["prediction"]
        for member in range(rank, len(SEEDS), 4):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.decoder.sample(
                    latent,
                    support,
                    torch.Generator(device="cuda").manual_seed(SEEDS[member]),
                    steps=32,
                    duration_days=metadata["duration_days"],
                    position_days=metadata["position_days"],
                )
            values = result[0].float().cpu().numpy()
            if member == 0:
                np.testing.assert_array_equal(values, original)
            path = directory / f"{row['name']}-{member}.npz"
            np.savez_compressed(path, normalized=values[CHANNELS])
            records.append(
                dict(
                    case=row["name"],
                    member=member,
                    seed=SEEDS[member],
                    file=str(path),
                    sha256=digest(path),
                    member0_exact=member == 0,
                )
            )
            print(json.dumps(records[-1]), flush=True)
        del model, saved
        torch.cuda.empty_cache()
    atomic_json(records, directory / "COMPLETE.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    args = parser.parse_args()
    if args.rank is not None:
        worker(args.root, args.rank)
        return

    def launch(rank):
        with (args.root / f"rank-{rank}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--root",
                    str(args.root),
                    "--rank",
                    str(rank),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(rank)},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        return json.loads((args.root / f"worker-{rank}/COMPLETE.json").read_text())

    with ThreadPoolExecutor(max_workers=4) as pool:
        records: list[dict] = sum(pool.map(launch, range(4)), [])
    config = json.loads((args.root / "config.json").read_text())
    arrays = {}
    for row in config["cases"]:
        members = sorted(
            (r for r in records if r["case"] == row["name"]), key=lambda r: r["member"]
        )
        if [r["seed"] for r in members] != SEEDS or not members[0]["member0_exact"]:
            raise ValueError("Incomplete or unverified ensemble")
        values = []
        for record in members:
            if digest(record["file"]) != record["sha256"]:
                raise ValueError("Member checksum mismatch")
            with np.load(record["file"]) as member:
                values.append(member["normalized"])
        with np.load(Path(row["bundle"]) / "grid.npz") as grid:
            physical = (
                np.stack(values).astype(np.float64) * grid["std"][CHANNELS, None, None]
                + grid["mean"][CHANNELS, None, None]
            )
            arrays[row["name"]] = np.where(
                grid["mask"][CHANNELS], physical.mean(0), np.nan
            )
            arrays[row["name"] + "_std"] = np.where(
                grid["mask"][CHANNELS], physical.std(0), np.nan
            )
            arrays["lat"], arrays["lon"] = grid["lat"], grid["lon"]
            arrays[row["name"] + "_support"] = grid["mask"][CHANNELS]
            with np.load(
                Path(row["bundle"]) / row["case"] / row["trace"] / "final.npz"
            ) as trace:
                gold = (
                    trace["gold"][CHANNELS].astype(np.float64)
                    * grid["std"][CHANNELS, None, None]
                    + grid["mean"][CHANNELS, None, None]
                )
                gold = np.where(grid["mask"][CHANNELS], gold, np.nan)
                if "gold" in arrays:
                    np.testing.assert_array_equal(arrays["gold"], gold)
                arrays["gold"] = gold
    arrays["channels"] = np.array(CHANNELS)
    output = args.root / "velocity-means.npz"
    np.savez_compressed(output, **arrays)
    atomic_json(
        dict(
            producer=os.environ["SAMUDRA_CODE_COMMIT"],
            config_sha256=digest(args.root / "config.json"),
            members=records,
            channels=CHANNELS,
            seeds=SEEDS,
            weights="ema",
            steps=32,
            stage="initializer",
            output_sha256=digest(output),
        ),
        args.root / "COMPLETE.json",
    )


if __name__ == "__main__":
    main()
