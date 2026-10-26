#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate saved raw/EMA checkpoints without interrupting or altering training.

Distribute whole checkpoints across GPUs, preserving the complete ordered origin
list and original fixed diffusion seeds independently for every evaluation.
"""

import argparse
import hashlib
import json
import math
import shutil
import time
from pathlib import Path

import torch

from samudra.experiments.monthly_distributed import DistributedUpdate
from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_metrics import selection_score
from samudra.experiments.observation_training import Samples


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assigned_steps(rank, world, end=5000):
    return list(range(500, end + 1, 500))[rank::world]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    runtime = DistributedUpdate()
    config = json.loads((args.root / "paths.json").read_text())
    row = next(r for r in config["runs"] if r["name"] == args.name)
    if row["mode"] != "shared-4k" or row.get("validation_schedule") != "posthoc-500":
        raise ValueError("Unexpected deferred-validation contract")
    out = args.root / "production" / args.name
    complete = json.loads((out / "TRAIN_COMPLETE.json").read_text())
    if complete["step"] != 5000 or complete["producer"] != config["producer"]:
        raise ValueError("Training completion contract differs")
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol["config_sha256"] != digest(args.root / "paths.json"):
        raise ValueError("Frozen configuration changed")
    torch.set_num_threads(8)
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    data = Samples(
        config["samples"],
        "cuda",
        surface_fill="zero",
        global_observations=True,
        cache_size=256,
    )
    data.cache_prepared = True
    paths = data.paths("validation")
    model = MonthlyLatentOcean(**row.get("architecture", {})).cuda().eval()
    # Eager decoding deliberately matches the qualified reference evaluator.
    control = evaluate(model, data, paths, climatology=True)
    for step in assigned_steps(runtime.rank, runtime.world):
        checkpoint = out / f"endpoint-{step:05d}.pt"
        sha = digest(checkpoint)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["protocol"] != protocol or saved["step"] != step:
            raise ValueError("Checkpoint protocol/step differs")
        records = {}
        for weights in ("ema", "model"):
            label = "ema" if weights == "ema" else "raw"
            started = time.monotonic()
            model.load_state_dict(saved[weights])
            metrics = evaluate(model, data, paths)
            blend = selection_score(metrics, control, [], metric="rmse-only")
            if not math.isfinite(blend):
                raise ValueError("Nonfinite validation score")
            suffix = "" if label == "ema" else "raw-"
            target = out / f"validation-{suffix}{step:05d}.json"
            temporary = target.with_suffix(".tmp")
            temporary.write_text(json.dumps(metrics, indent=2) + "\n")
            temporary.replace(target)
            records[label] = dict(
                blend=blend,
                seconds=time.monotonic() - started,
                metrics=metrics["metrics"],
            )
        receipt = dict(
            step=step,
            checkpoint_sha256=sha,
            producer=config["producer"],
            origins=[p.stem for p in paths],
            members=8,
            sampler_steps=32,
            weights=records,
            rank=runtime.rank,
        )
        (out / f"validation-receipt-{step:05d}.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(json.dumps(receipt), flush=True)
        del saved
    runtime.barrier()
    if runtime.rank == 0:
        receipts = [
            json.loads((out / f"validation-receipt-{s:05d}.json").read_text())
            for s in range(500, 5001, 500)
        ]
        selected = min(receipts, key=lambda r: r["weights"]["ema"]["blend"])
        source = out / f"endpoint-{selected['step']:05d}.pt"
        shutil.copyfile(source, out / "best.pt.tmp")
        if digest(out / "best.pt.tmp") != selected["checkpoint_sha256"]:
            raise ValueError("Selected checkpoint copy changed")
        (out / "best.pt.tmp").replace(out / "best.pt")
        # Use a distinct x-axis: logging old optimizer steps on a resumed W&B
        # run would discard the post-hoc curve after its final training step.
        import wandb

        run = wandb.init(
            entity=config.get("wandb_entity", "ocean_emulators"),
            project="observational-transfer",
            name="monthly-" + args.name,
            id=hashlib.sha256(str(out).encode()).hexdigest()[:12],
            resume="allow",
            dir=str(out),
            mode="online",
        )
        run.define_metric("posthoc/checkpoint_step")
        run.define_metric("posthoc/*", step_metric="posthoc/checkpoint_step")
        for receipt in receipts:
            record = {"posthoc/checkpoint_step": receipt["step"]}
            for label, result in receipt["weights"].items():
                record[f"posthoc/{label}/rmse_blend"] = result["blend"]
                for name, value in result["metrics"].items():
                    record[f"posthoc/{label}/{name}"] = value
            run.log(record)
        run.finish()
        result = dict(
            producer=config["producer"],
            selection="original rmse-only blend",
            selection_cadence=500,
            selected_step=selected["step"],
            selected_sha256=selected["checkpoint_sha256"],
            best=selected["weights"]["ema"]["blend"],
            checkpoints=receipts,
        )
        (out / "EVALUATION_COMPLETE.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
    runtime.barrier()


if __name__ == "__main__":
    main()
