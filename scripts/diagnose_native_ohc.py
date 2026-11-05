#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only matched train/validation decodes for native latent OHC diagnosis."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.monthly_latent import MonthlyLatentOcean
from samudra.experiments.monthly_pilot_evaluation import evaluate
from samudra.experiments.observation_training import Samples


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CaptureSamples(Samples):
    captured: list[np.ndarray]

    def ohc(self, state):
        self.captured.append(self.project_interior(state).detach().cpu().numpy()[0])
        return super().ohc(state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    torch.set_num_threads(8)
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.root / "tasks.json").read_text())
    for index, task in enumerate(manifest):
        if index % world != rank:
            continue
        out = args.root / task["id"]
        out.mkdir(exist_ok=True)
        if (out / "COMPLETE.json").exists():
            continue
        source = Path(task["root"])
        config = json.loads((source / "paths.json").read_text())
        row = config["runs"][0]
        run = source / "production" / row["name"]
        checkpoint = run / f"endpoint-{task['step']:05d}.pt"
        receipt = json.loads(
            (run / f"validation-receipt-{task['step']:05d}.json").read_text()
        )
        sha = digest(checkpoint)
        if sha != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash mismatch")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["step"] != task["step"] or saved["protocol"][
            "config_sha256"
        ] != digest(source / "paths.json"):
            raise ValueError("Checkpoint configuration mismatch")
        model = MonthlyLatentOcean(**row.get("architecture", {})).cuda().eval()
        model.load_state_dict(saved["ema" if task["weights"] == "ema" else "model"])
        del saved
        data = CaptureSamples(
            config["samples"],
            "cuda",
            surface_fill="zero",
            global_observations=True,
            cache_size=24,
        )
        data.cache_prepared = True
        paths = (
            data.paths("validation")
            if task["split"] == "validation"
            else [data.root / "train" / (name + ".npz") for name in task["origins"]]
        )
        if not all(p.is_file() for p in paths) or len(paths) != 9:
            raise ValueError("Incomplete fixed nine-origin cohort")
        data.captured = []
        metrics = evaluate(model, data, paths, prediction_path=out / "predictions.npz")
        if task["split"] == "validation":
            for key, expected in receipt["weights"][task["weights"]]["metrics"].items():
                if not np.isclose(
                    metrics["metrics"][key], expected, rtol=1e-6, atol=1e-8
                ):
                    raise ValueError(f"Reference evaluation changed: {key}")
        samples = [data.load(p) for p in paths]
        np.savez_compressed(
            out / "interior.npz",
            prediction=np.stack(data.captured),
            reference=np.stack([s["interior"][0].cpu().numpy() for s in samples]),
            climatology=np.stack(
                [data.interior_climatology[int(p.stem[5:7]) - 1] for p in paths]
            ),
            mask=data.interior_support.cpu().numpy(),
            area=data.interior_area.cpu().numpy(),
            lat=data.grid["lat"],
            lon=data.grid["lon"],
            ohc_area=data.ohc_area.cpu().numpy(),
            ohc_support=data.ohc_support.cpu().numpy(),
        )
        result = dict(
            task=task,
            checkpoint_sha256=sha,
            origins=[p.stem for p in paths],
            metrics=metrics,
            files={
                name: digest(out / name) for name in ("predictions.npz", "interior.npz")
            },
        )
        (out / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(dict(task=task["id"], complete=True)), flush=True)
        del model, data, samples
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
