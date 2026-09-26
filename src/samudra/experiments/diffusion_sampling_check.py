# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Paired 8/16/32-step sampling diagnostics on frozen validation observations."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from samudra.experiments.diffusion_calibration import observation_ensemble_statistics
from samudra.experiments.diffusion_physical import JointPhysicalForecast
from samudra.experiments.observation_model import ObservationTransfer
from samudra.experiments.observation_pilot import atomic_json, digest
from samudra.experiments.observation_training import Samples


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--members", type=int, default=8)
    args = parser.parse_args()
    if args.members < 2:
        parser.error("At least two members required")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    signature = saved["protocol"]["signature"]
    if signature["arm"] != "B" or signature["phase"] not in ("om4", "observation"):
        raise ValueError("This diagnostic requires a trained B checkpoint")
    marker = (
        "PRETRAIN_COMPLETE.json"
        if signature["phase"] == "om4"
        else "OBSERVATION_COMPLETE.json"
    )
    complete = json.loads((args.checkpoint.parent / marker).read_text())
    checksum = digest(args.checkpoint)
    if (
        not complete["state"]["complete"]
        or complete["best_checkpoint_sha256"] != checksum
    ):
        raise ValueError("Require completed validation-selected weights")
    data = Samples(args.root / "data/observations", "cuda")
    data.use_observation_normalization()
    if digest(data.root / "SHA256SUMS") != signature["observation_manifest_sha256"]:
        raise ValueError("Observation data differ from training")
    paths = data.paths("validation")
    if len(paths) != 9:
        raise ValueError("Require all nine frozen validation origins")
    model = (
        JointPhysicalForecast(
            ObservationTransfer(list(data.grid["names"]), "instance"), stochastic=True
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(saved["model"], strict=True)
    del saved
    protocol = dict(
        checkpoint_sha256=checksum,
        training_protocol=signature,
        evaluator_commit=os.environ["SAMUDRA_CODE_COMMIT"],
        members=args.members,
        seed=4041729,
        sampling_steps=[8, 16, 32],
        origins=[p.stem for p in paths],
        scope="Paired sampling sensitivity on validation only; 32 steps is a comparison, not exact truth",
    )
    args.output.mkdir(parents=True, exist_ok=True)
    contract = args.output / "protocol.json"
    if contract.exists() and json.loads(contract.read_text()) != protocol:
        raise ValueError("Sampling-check protocol differs")
    atomic_json(protocol, contract)
    for path in paths:
        sample = data.load(path)
        reference = None
        results = {}
        for steps in (32, 16, 8):
            model.sampling_steps = steps
            torch.cuda.synchronize()
            started = time.monotonic()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction, _ = model.forecast(
                    sample["surface"],
                    sample["atmosphere"],
                    sample["contexts"],
                    data.mask,
                    sample["validity"],
                    members=args.members,
                    generator=torch.Generator(device="cuda").manual_seed(
                        protocol["seed"]
                    ),
                )
            if not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError("Nonfinite sampled trajectory")
            mean = prediction.float().mean(0)
            if reference is None:
                reference = mean.clone()
            weight = data.mask * data.area
            difference = ((mean - reference).square() * weight).sum((-2, -1))
            denominator = weight.sum((-2, -1)).clamp_min(1e-12)
            statistics = observation_ensemble_statistics(data, prediction, sample)
            torch.cuda.synchronize()
            results[str(steps)] = dict(
                elapsed_seconds=time.monotonic() - started,
                mean_difference_from_32_mse=(difference / denominator).cpu().tolist(),
                statistics={
                    group: {
                        key: value.cpu().tolist()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in fields.items()
                    }
                    for group, fields in statistics.items()
                },
            )
            del prediction, mean
        atomic_json(
            dict(origin=path.stem, results=results), args.output / (path.stem + ".json")
        )
        print(
            json.dumps(dict(event="sampling_check_origin", origin=path.stem)),
            flush=True,
        )
    atomic_json(protocol, args.output / "COMPLETE.json")


if __name__ == "__main__":
    main()
