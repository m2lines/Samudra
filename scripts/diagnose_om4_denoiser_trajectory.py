#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare saved sampler inputs with training-style corrupted gold, no training."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from samudra.experiments.diffusion_noise import diffusion_noise
from samudra.experiments.monthly_latent import WindowDecoder


def sha(path):
    return hashlib.file_digest(open(path, "rb"), "sha256").hexdigest()


def channel_rmse(a, b, support):
    valid = support.bool() & torch.isfinite(a) & torch.isfinite(b)
    n = valid.sum((-2, -1))
    error = torch.where(valid, (a.double() - b.double()).square(), 0).sum((-2, -1))
    return (
        torch.where(n > 0, (error / n.clamp_min(1)).sqrt(), torch.nan)[0].cpu().tolist()
    )


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stages", nargs="+", default=["initial", "day5", "monthly"])
    parser.add_argument("--calls", nargs="+", type=int, default=list(range(0, 63, 2)))
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[905101, 905102, 905103]
    )
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((args.bundle / "COMPLETE.json").read_text())
    expected = manifest["cases"]["om4"]["checkpoint_sha256"]
    assert sha(args.checkpoint) == expected
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert saved["step"] == 2000
    decoder = WindowDecoder().cuda().eval()
    decoder.load_state_dict(
        {
            k.removeprefix("decoder."): v
            for k, v in saved["ema"].items()
            if k.startswith("decoder.")
        },
        strict=True,
    )
    del saved

    def tensor(a):
        return torch.as_tensor(a, device="cuda")[None]

    metadata = dict(
        checkpoint_sha256=expected,
        weights="ema",
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        seeds=args.seeds,
        calls=args.calls,
        source_sha256=sha(Path(__file__)),
        noise_correlation=decoder.core.noise_correlation,
        note="Same noise draw reused across sigmas within each seed; no teacher input enters latent evolution. Predictor calls only; Heun correctors share noise levels.",
    )
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for stage in args.stages:
        directory = args.bundle / "om4" / stage
        with np.load(directory / "final.npz") as final:
            gold, support, latent = (
                tensor(final["gold"]).float(),
                tensor(final["support"]).bool(),
                tensor(final["latent"]),
            )
        clean = torch.where(support, gold, 0)
        meta = json.loads((directory / "trace.json").read_text())
        with torch.autocast("cuda", dtype=torch.bfloat16):
            conditioning = decoder.condition(
                latent, support, meta["duration_days"], meta["position_days"]
            )
        noises = [
            diffusion_noise(
                gold.shape,
                support,
                torch.Generator(device="cuda").manual_seed(seed),
                decoder.core.noise_correlation,
            )
            for seed in args.seeds
        ]
        results = []
        for call in args.calls:
            trace_path = directory / f"call-{call:02d}.npz"
            with np.load(trace_path) as trace:
                noisy, original = (
                    tensor(trace["noisy"]).float(),
                    tensor(trace["clean"]).float(),
                )
                sigma = torch.tensor([float(trace["sigma"])], device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                replay = decoder.core(noisy, sigma, conditioning, support)
            original_error = channel_rmse(original, gold, support)
            replay_error = channel_rmse(replay, gold, support)
            replay_delta = channel_rmse(replay, original, support)
            # Gate the cross-GPU/runtime replay before interpreting gold corruption.
            assert np.allclose(replay_error, original_error, rtol=0.01, atol=1e-4), (
                stage,
                call,
                "saved/local RMSE differs >1%+1e-4",
            )
            assert max(replay_delta) < 0.02, (
                stage,
                call,
                "saved/local normalized field RMS differs >0.02",
            )
            gold_errors, input_errors = [], []
            for noise in noises:
                corrupted = (clean + sigma[:, None, None, None] * noise) * support
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    predicted = decoder.core(corrupted, sigma, conditioning, support)
                gold_errors.append(channel_rmse(predicted, gold, support))
                input_errors.append(channel_rmse(corrupted, gold, support))
            record = dict(
                stage=stage,
                call=call,
                sigma=float(sigma),
                saved_rmse=original_error,
                local_replay_rmse=replay_error,
                local_replay_delta=replay_delta,
                corrupted_gold_rmse=gold_errors,
                corrupted_input_rmse=input_errors,
                trace_sha256=sha(trace_path),
            )
            results.append(record)
            print(
                json.dumps(
                    dict(
                        stage=stage,
                        call=call,
                        seconds=round(time.monotonic() - started, 1),
                        sst_saved=original_error[38],
                        sst_gold=[v[38] for v in gold_errors],
                    )
                ),
                flush=True,
            )
            (args.output / f"{stage}.json").write_text(
                json.dumps(results, indent=2, allow_nan=False)
            )
    (args.output / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                **metadata,
                stages=args.stages,
                seconds=time.monotonic() - started,
                replay_gate_passed=True,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
