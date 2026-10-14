# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Longer independent-context controls with exact original schedule prefixes."""

import hashlib
import json
import math
import time
from pathlib import Path

import run as experiment
import torch

BASE = Path(__file__).resolve().parent
ORIGINAL = BASE / "outputs/main"
OUT = BASE / "outputs/longer"


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def gradients(data, scales):
    schedule = experiment.make_schedule(len(data), 1, 4, 60101, "cuda")
    c, f, _, ft, origins = experiment.batch_at(data, schedule, 0)
    result = {}
    for arm in ["shared_context", "independent_context"]:
        model = experiment.Model(*experiment.ARMS[arm]).cuda()
        model.load_state_dict(
            torch.load(ORIGINAL / f"{arm}_seed0/best.pt", weights_only=True)
        )
        parameters = tuple(model.global_route.core.parameters())
        vectors = []
        for detach in [False, True]:
            _, global_features = model.global_step(c)
            if detach:
                global_features = global_features.detach()
            fp = model.fine_step(f, global_features, origins)
            loss = (
                experiment.interior(fp) - experiment.interior(ft)
            ).square().mean() / scales[1]
            grads = torch.autograd.grad(loss, parameters, allow_unused=True)
            vectors.append(
                torch.cat(
                    [
                        (torch.zeros_like(p) if g is None else g).flatten()
                        for p, g in zip(parameters, grads)
                    ]
                )
            )
        total, direct = vectors
        result[arm] = {
            "fine_loss_global_core_gradient_norm": total.norm().item(),
            "with_context_detached_gradient_norm": direct.norm().item(),
            "context_path_gradient_norm": (total - direct).norm().item(),
        }
    assert result["independent_context"]["with_context_detached_gradient_norm"] == 0
    assert result["shared_context"]["with_context_detached_gradient_norm"] > 0
    assert all(r["context_path_gradient_norm"] > 0 for r in result.values())
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, val, test = torch.load(
        ORIGINAL / "data.pt", map_location="cuda", weights_only=True
    )
    scales = json.loads((ORIGINAL / "loss_scales.json").read_text())
    original = json.loads((ORIGINAL / "results.json").read_text())
    provenance = {
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_source_sha256": hashlib.sha256(
            (BASE / "run.py").read_bytes()
        ).hexdigest(),
        "original_provenance": json.loads((ORIGINAL / "provenance.json").read_text()),
        "schedules": {
            "prefix": "make_schedule(n,2000,4,30000+seed)",
            "extension": "make_schedule(n,2000,4,50000+seed)",
        },
        "learning_rates": {
            "continue_floor": "original cosine to update 2000, then constant 0.0002",
            "cosine_4000": "cosine 0.001 to 0.0002 over 4000 updates",
        },
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "runs": 6,
        "steps_per_run": 4000,
    }
    write(OUT / "provenance.json", provenance)
    (OUT / "longer_source.py").write_bytes(Path(__file__).read_bytes())
    write(OUT / "gradient_paths.json", gradients(train, scales))
    validation = experiment.make_schedule(len(val), 8, 4, 20001, "cuda", max_time=5)
    testing = experiment.make_schedule(len(test), 16, 4, 20002, "cuda", max_time=5)
    results = []
    for seed in [0, 1, 2]:
        first = experiment.make_schedule(len(train), 2000, 4, 30000 + seed, "cuda")
        second = experiment.make_schedule(len(train), 2000, 4, 50000 + seed, "cuda")
        schedule = [torch.cat((a, b)) for a, b in zip(first, second)]
        for variant in ["continue_floor", "cosine_4000"]:
            rd = OUT / f"{variant}_seed{seed}"
            rd.mkdir(exist_ok=True)
            if (rd / "result.json").exists():
                raise ValueError("Existing completed run: use a fresh output directory")
            torch.manual_seed(40000 + seed)
            model = experiment.Model(False, True).cuda()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=0.001, weight_decay=1e-4
            )
            best, best_step, history, replay = float("inf"), 0, [], None
            prediction_difference = None
            torch.cuda.synchronize()
            start = time.monotonic()
            for step in range(4000):
                model.train()
                c, f, ct, ft, origins = experiment.batch_at(train, schedule, step)
                optimizer.zero_grad(set_to_none=True)
                cp, fp = model(c, f, origins)
                lc = (cp - ct).square().mean() / scales[0]
                lf = (
                    experiment.interior(fp) - experiment.interior(ft)
                ).square().mean() / scales[1]
                loss = 0.5 * (lc + lf)
                assert torch.isfinite(loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 5.0, error_if_nonfinite=True
                )
                denominator = 2000 if variant == "continue_floor" else 4000
                lr = 0.0002 + 0.0008 * 0.5 * (
                    1 + math.cos(math.pi * min(step, denominator) / denominator)
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                if (step + 1) % 100 == 0:
                    met = experiment.evaluate(model, val, validation, leads=(1,))["1"]
                    score = 0.5 * (
                        met["coarse"] ** 2 / scales[0] + met["fine"] ** 2 / scales[1]
                    )
                    history.append(
                        {
                            "step": step + 1,
                            "val_score": score,
                            "val_coarse_rmse": met["coarse"],
                            "val_fine_rmse": met["fine"],
                            "lr": lr,
                        }
                    )
                    if score < best:
                        best, best_step = score, step + 1
                        torch.save(model.state_dict(), rd / "best.pt")
                    if (step + 1) % 500 == 0:
                        print(
                            json.dumps(
                                {
                                    "variant": variant,
                                    "seed": seed,
                                    **history[-1],
                                    "seconds": round(time.monotonic() - start, 1),
                                }
                            ),
                            flush=True,
                        )
                if step + 1 == 2000 and variant == "continue_floor":
                    old_hist = json.loads(
                        (
                            ORIGINAL / f"independent_context_seed{seed}/history.json"
                        ).read_text()
                    )
                    replay = max(
                        abs(a["val_score"] - b["val_score"])
                        for a, b in zip(history, old_hist)
                    )
                    assert replay < 1e-4, replay
                    old = next(
                        r
                        for r in original
                        if r["arm"] == "independent_context" and r["seed"] == seed
                    )
                    assert best_step == old["best_step"]
                    old_state = torch.load(
                        ORIGINAL / f"independent_context_seed{seed}/best.pt",
                        weights_only=True,
                    )
                    state = torch.load(rd / "best.pt", weights_only=True)
                    old_model = experiment.Model(False, True).cuda()
                    replay_model = experiment.Model(False, True).cuda()
                    old_model.load_state_dict(old_state)
                    replay_model.load_state_dict(state)
                    vc, vf, _, _, vo = experiment.batch_at(val, validation, 0)
                    with torch.no_grad():
                        prediction_difference = max(
                            (a - b).abs().max().item()
                            for a, b in zip(
                                old_model(vc, vf, vo), replay_model(vc, vf, vo)
                            )
                        )
                    assert prediction_difference < 1e-3, prediction_difference
                    del old_model, replay_model
                    print(
                        f"PREFIX_CHECK seed={seed} maximum validation score difference={replay}",
                        flush=True,
                    )
            torch.cuda.synchronize()
            seconds = time.monotonic() - start
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": 4000,
                },
                rd / "last.pt",
            )
            model.load_state_dict(torch.load(rd / "best.pt", weights_only=True))
            metrics = experiment.evaluate(model, test, testing)
            result = {
                "arm": "independent_context",
                "variant": variant,
                "seed": seed,
                "best_step": best_step,
                "best_val": best,
                "seconds": seconds,
                "prefix_max_validation_difference": replay,
                "prefix_max_prediction_difference": prediction_difference,
                "test": metrics,
            }
            write(rd / "history.json", history)
            write(rd / "result.json", result)
            results.append(result)
            write(OUT / "results.json", results)
            print("RESULT " + json.dumps(result), flush=True)
    write(OUT / "COMPLETE.json", {"runs": len(results)})


if __name__ == "__main__":
    main()
