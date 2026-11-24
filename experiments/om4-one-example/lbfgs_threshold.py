"""Matched L-BFGS test of extending the error penalty to wet mask boundaries."""

import argparse
import json
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "om4-temporal-memory"))
from common import OceanModel, Prepared, highpass, interior_mask, mse
from absolute_relative_loss import objective as interior_objective
from threshold_optimizer import make_optimizer

DEADLINE = 1792436400


def wet_neighbor_curvature(error, wet):
    numerator = torch.zeros_like(error)
    degree = torch.zeros_like(error)
    for axis, shift in ((-2, -1), (-2, 1), (-1, -1), (-1, 1)):
        valid = wet & torch.roll(wet, shift, axis)
        if axis == -2:
            valid = valid.clone()
            valid[..., -1 if shift == -1 else 0, :] = False
        numerator = numerator + (error - torch.roll(error, shift, axis)) * valid
        degree = degree + valid
    return numerator / degree.clamp_min(1), wet & (degree > 0)


def objective(prediction, target, wet, inside, weight, boundary):
    loss = interior_objective(prediction, target, wet, inside, weight)
    if boundary == "interior" or weight == 0:
        return loss
    assert boundary == "wet_neighbors"
    curvature, supported = wet_neighbor_curvature(prediction - target, wet)
    edge = supported & ~inside
    # Keep the existing interior coefficient and denominator exactly unchanged.
    extra = ((curvature.square() * edge).sum((-2, -1)) / inside.sum((-2, -1))).mean()
    return loss + weight * extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curvature-threshold", type=float, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--boundary", choices=("interior", "wet_neighbors"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-outer-steps", type=int, default=1000)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "COMPLETE").exists() or time.time() >= DEADLINE:
        return
    torch.set_num_threads(4)
    torch.manual_seed(15)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = Prepared(args.data)
    x, bs, ys = data.window(0, 1, "cuda")
    b, y = bs[0], ys[0][:, :77]
    wet = torch.tensor(data.mask[None], device="cuda")
    inside = interior_mask(wet)
    model = OceanModel(wet[0]).cuda().eval()
    optimizer_class = make_optimizer(args.curvature_threshold, out)
    optimizer = optimizer_class(
        model.parameters(), lr=1, max_iter=10, history_size=10,
        line_search_fn="strong_wolfe", tolerance_grad=1e-12,
        tolerance_change=1e-15,
    )
    resume = (out / "latest.pt").exists()
    saved = torch.load(out / "latest.pt" if resume else args.source / "latest.pt",
                       map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"])
    source_history = [json.loads(line) for line in (args.source / "history.jsonl").read_text().splitlines()]
    assert source_history[-1]["step"] == 5000 and (args.source / "COMPLETE").exists()
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        start, elapsed = saved["step"], saved["elapsed"]
        torch.set_rng_state(saved["rng_cpu"].cpu())
        torch.cuda.set_rng_state(saved["rng_cuda"].cpu())
    else:
        assert saved["step"] == 5000
        start, elapsed = 0, 0.
    del saved
    first_parameter = next(model.parameters())
    began = time.monotonic()
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, stopping)
    signal.signal(signal.SIGTERM, stopping)
    protocol = vars(args) | {
        "campaign_deadline_utc": "2026-10-19T19:00:00Z",
        "mode": "eval; original BatchNorm running statistics stay frozen",
        "optimizer_history": "fresh L-BFGS at initial source; exact history on requeue",
        "supervision": "first target, all 77 channels", "steps": args.total_outer_steps,
        "source_mse": source_history[-1]["eval_mse"],
        "source_highpass_mse": source_history[-1]["highpass_mse"],
        "initial_outer_step": 0, "source_adam_updates": 5000,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "purpose": "Exact-state continuation testing only the L-BFGS curvature acceptance threshold",
        "boundary_normalization": "Extra boundary squared error divided by original interior count per channel",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, default=str, indent=2))

    def save(step):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "step": step, "elapsed": elapsed + time.monotonic() - began,
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state()},
                   out / "latest.tmp.pt")
        (out / "latest.tmp.pt").replace(out / "latest.pt")

    def measure(step):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x, b)[0][:, :77]
        loss = mse(prediction, y, wet)
        training_loss = objective(prediction, y, wet, inside, args.weight, args.boundary)
        training_loss.backward()
        grad_squared = sum(p.grad.double().square().sum() for p in model.parameters())
        with torch.no_grad():
            error = prediction - y
            row = {
                "step": step, "seconds": elapsed + time.monotonic() - began,
                "eval_mse": float(loss), "training_objective": float(training_loss),
                "highpass_mse": float(mse(highpass(error, inside), torch.zeros_like(error), inside)),
                "objective_gradient_norm": float(grad_squared.sqrt()),
                "interior_state_mse_contribution": float(((error.square() * inside).sum((-2,-1)) / wet.sum((-2,-1))).mean()),
                "boundary_state_mse_contribution": float(((error.square() * (wet & ~inside)).sum((-2,-1)) / wet.sum((-2,-1))).mean()),
                "closure_evaluations": optimizer.state[first_parameter].get("func_evals", 0),
                "internal_iterations": optimizer.state[first_parameter].get("n_iter", 0),
                "curvature_candidates": optimizer.state[first_parameter].get("curvature_candidates", 0),
                "curvature_accepted": optimizer.state[first_parameter].get("curvature_accepted", 0),
                "last_curvature_dot": optimizer.state[first_parameter].get("last_curvature_dot"),
                "per_channel_mse": ((error.square() * wet).sum((-2, -1)) / wet.sum((-2, -1))).mean(0).tolist(),
            }
            if step in (0, 100, 250, 500, 750, args.total_outer_steps):
                np.savez_compressed(out / f"maps-{step}.npz", prediction=prediction.detach().cpu().numpy(),
                                    truth=y.cpu().numpy(), mask=data.mask)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print({k: v for k, v in row.items() if k != "per_channel_mse"}, flush=True)
        if not resume and step == 0:
            assert abs(row["eval_mse"] - source_history[-1]["eval_mse"]) < 1e-9, "Source MSE reproduction failed"
            assert abs(row["highpass_mse"] - source_history[-1]["highpass_mse"]) < 1e-10, "Source high-pass reproduction failed"

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(x, b)[0][:, :77], y, wet, inside, args.weight, args.boundary)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        return loss

    measure(start)
    for step in range(start, args.total_outer_steps):
        optimizer.step(closure)
        if (step + 1) % 25 == 0 or step + 1 == args.total_outer_steps:
            measure(step + 1)
            save(step + 1)
        if stop:
            save(step + 1)
            sys.exit(75)
        if time.time() >= DEADLINE:
            save(step + 1)
            (out / "DEADLINE").write_text("Campaign deadline reached\n")
            return
    (out / "COMPLETE").write_text("Completed matched spatial-objective L-BFGS continuation\n")


if __name__ == "__main__":
    main()
