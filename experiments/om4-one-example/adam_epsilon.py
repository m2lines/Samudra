"""Matched Adam epsilon continuations of the absolute spatial-error objective."""

import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "om4-temporal-memory"))
from common import OceanModel, Prepared, highpass, interior_mask, mse
from absolute_relative_loss import objective

DEADLINE = 1792414800


@torch.no_grad()
def damping_summary(optimizer):
    active_count = 0
    zero_count = 0
    below_count = 0
    contribution_sum = 0.
    for group in optimizer.param_groups:
        epsilon, beta2 = group["eps"], group["betas"][1]
        for parameter in group["params"]:
            state = optimizer.state.get(parameter, {})
            if "exp_avg_sq" not in state:
                continue
            step = float(state["step"])
            if step == 0:
                continue
            rms = state["exp_avg_sq"].sqrt() / math.sqrt(1 - beta2**step)
            positive = rms > 0
            active_count += int(positive.sum())
            zero_count += int((~positive).sum())
            below_count += int((positive & (rms < epsilon)).sum())
            contribution_sum += float((epsilon / (rms[positive] + epsilon)).double().sum())
    return {
        "positive_second_moment_parameters": active_count,
        "zero_second_moment_parameters": zero_count,
        "fraction_positive_rms_below_epsilon": below_count / active_count if active_count else None,
        "mean_epsilon_share_of_denominator_positive_only": contribution_sum / active_count if active_count else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-outer-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    args = parser.parse_args()
    assert args.epsilon > 0 and args.total_outer_steps > 1
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    resume = (out / "latest.pt").exists()
    saved = torch.load(out / "latest.pt" if resume else args.source / "latest.pt",
                       map_location="cuda", weights_only=False)
    model.load_state_dict(saved["model"])
    source_history = [json.loads(line) for line in (args.source / "history.jsonl").read_text().splitlines()]
    assert source_history[-1]["step"] == 5000 and (args.source / "COMPLETE").exists()
    if resume:
        optimizer.load_state_dict(saved["optimizer"])
        assert all(group["eps"] == args.epsilon for group in optimizer.param_groups)
        start, elapsed = saved["step"], saved["elapsed"]
        torch.set_rng_state(saved["rng_cpu"].cpu())
        torch.cuda.set_rng_state(saved["rng_cuda"].cpu())
    else:
        assert saved["step"] == 5000
        start, elapsed = 0, 0.
    del saved
    began = time.monotonic()
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGUSR1, stopping)
    signal.signal(signal.SIGTERM, stopping)
    protocol = vars(args) | {
        "mode": "eval; original BatchNorm running statistics stay frozen",
        "optimizer_history": "fresh Adam at initial source; exact history on requeue",
        "supervision": "first target, all 77 channels", "steps": args.total_outer_steps,
        "source_mse": source_history[-1]["eval_mse"],
        "source_highpass_mse": source_history[-1]["highpass_mse"],
        "initial_outer_step": 0, "source_adam_updates": 5000,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "purpose": "Isolate Adam epsilon with fixed checkpoint, spatial objective and LR",
        "schedule": "cosine requested lr to1e-7 over10000 updates",
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
        training_loss = objective(prediction, y, wet, inside, args.weight)
        training_loss.backward()
        grad_squared = sum(p.grad.double().square().sum() for p in model.parameters())
        with torch.no_grad():
            error = prediction - y
            row = {
                "step": step, "seconds": elapsed + time.monotonic() - began,
                "eval_mse": float(loss), "training_objective": float(training_loss),
                "highpass_mse": float(mse(highpass(error, inside), torch.zeros_like(error), inside)),
                "objective_gradient_norm": float(grad_squared.sqrt()),
                "logical_gradient_evaluations": step + step // 100 + 1,
                "per_channel_mse": ((error.square() * wet).sum((-2, -1)) / wet.sum((-2, -1))).mean(0).tolist(),
            }
            if step in (0, 2500, 5000, 7500, args.total_outer_steps):
                np.savez_compressed(out / f"maps-{step}.npz", prediction=prediction.detach().cpu().numpy(),
                                    truth=y.cpu().numpy(), mask=data.mask)
        if step % 1000 == 0:
            row["adam_damping"] = damping_summary(optimizer)
        with (out / "history.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        print({k: v for k, v in row.items() if k != "per_channel_mse"}, flush=True)
        if not resume and step == 0:
            assert abs(row["eval_mse"] - source_history[-1]["eval_mse"]) < 1e-9, "Source MSE reproduction failed"
            assert abs(row["highpass_mse"] - source_history[-1]["highpass_mse"]) < 1e-10, "Source high-pass reproduction failed"

    def closure():
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model(x, b)[0][:, :77], y, wet, inside, args.weight)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        return loss

    measure(start)
    for step in range(start, args.total_outer_steps):
        lr = 1e-7 + (args.lr-1e-7)*.5*(1+math.cos(math.pi*step/(args.total_outer_steps-1)))
        for group in optimizer.param_groups:
            group["lr"] = lr
        closure()
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.total_outer_steps:
            measure(step + 1)
            save(step + 1)
        if stop:
            save(step + 1)
            sys.exit(75)
        if time.time() >= DEADLINE:
            save(step + 1)
            (out / "DEADLINE").write_text("Campaign deadline reached\n")
            return
    (out / "COMPLETE").write_text("Completed spatial-objective Adam continuation\n")


if __name__ == "__main__":
    main()
