#!/usr/bin/env python3
"""Per-GPU health on this node: nvidia-smi state plus a bf16 matmul and conv burn-in on every GPU at once.

    srun -N 2 --ntasks-per-node=1 .venv/bin/python scripts/gpu_health.py [--seconds 60]
"""

from __future__ import annotations

import argparse
import socket
import statistics
import subprocess
import time

import torch
import torch.multiprocessing as mp

QUERY = (
    "index,pci.bus_id,temperature.gpu,power.draw,clocks.sm,clocks.max.sm,"
    "pcie.link.width.current,ecc.errors.uncorrected.volatile.total"
)


def nvidia_smi() -> str:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout or result.stderr


def rate(work, seconds: float) -> float:
    """Calls of `work` per second over `seconds`, after a short warmup."""
    for _ in range(3):
        work()
    torch.cuda.synchronize()
    calls, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        work()
        calls += 1
        if calls % 10 == 0:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    return calls / (time.perf_counter() - start)


def burn(index: int, seconds: float, results) -> None:
    torch.cuda.set_device(index)
    a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
    tflops = rate(lambda: a @ b, seconds / 2) * 2 * 8192**3 / 1e12
    # A dilated conv at the training tile size, the model's dominant kernel.
    conv = torch.nn.Conv2d(256, 256, 3, padding=4, dilation=4).cuda().to(torch.bfloat16)
    x = torch.randn(2, 256, 752, 752, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        convs = rate(lambda: conv(x), seconds / 2)
    results[index] = (tflops, convs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=60.0, help="burn-in per GPU, split matmul/conv")
    parser.add_argument("--tolerance", type=float, default=0.10, help="flag below (1 - tol) x node median")
    args = parser.parse_args()

    host = socket.gethostname().split(".")[0]
    print(f"== {host}\n{nvidia_smi()}")
    count = torch.cuda.device_count()
    results = mp.Manager().dict()
    workers = [mp.get_context("spawn").Process(target=burn, args=(i, args.seconds, results)) for i in range(count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    failed = [i for i in range(count) if i not in results]
    tflops = statistics.median(results[i][0] for i in results) if results else 0.0
    convs = statistics.median(results[i][1] for i in results) if results else 0.0
    healthy = []
    for i in range(count):
        if i in failed:
            print(f"  GPU {i}: FAILED (burn-in process died)")
            continue
        matmul, conv = results[i]
        slow = matmul < (1 - args.tolerance) * tflops or conv < (1 - args.tolerance) * convs
        print(f"  GPU {i}: {matmul:6.1f} TFLOP/s bf16 matmul, {conv:6.2f} conv/s  {'SLOW' if slow else 'ok'}")
        if not slow:
            healthy.append(str(i))
    print(f"  node median {tflops:.1f} TFLOP/s, {convs:.2f} conv/s")
    # Visible-device indices; use them in NODE_GPUS to train on the healthy GPUs only.
    print(f"  NODE_GPUS entry: {host}={','.join(healthy)}")


if __name__ == "__main__":
    main()
