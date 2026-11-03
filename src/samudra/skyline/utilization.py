# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""GPU telemetry over real wall time, including validation and checkpoint I/O."""

import json
import math
import subprocess
import threading
import time
from collections import deque


class UtilizationWindow:
    def __init__(self, minimum_percent=60, warmup_seconds=120, window_seconds=300):
        if not 0 < minimum_percent <= 100 or warmup_seconds < 0 or window_seconds <= 0:
            raise ValueError("Invalid utilization guard settings")
        self.minimum = minimum_percent
        self.warmup = warmup_seconds
        self.window = window_seconds
        self.samples: deque[tuple[float, float]] = deque()

    def add(self, elapsed, utilization):
        if not math.isfinite(utilization) or not 0 <= utilization <= 100:
            raise ValueError("Invalid GPU utilization sample")
        self.samples.append((elapsed, utilization))
        while self.samples and self.samples[0][0] < elapsed - self.window:
            self.samples.popleft()
        selected = [(t, u) for t, u in self.samples if t >= self.warmup]
        if len(selected) < 2 or selected[-1][0] - selected[0][0] < self.window * 0.95:
            return None
        # Weight each sample by the time to the next observation. A tiny polling
        # delay should not give one measurement disproportionate influence.
        duration = selected[-1][0] - selected[0][0]
        return (
            sum(u * (next_t - t) for (t, u), (next_t, _) in zip(selected, selected[1:]))
            / duration
        )


class GpuMonitor:
    def __init__(self, output, uuid, restart_count, config):
        config = dict(config)
        self.interval = config.pop("sample_seconds", 5)
        if not 1 <= self.interval <= 10:
            raise ValueError("GPU sampling interval must be between 1 and 10 seconds")
        self.window = UtilizationWindow(**config)
        self.path = output / f"gpu-utilization-restart-{restart_count}.jsonl"
        # CUDA/PyTorch can return the bare UUID; nvidia-smi requires GPU-.
        self.uuid = str(uuid)
        if not self.uuid.startswith("GPU-"):
            self.uuid = f"GPU-{self.uuid}"
        self.stop = threading.Event()
        self.reason: str | None = None
        self.mean: float | None = None
        self.thread = threading.Thread(
            target=self.run, name="gpu-utilization", daemon=True
        )

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop.set()
        self.thread.join(timeout=15)
        if self.thread.is_alive():
            raise RuntimeError("GPU telemetry failed to stop")
        if exc_type is None and self.reason:
            (self.path.parent / "COMPLETE.json").unlink(missing_ok=True)
            raise RuntimeError(self.reason)

    def run(self):
        begin = time.monotonic()
        try:
            with self.path.open("a", buffering=1) as f:
                while not self.stop.is_set():
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "-i",
                            self.uuid,
                            "--query-gpu=utilization.gpu,memory.used,power.draw",
                            "--format=csv,noheader,nounits",
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )
                    utilization, memory, power = map(
                        float, result.stdout.strip().split(",")
                    )
                    elapsed = time.monotonic() - begin
                    self.mean = self.window.add(elapsed, utilization)
                    if self.mean is not None and self.mean < self.window.minimum:
                        self.reason = (
                            f"GPU utilization {self.mean:.1f}% over the last "
                            f"{self.window.window}s is below {self.window.minimum}%; "
                            "stop and diagnose before continuing"
                        )
                    f.write(
                        json.dumps(
                            {
                                "unix_time": time.time(),
                                "elapsed_seconds": elapsed,
                                "gpu_uuid": self.uuid,
                                "utilization_percent": utilization,
                                "memory_mib": memory,
                                "power_w": power,
                                "rolling_utilization_percent": self.mean,
                                "stop_reason": self.reason,
                            }
                        )
                        + "\n"
                    )
                    self.stop.wait(self.interval)
        except Exception as exc:
            # A missing monitor must not silently disable the utilization guard.
            self.reason = f"GPU telemetry failed: {type(exc).__name__}: {exc}"
