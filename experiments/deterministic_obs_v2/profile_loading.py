# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Bounded CPU profile of a complete packed-OM4 update, repeated with the same plan."""

import argparse
import cProfile
import gc
import io
import json
import pstats
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from samudra.observations import model_data
from samudra.observations.config import ObservationTrainConfig
from samudra.observations.pipeline import UpdateLoader
from samudra.observations.train import build_sources


def profile(config, output):
    cfg = ObservationTrainConfig.from_yaml(config)
    sources = build_sources(cfg, torch.device("cpu"))
    reader = sources.models["om4"]
    totals: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])

    def timed(name, function):
        def call(*args, **kwargs):
            wall, cpu = time.perf_counter(), time.process_time()
            result = function(*args, **kwargs)
            totals[name][0] += time.perf_counter() - wall
            totals[name][1] += time.process_time() - cpu
            totals[name][2] += 1
            return result

        return call

    assert reader.planes is not None and reader.era5 is not None
    setattr(
        reader.planes,
        "read_into",
        timed("rust_physical_planes", reader.planes.read_into),
    )
    setattr(reader.era5, "load", timed("era5_forcing", reader.era5.load))
    setattr(reader, "_read", timed("model_read_inclusive", reader._read))
    model_data.calendar_context = timed("calendar_context", model_data.calendar_context)
    setattr(sources.obs, "load", timed("observation_coverage", sources.obs.load))
    rng = np.random.default_rng(1729)
    plan = []
    for index in range(8):
        origin = sources.fresh_origin("om4", rng)
        month = index % 2 if sources.layout("om4", origin, index % 2) is not None else 0
        plan.append(
            {
                "origin": origin,
                "month": month,
                "coverage": sources.fresh_origin("obs", rng),
                "cached": None,
                "seed": int(rng.integers(2**31)),
            }
        )
    loader = UpdateLoader(sources, "cpu", enabled=False, to_device=False)
    report: dict[str, Any] = {"config": str(config), "plan": plan, "passes": []}
    profiler = cProfile.Profile()
    for repeat in range(2):
        totals.clear()
        started, cpu = time.perf_counter(), time.process_time()
        profiler.enable()
        batch = loader.read("om4", plan)
        profiler.disable()
        row = {
            "repeat": repeat,
            "wall_seconds": time.perf_counter() - started,
            "cpu_seconds": time.process_time() - cpu,
            "components_wall_cpu_calls": dict(totals),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
        }
        report["passes"].append(row)
        print(json.dumps(row), flush=True)
        del batch
        gc.collect()
    loader.close()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(40)
    report["profile"] = stream.getvalue()
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["profile"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    profile(args.config, args.output)
