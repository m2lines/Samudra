#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Instrument a disposable copy of the qualified runner; never writes production."""

from pathlib import Path

source = Path(__file__).with_name("run_monthly_replay.py").read_text()


def replace(old, new):
    global source
    if source.count(old) != 1:
        raise ValueError(f"Expected one instrumentation anchor: {old[:80]}")
    source = source.replace(old, new)


replace(
    "    def update():\n",
    """    profile_times = {}
    profile_last = [time.monotonic()]
    profile_rng = {}
    profile_cache = {}

    def mark(name):
        torch.cuda.synchronize()
        now = time.monotonic()
        profile_times[name] = profile_times.get(name, 0.0) + now - profile_last[0]
        profile_last[0] = now

    def cached(kind, key, call):
        cache_key = (kind, str(key))
        if step >= 4:
            return profile_cache[cache_key]
        value = call()
        profile_cache[cache_key] = value
        return value

    def update():
""",
)
replace(
    "        started = time.monotonic()\n",
    """        started = time.monotonic()
        profile_times.clear()
        torch.cuda.synchronize()
        profile_last[0] = time.monotonic()
        if source not in profile_rng:
            profile_rng[source] = copy.deepcopy(rng.bit_generator.state)
        rng.bit_generator.state = copy.deepcopy(profile_rng[source])
""",
)
replace(
    "                        sample = data.load(path)\n",
    '                        sample = cached("obs", path, lambda: data.load(path))\n                        mark("observation_load")\n',
)
replace(
    '                batch = reader.sample(window_index, "cuda")\n',
    '                batch = cached("model", window_index, lambda: reader.sample(window_index, "cuda"))\n                mark("model_sample_integrity")\n',
)
replace(
    "                        forcing_cache.load(reader, window_index, data)\n",
    '                        cached("forcing", window_index, lambda: forcing_cache.load(reader, window_index, data))\n',
)
replace(
    "                    if optimization is not None:\n                        from samudra.experiments.monthly_batching import (",
    '                    mark("forcing_load")\n                    if optimization is not None:\n                        from samudra.experiments.monthly_batching import (',
)
# Both sources share these exact anchors with different indentation.
source = source.replace(
    "draw_trace.append(draw_digest(item))",
    'mark("prepare_draws")\n'
    + " " * 32
    + "draw_trace.append(draw_digest(item))\n"
    + " " * 32
    + 'mark("draw_digest")',
    1,
)
# Fix the second occurrence separately.
source = source.replace(
    "                            draw_trace.append(draw_digest(item))\n                        continue",
    '                            mark("prepare_draws")\n                            draw_trace.append(draw_digest(item))\n                            mark("draw_digest")\n                        continue',
)
replace(
    "            staged = runtime.prepared(staged if runtime.rank == 0 else None)\n",
    '            mark("preparation_rest_or_wait")\n            staged = runtime.prepared(staged if runtime.rank == 0 else None)\n            mark("prepared_broadcast")\n',
)
replace(
    "            local_items = staged[2 * runtime.rank : 2 * runtime.rank + 2]\n",
    '            mark("metadata_share")\n            local_items = staged[2 * runtime.rank : 2 * runtime.rank + 2]\n',
)
replace(
    "            loss.backward()\n            summaries = runtime.gather(\n",
    '            mark("forward")\n            loss.backward()\n            mark("backward_allreduce")\n            summaries = runtime.gather(\n',
)
replace(
    "        adapter_norm = (\n",
    '        mark("summary_gather")\n        adapter_norm = (\n',
)
replace(
    "        optimizer.step()\n",
    '        mark("gradient_checks_clip")\n        optimizer.step()\n        mark("optimizer")\n',
)
replace("        step += 1\n", '        mark("ema")\n        step += 1\n')
replace(
    "        return record\n",
    """        mark("record")
        profile = dict(rank=runtime.rank, step=step, source=source,
                       cached_inputs=step > 4, timings=dict(profile_times))
        with (out / f"profile-rank{runtime.rank}.jsonl").open("a") as handle:
            handle.write(json.dumps(profile) + "\\n")
        print(json.dumps(profile), flush=True)
        return record
""",
)
start = source.index("    if a.qualify:\n        # Exercise both sources")
stop = source.index("    import wandb\n", start)
source = (
    source[:start]
    + """    if a.qualify:
        import cProfile
        import pstats
        profiler = cProfile.Profile()
        profiler.enable()
        records = [update() for _ in range(8)]
        profiler.disable()
        profiler.dump_stats(str(out / f"cpu-rank{runtime.rank}.prof"))
        with (out / f"cpu-rank{runtime.rank}.txt").open("w") as handle:
            pstats.Stats(profiler, stream=handle).sort_stats("cumulative").print_stats(80)
        runtime.barrier()
        runtime.write(out / "PROFILE_COMPLETE.json", json.dumps(dict(records=records, diagnostic_only=True)))
        return
"""
    + source[stop:]
)
exec(compile(source, __file__ + ":instrumented", "exec"))
