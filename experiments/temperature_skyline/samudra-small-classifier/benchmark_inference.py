# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numcodecs import blosc  # type: ignore[import-untyped]

from samudra.skyline.multitask import ComparisonSource
from samudra.skyline.run import to_device
from samudra.skyline.soft_prior_maps import load_selection

ROOT = Path("/scratch/jr7309/skyline-inference-timing-20261111")
TRAIN = Path("/scratch/jr7309/skyline-soft-priors-25k-20261110")


def main():
    torch.set_num_threads(4)
    blosc.set_nthreads(1)
    device = torch.device("cuda")
    start = time.perf_counter()
    checkpoint = TRAIN / "all-fields-both-25k/latest.pt"
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    cfg = state["config"]
    source = ComparisonSource(
        cfg["source"], 4, cfg.get("reference_root"), arm="all_fields"
    )
    model, vocab, cfg, receipt = load_selection(
        dict(checkpoint=str(checkpoint), step=25000, from_latest=True), source, device
    )
    setup = time.perf_counter() - start
    start = time.perf_counter()
    frame = source.read(4273)
    frame_seconds = time.perf_counter() - start
    counts = source.wet.sum((1, 2)).astype(int)
    rng = np.random.default_rng(20261111)
    points = []
    for depth in range(19):
        wet = np.flatnonzero(source.wet[depth])
        chosen = rng.choice(wet, 1024, replace=False)
        points.append(
            np.column_stack(np.unravel_index(chosen, source.wet[depth].shape))
        )

    def batch_at(depth, pp):
        return to_device(
            source.sample(
                frame, rng, len(pp), cfg["crop"], np.full(len(pp), depth), pp
            ),
            device,
        )

    def predict(batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model({k: v for k, v in batch.items() if k != "target"})
        ids = logits.argmax(-1)
        return vocab.centers[batch["query_channel"], ids]

    result = dict(
        gpu=torch.cuda.get_device_name(),
        torch=torch.__version__,
        setup_seconds=setup,
        frame_read_seconds=frame_seconds,
        wet_counts=counts.tolist(),
        depth_point_queries=int(counts.sum()),
        scalar_wet_values=int(4 * counts.sum() + counts[0]),
        grid=[source.height, source.width],
        scalar_dense_float32_bytes=77 * source.height * source.width * 4,
        source_commit="ea592e8a71ebf9b50285ef835d641aae1de4d3c9",  # pragma: allowlist secret
        checkpoint_sha256=receipt["checkpoint_sha256"],
        checkpoint_step=25000,
        weights="EMA",
        batches=[],
    )
    with torch.inference_mode():
        for size in (32, 64, 128):
            torch.cuda.reset_peak_memory_stats()
            batch = batch_at(0, points[0][:size])
            for _ in range(5):
                predict(batch).cpu()
            torch.cuda.synchronize()
            gpu_repeats = []
            for _ in range(3):
                before = torch.cuda.Event(enable_timing=True)
                after = torch.cuda.Event(enable_timing=True)
                before.record()
                for _ in range(64):
                    predict(batch)
                after.record()
                after.synchronize()
                gpu_repeats.append(before.elapsed_time(after) / 1000)
            rows: list[dict[str, Any]] = []
            for depth in range(19):
                elapsed = []
                for repeat in range(3):
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    for offset in range(0, 1024, size):
                        pred = (
                            predict(
                                batch_at(depth, points[depth][offset : offset + size])
                            )
                            .cpu()
                            .numpy()
                        )
                        if not np.isfinite(pred).all():
                            raise RuntimeError("Nonfinite predictions")
                    elapsed.append(time.perf_counter() - begin)
                rows.append(dict(depth=depth, queries_per_repeat=1024, seconds=elapsed))
            total = sum(
                int(counts[row["depth"]]) * np.median(row["seconds"]) / 1024
                for row in rows
            )
            item = dict(
                batch_size=size,
                gpu_only_queries_per_repeat=64 * size,
                gpu_only_seconds=gpu_repeats,
                depths=rows,
                extrapolated_pipeline_seconds=float(total),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            )
            result["batches"].append(item)
            (ROOT / "timing.json").write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    dict(batch_size=size, extrapolated_pipeline_seconds=float(total))
                ),
                flush=True,
            )
    source.close()
    print("COMPLETE", flush=True)


if __name__ == "__main__":
    main()
