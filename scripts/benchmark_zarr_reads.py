# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Benchmark random per-file reads from a local Zarr directory store.

This harness walks a local ``.zarr`` directory, picks random regular files,
and reads them with thread-based concurrency. It reports latency and
throughput summaries for each concurrency point.

Example:
    uv run scripts/benchmark_zarr_reads.py /data/OM4.zarr

    uv run scripts/benchmark_zarr_reads.py /data/OM4.zarr \
      --min-bytes $((8 * 1024 * 1024)) \
      --reads 64 \
      --threads 1 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_THREADS = (1, 2, 4, 8, 16, 32)
METADATA_FILE_NAMES = frozenset(
    {
        ".zarray",
        ".zattrs",
        ".zgroup",
        ".zmetadata",
        "zarr.json",
    }
)
MIB = 1024 * 1024


@dataclass(frozen=True)
class CandidateFile:
    path: Path
    size_bytes: int


@dataclass(frozen=True)
class ReadMeasurement:
    size_bytes: int
    open_s: float
    read_s: float
    close_s: float
    end_to_end_s: float


@dataclass(frozen=True)
class Stats:
    minimum: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    maximum: float


@dataclass(frozen=True)
class SweepPointSummary:
    threads: int
    reads: int
    sample_mode: str
    total_bytes: int
    wall_s: float
    aggregate_mib_per_s: float
    file_size_mib: Stats
    open_ms: Stats
    read_ms: Stats
    close_ms: Stats
    end_to_end_ms: Stats
    read_mib_per_s: Stats
    end_to_end_mib_per_s: Stats


@dataclass(frozen=True)
class HarnessSummary:
    zarr_path: str
    candidate_count: int
    candidate_size_mib: Stats
    seed: int
    summaries: list[SweepPointSummary]


def percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if len(sorted_values) == 1:
        return sorted_values[0]

    index = (len(sorted_values) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_values[lower]

    fraction = index - lower
    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def summarize(values: Iterable[float]) -> Stats:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("cannot summarize an empty sequence")

    return Stats(
        minimum=ordered[0],
        mean=statistics.fmean(ordered),
        p50=percentile(ordered, 0.50),
        p90=percentile(ordered, 0.90),
        p95=percentile(ordered, 0.95),
        p99=percentile(ordered, 0.99),
        maximum=ordered[-1],
    )


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def format_stats(stats: Stats, digits: int = 2) -> str:
    return (
        f"min={format_float(stats.minimum, digits)} "
        f"mean={format_float(stats.mean, digits)} "
        f"p50={format_float(stats.p50, digits)} "
        f"p95={format_float(stats.p95, digits)} "
        f"p99={format_float(stats.p99, digits)} "
        f"max={format_float(stats.maximum, digits)}"
    )


def discover_candidate_files(
    zarr_path: Path,
    *,
    include_metadata: bool,
    min_bytes: int,
    max_bytes: int | None,
) -> list[CandidateFile]:
    candidates: list[CandidateFile] = []
    for dirpath, _, filenames in os.walk(zarr_path):
        for filename in filenames:
            if not include_metadata and filename in METADATA_FILE_NAMES:
                continue

            path = Path(dirpath) / filename
            size_bytes = path.stat().st_size
            if size_bytes < min_bytes:
                continue
            if max_bytes is not None and size_bytes > max_bytes:
                continue
            candidates.append(CandidateFile(path=path, size_bytes=size_bytes))
    return candidates


def sample_workload(
    candidates: Sequence[CandidateFile],
    *,
    reads: int,
    rng: random.Random,
) -> tuple[list[CandidateFile], str]:
    if reads <= len(candidates):
        return rng.sample(list(candidates), reads), "without_replacement"

    workload = list(candidates)
    rng.shuffle(workload)
    workload.extend(rng.choice(candidates) for _ in range(reads - len(candidates)))
    return workload, "with_replacement"


def maybe_advise_random(fd: int) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_RANDOM"):
        return
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    except OSError:
        # This is only a best-effort hint to the kernel.
        return


def read_file(candidate: CandidateFile) -> ReadMeasurement:
    start = time.perf_counter()
    fd = os.open(candidate.path, os.O_RDONLY)
    after_open = time.perf_counter()
    try:
        maybe_advise_random(fd)
        payload = os.pread(fd, candidate.size_bytes, 0)
        after_read = time.perf_counter()
    finally:
        os.close(fd)
    after_close = time.perf_counter()

    bytes_read = len(payload)
    if bytes_read != candidate.size_bytes:
        raise RuntimeError(
            f"short read for {candidate.path}: expected {candidate.size_bytes} "
            f"bytes, got {bytes_read}"
        )

    return ReadMeasurement(
        size_bytes=bytes_read,
        open_s=after_open - start,
        read_s=after_read - after_open,
        close_s=after_close - after_read,
        end_to_end_s=after_close - start,
    )


def build_summary(
    *,
    threads: int,
    workload: Sequence[CandidateFile],
    sample_mode: str,
    measurements: Sequence[ReadMeasurement],
    wall_s: float,
) -> SweepPointSummary:
    total_bytes = sum(item.size_bytes for item in measurements)
    read_mib_per_s = [
        (item.size_bytes / MIB) / item.read_s for item in measurements if item.read_s > 0
    ]
    end_to_end_mib_per_s = [
        (item.size_bytes / MIB) / item.end_to_end_s
        for item in measurements
        if item.end_to_end_s > 0
    ]

    return SweepPointSummary(
        threads=threads,
        reads=len(measurements),
        sample_mode=sample_mode,
        total_bytes=total_bytes,
        wall_s=wall_s,
        aggregate_mib_per_s=(total_bytes / MIB) / wall_s,
        file_size_mib=summarize(item.size_bytes / MIB for item in workload),
        open_ms=summarize(item.open_s * 1e3 for item in measurements),
        read_ms=summarize(item.read_s * 1e3 for item in measurements),
        close_ms=summarize(item.close_s * 1e3 for item in measurements),
        end_to_end_ms=summarize(item.end_to_end_s * 1e3 for item in measurements),
        read_mib_per_s=summarize(read_mib_per_s),
        end_to_end_mib_per_s=summarize(end_to_end_mib_per_s),
    )


def benchmark_concurrency(
    candidates: Sequence[CandidateFile],
    *,
    threads: int,
    reads: int,
    seed: int,
) -> SweepPointSummary:
    rng = random.Random(seed)
    workload, sample_mode = sample_workload(candidates, reads=reads, rng=rng)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="zarr-read") as pool:
        measurements = list(pool.map(read_file, workload))
    wall_s = time.perf_counter() - started

    return build_summary(
        threads=threads,
        workload=workload,
        sample_mode=sample_mode,
        measurements=measurements,
        wall_s=wall_s,
    )


def print_candidate_summary(candidates: Sequence[CandidateFile]) -> None:
    size_stats = summarize(item.size_bytes / MIB for item in candidates)
    print(f"candidate files: {len(candidates)}")
    print(f"candidate size MiB: {format_stats(size_stats)}")


def print_sweep_summary(summary: SweepPointSummary) -> None:
    total_gib = summary.total_bytes / (1024**3)
    print(
        f"threads={summary.threads:>2} reads={summary.reads:>4} "
        f"mode={summary.sample_mode:<20} data={total_gib:>6.2f} GiB "
        f"wall={summary.wall_s:>7.2f} s "
        f"agg={summary.aggregate_mib_per_s:>8.2f} MiB/s"
    )
    print(f"  file size MiB:       {format_stats(summary.file_size_mib)}")
    print(f"  open latency ms:     {format_stats(summary.open_ms)}")
    print(f"  read latency ms:     {format_stats(summary.read_ms)}")
    print(f"  close latency ms:    {format_stats(summary.close_ms)}")
    print(f"  end-to-end ms:       {format_stats(summary.end_to_end_ms)}")
    print(f"  read MiB/s:          {format_stats(summary.read_mib_per_s)}")
    print(
        f"  end-to-end MiB/s:    "
        f"{format_stats(summary.end_to_end_mib_per_s)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zarr_path", type=Path, help="Path to a local .zarr directory")
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=list(DEFAULT_THREADS),
        help="Concurrency points to measure",
    )
    reads_group = parser.add_mutually_exclusive_group()
    reads_group.add_argument(
        "--reads",
        type=int,
        default=64,
        help="Total random file reads to issue at each concurrency point",
    )
    reads_group.add_argument(
        "--reads-per-thread",
        type=int,
        default=None,
        help="Random file reads to issue per thread at each concurrency point",
    )
    parser.add_argument(
        "--min-bytes",
        type=int,
        default=1,
        help="Skip candidate files smaller than this many bytes",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Skip candidate files larger than this many bytes",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include Zarr metadata files such as .zarray and zarr.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to choose files for each sweep point",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the structured summary as JSON",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    zarr_path = args.zarr_path.expanduser().resolve()
    if not zarr_path.is_dir():
        raise NotADirectoryError(f"expected a local Zarr directory, got {zarr_path}")

    threads = tuple(args.threads)
    if any(thread <= 0 for thread in threads):
        raise ValueError(f"all thread counts must be positive: {threads}")

    if args.reads is not None and args.reads <= 0:
        raise ValueError("--reads must be positive")
    if args.reads_per_thread is not None and args.reads_per_thread <= 0:
        raise ValueError("--reads-per-thread must be positive")
    if args.min_bytes < 0:
        raise ValueError("--min-bytes must be non-negative")
    if args.max_bytes is not None and args.max_bytes <= 0:
        raise ValueError("--max-bytes must be positive")

    print(f"zarr path: {zarr_path}")
    print(f"seed: {args.seed}")
    candidates = discover_candidate_files(
        zarr_path,
        include_metadata=args.include_metadata,
        min_bytes=args.min_bytes,
        max_bytes=args.max_bytes,
    )
    if not candidates:
        raise RuntimeError(
            "no candidate files matched the filter; try lowering --min-bytes "
            "or using --include-metadata"
        )
    print_candidate_summary(candidates)

    summaries: list[SweepPointSummary] = []
    for index, thread_count in enumerate(threads):
        reads = (
            args.reads
            if args.reads_per_thread is None
            else args.reads_per_thread * thread_count
        )
        summary = benchmark_concurrency(
            candidates,
            threads=thread_count,
            reads=reads,
            seed=args.seed + index,
        )
        summaries.append(summary)
        print_sweep_summary(summary)

    harness_summary = HarnessSummary(
        zarr_path=str(zarr_path),
        candidate_count=len(candidates),
        candidate_size_mib=summarize(item.size_bytes / MIB for item in candidates),
        seed=args.seed,
        summaries=summaries,
    )
    if args.json_out is not None:
        json_out = args.json_out.expanduser()
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(asdict(harness_summary), indent=2) + "\n")
        print(f"wrote JSON summary to {json_out.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
