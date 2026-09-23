#!/usr/bin/env python3
"""Measure how long one face-frame of truth takes to read, two ways.

Full-face replay needs every tile of a face at one timestamp, every step. The
question this answers is whether that read can keep up with the GPUs, and how
much the tile geometry costs.

A tile window is ``[720c - 16, 720c + 736)``: it starts 16 cells *before* chunk
boundary ``c`` and ends 16 cells *after* boundary ``c+1``, so it spans THREE
chunk columns. Read one window at a time and an interior tile costs 9
chunk decodes; the clamped end tiles span two columns, so the face actually
costs 256 decodes to deliver 36 chunks of distinct data.

    naive    one zarr read per tile window; zarr fetches and decodes all 9
             chunks the window touches, then crops.
    chunked  read each chunk the rank's tiles touch exactly once, and scatter
             its overlapping sub-rectangle into every tile buffer that wants
             it. A rank owning a contiguous 3x3 block of tiles touches 16
             chunks to fill 9 tiles.

Both are measured with the ranks running CONCURRENTLY, as they do in training,
because the second rank to want a chunk may find it in page cache and that
sharing is part of the design. Page cache is dropped before each frame, and
every frame uses a timestamp no earlier phase has touched, so "cold" means
cold.

Usage (must be a compute node; a login node has too few cores to be meaningful):

    srun -p mit_normal_gpu --gres=gpu:h200:1 -c 60 --mem=256G -t 30 \
        .venv/bin/python scripts/_bench/face_read.py --ranks 4 --threads 8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numcodecs.blosc
import numpy as np
import zarr

from ocean_emulators.tiling import face_tile_windows

CHUNK_GRID = 6

STORE = (
    "/orcd/data/abodner/002/cody/LLC_patch/face_1/"
    "LLC4320_face1_i0-4320_j0-4320.zarr"
)
CHUNK = 720


# --------------------------------------------------------------------------
# Work partitioning
# --------------------------------------------------------------------------
def rank_tiles(windows: list[tuple[int, int, int, int, int]], ranks: int) -> list[list[int]]:
    """Split the tile grid into ``ranks`` contiguous blocks, one per rank.

    Contiguity is the point: a rank that owns a 3x3 block of tiles touches 16
    chunks, where the same 9 tiles scattered over the face would touch up to
    81. Chunk locality is why tiles are assigned this way rather than to
    balance land.
    """
    side = int(round(len(windows) ** 0.5))
    if side * side != len(windows):
        raise ValueError(f"{len(windows)} tiles is not a square grid")
    if len(windows) % ranks:
        raise ValueError(f"{len(windows)} tiles do not divide over {ranks} ranks")

    # Choose a block shape (rows x cols of tiles) whose area is the per-rank
    # count and which tiles the grid.
    per_rank = len(windows) // ranks
    best = None
    for rows in range(1, side + 1):
        if per_rank % rows or side % rows:
            continue
        cols = per_rank // rows
        if cols > side or side % cols:
            continue
        # Prefer the squarest block: it has the smallest chunk footprint.
        score = abs(rows - cols)
        if best is None or score < best[0]:
            best = (score, rows, cols)
    if best is None:
        raise ValueError(f"cannot tile {side}x{side} into {ranks} equal blocks")
    _, rows, cols = best

    groups: list[list[int]] = []
    for block_j in range(side // rows):
        for block_i in range(side // cols):
            groups.append(
                [
                    (block_j * rows + dj) * side + (block_i * cols + di)
                    for dj in range(rows)
                    for di in range(cols)
                ]
            )
    return groups


def chunks_for(window: tuple[int, int, int, int, int]) -> list[tuple[int, int]]:
    _, i0, i1, j0, j1 = window
    return [
        (cj, ci)
        for cj in range((j0) // CHUNK, (j1 - 1) // CHUNK + 1)
        for ci in range((i0) // CHUNK, (i1 - 1) // CHUNK + 1)
    ]


# --------------------------------------------------------------------------
# The two readers
# --------------------------------------------------------------------------
_ARRAY = None
_STATE: dict = {}


def _init(store: str, state: dict) -> None:
    global _ARRAY, _STATE
    # Each rank already runs `threads` read threads and there are `ranks`
    # processes, so letting blosc spawn its own pool on top oversubscribes the
    # node and makes the measurement about scheduling rather than I/O.
    numcodecs.blosc.set_nthreads(state["blosc_threads"])
    _ARRAY = zarr.open(store, mode="r")["prognostic"]
    _STATE = state


def read_naive(tiles: list[tuple[int, int, int, int, int]], step: int, dtype) -> tuple[int, int]:
    """One zarr read per tile window. Returns (bytes delivered, chunk decodes)."""
    delivered = 0
    decodes = 0
    for window in tiles:
        _, i0, i1, j0, j1 = window
        out = np.asarray(_ARRAY[step, :, j0:j1, i0:i1], dtype=dtype)
        delivered += out.nbytes
        decodes += len(chunks_for(window))
    return delivered, decodes


def read_chunked(
    tiles: list[tuple[int, int, int, int, int]], step: int, dtype, threads: int
) -> tuple[int, int]:
    """Decode each needed chunk once, scatter it into every tile that wants it.

    The face is never materialized: a chunk is dropped as soon as it has been
    scattered, so peak memory is the tile buffers plus one chunk per thread.
    """
    channels = _ARRAY.shape[1]
    buffers = [
        np.empty((channels, j1 - j0, i1 - i0), dtype=dtype)
        for _, i0, i1, j0, j1 in tiles
    ]

    # chunk -> the tiles that overlap it, with the copy each one needs.
    plan: dict[tuple[int, int], list] = {}
    for index, window in enumerate(tiles):
        _, i0, i1, j0, j1 = window
        for cj, ci in chunks_for(window):
            cj0, ci0 = cj * CHUNK, ci * CHUNK
            # Intersection, in absolute face coordinates.
            aj0, aj1 = max(j0, cj0), min(j1, cj0 + CHUNK)
            ai0, ai1 = max(i0, ci0), min(i1, ci0 + CHUNK)
            plan.setdefault((cj, ci), []).append(
                (
                    index,
                    (slice(aj0 - j0, aj1 - j0), slice(ai0 - i0, ai1 - i0)),
                    (slice(aj0 - cj0, aj1 - cj0), slice(ai0 - ci0, ai1 - ci0)),
                )
            )

    def fill(item) -> None:
        (cj, ci), targets = item
        cj0, ci0 = cj * CHUNK, ci * CHUNK
        chunk = _ARRAY[step, :, cj0 : cj0 + CHUNK, ci0 : ci0 + CHUNK]
        for index, dst, src in targets:
            buffers[index][:, dst[0], dst[1]] = chunk[:, src[0], src[1]]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(fill, plan.items()))

    return sum(b.nbytes for b in buffers), len(plan)


def _work(rank: int) -> tuple[float, int, int]:
    tiles = [_STATE["windows"][i] for i in _STATE["groups"][rank]]
    started = time.perf_counter()
    if _STATE["mode"] == "naive":
        delivered, decodes = read_naive(tiles, _STATE["step"], _STATE["dtype"])
    else:
        delivered, decodes = read_chunked(
            tiles, _STATE["step"], _STATE["dtype"], _STATE["threads"]
        )
    return time.perf_counter() - started, delivered, decodes


# --------------------------------------------------------------------------
# Page cache
# --------------------------------------------------------------------------
def evict(step: int) -> None:
    """Drop this timestamp's chunks from page cache so a cold read is cold."""
    root = os.path.join(STORE, "prognostic")
    paths = [
        os.path.join(root, f"{step}.0.{cj}.{ci}")
        for cj in range(CHUNK_GRID)
        for ci in range(CHUNK_GRID)
    ]

    def drop(path: str) -> None:
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass

    with ThreadPoolExecutor(max_workers=36) as pool:
        list(pool.map(drop, paths))


# --------------------------------------------------------------------------
def verify(windows, dtype) -> None:
    """A fast reader that returns the wrong bytes is worthless.

    Checks a face corner, an interior tile and the far corner, because those
    are the three overlap regimes the clamped origins produce.
    """
    array = zarr.open(STORE, mode="r")["prognostic"]
    step = 3
    for label, window in (("corner", windows[0]), ("interior", windows[14]), ("far", windows[35])):
        _, i0, i1, j0, j1 = window
        want = np.asarray(array[step, :, j0:j1, i0:i1], dtype=dtype)
        got = np.empty_like(want)
        for cj, ci in chunks_for(window):
            cj0, ci0 = cj * CHUNK, ci * CHUNK
            aj0, aj1 = max(j0, cj0), min(j1, cj0 + CHUNK)
            ai0, ai1 = max(i0, ci0), min(i1, ci0 + CHUNK)
            chunk = array[step, :, cj0 : cj0 + CHUNK, ci0 : ci0 + CHUNK]
            got[:, aj0 - j0 : aj1 - j0, ai0 - i0 : ai1 - i0] = chunk[
                :, aj0 - cj0 : aj1 - cj0, ai0 - ci0 : ai1 - ci0
            ]
        assert np.array_equal(got, want, equal_nan=True), (
            f"chunk scatter != window read for the {label} tile {window}"
        )
        print(f"verify: {label} tile {window[1:]} reproduced exactly", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8, help="read threads per rank")
    parser.add_argument("--blosc-threads", type=int, default=1)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--first-step", type=int, default=6000)
    parser.add_argument("--modes", default="chunked,naive")
    parser.add_argument("--no-evict", action="store_true")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    dtype = np.dtype(args.dtype)
    windows = face_tile_windows(1, overlap=args.overlap)
    groups = rank_tiles(windows, args.ranks)
    modes = args.modes.split(",")

    print(f"store   {STORE}")
    print(f"tiles   {len(windows)} x {windows[0][2] - windows[0][1]}^2, overlap {args.overlap}")
    print(f"ranks   {args.ranks} x {len(groups[0])} tiles, {args.threads} read threads each")
    print(f"dtype   {dtype}  ({np.prod([205, 752, 752]) * dtype.itemsize / 2**30:.2f} GiB per tile-set of 1)")
    footprint = {len(set(sum((chunks_for(windows[i]) for i in g), []))) for g in groups}
    print(f"chunks  {sorted(footprint)} per rank (36 distinct in the face)")
    print()

    if not args.skip_verify:
        verify(windows, dtype)
        print()

    results: dict[str, list[float]] = {mode: [] for mode in modes}
    step = args.first_step
    for rep in range(args.reps):
        for mode in modes:
            # A fresh timestamp every measurement: nothing can be warm from an
            # earlier phase, so modes are compared on equal terms.
            if not args.no_evict:
                evict(step)
            state = {
                "windows": windows,
                "groups": groups,
                "mode": mode,
                "step": step,
                "dtype": dtype,
                "threads": args.threads,
                "blosc_threads": args.blosc_threads,
            }
            started = time.perf_counter()
            with Pool(args.ranks, initializer=_init, initargs=(STORE, state)) as pool:
                per_rank = pool.map(_work, range(args.ranks))
            wall = time.perf_counter() - started

            delivered = sum(r[1] for r in per_rank) / 2**30
            decodes = sum(r[2] for r in per_rank)
            slowest = max(r[0] for r in per_rank)
            results[mode].append(slowest)
            print(
                f"rep {rep}  {mode:8s} step={step:5d}  "
                f"wall {wall:6.2f}s  slowest rank {slowest:6.2f}s  "
                f"{decodes:4d} chunk decodes  {delivered:5.2f} GiB delivered  "
                f"{delivered / slowest:5.2f} GiB/s",
                flush=True,
            )
            step += 1

    print()
    summary = {}
    for mode in modes:
        times = results[mode]
        summary[mode] = {
            "median_s": statistics.median(times),
            "min_s": min(times),
            "max_s": max(times),
        }
        print(
            f"{mode:8s} median {statistics.median(times):6.2f}s  "
            f"range [{min(times):.2f}, {max(times):.2f}]"
        )
    if len(modes) == 2 and all(results[m] for m in modes):
        a, b = (statistics.median(results[m]) for m in modes)
        print(f"\nspeedup {modes[1]} -> {modes[0]}: {b / a:.2f}x")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"args": vars(args) | {"dtype": args.dtype}, "summary": summary}, handle, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
