# Face-1 cache recut plan

> Superseded 2026-10-07 by `aligned-rank-blocks-plan.md`; the pre-cut layout was not built.

## Decision

Build **36 complete, uniformly shaped 752 x 752 tile stores**, one store per
training tile. Each time slice of each store will contain:

- one prognostic chunk with shape `(1, 205, 752, 752)`; and
- one boundary chunk with shape `(1, 4, 752, 752)`.

Do not make edge and corner tiles depend on neighbouring stores, and do not
make the trainer assemble their halos. The existing `face_tile_windows(1)`
geometry already solves the boundary problem without padding or new exchange
logic: it clamps every window inside the face while keeping every tile 752 x
752.

The six origins on either axis are:

```text
0, 704, 1424, 2144, 2864, 3568
```

Their Cartesian product is the 36-tile face. Interior neighbours overlap by
32 cells. The first and last seams overlap by 48 cells because the exterior
windows are shifted inward to remain 752 cells wide. The existing blender
derives the overlap width per side and already handles both widths.

This is the simplest layout for both intended consumers:

- synchronous face blending can assign six complete tile stores to each of six
  ranks; and
- rank-local blending can draw a spatial block and read every member without
  asking another rank for pixels.

## Why not special-case edges and corners?

The uniform layout duplicates only 9.09% of the face's pixels:

```text
36 * 752 * 752 / (4320 * 4320) = 1.090864...
```

Avoiding that duplication would require the trainer to reconstruct some tiles
from multiple stores. That would bring back multiple object reads per tile,
new host-side stitching buffers, mask/grid-field stitching, and a requirement
that the needed neighbours stay on one rank. Dynamic spatial sampling could
put a needed neighbour on another rank, forcing either a special assignment or
another cross-rank exchange. The storage saving is too small to justify making
the hot training path and both blending topologies more fragile.

Uniform 752 x 752 stores also make the invariant obvious: one requested tile
is one spatial chunk. More precisely, a replay seed reads one prognostic chunk;
a normal replay transition reads one target prognostic chunk and one current
boundary chunk. Prognostic and boundary are separate arrays and different
times, so the transition is two Zarr objects, not literally one object total.
The 205-channel prognostic object dominates the cost.

## Expected I/O change

The current packed face has 720 x 720 spatial chunks. An isolated interior
752 x 752 request intersects nine source chunks per array. The group reader
removes most duplicate decoding, but it cannot make a misaligned 752 x 752
window into one object:

| Local tile block | Packed face, chunks per array | Pre-cut, chunks per array | Reduction in decoded pixels |
| --- | ---: | ---: | ---: |
| 2 x 3 (six-rank synchronous) | 12 | 6 | about 45% |
| 3 x 3 (rank-local block) | 16 | 9 | about 39% |

The pre-cut layout also removes the extra object requests and all per-frame
scatter copies. It does not remove DDP synchronization, and it does not make
seven independent rank-local timestamps into one shared stream. It should
materially reduce the rank-local I/O tail, but it is not by itself a guarantee
that rank-local training will meet the 7 s target. Synchronous face training is
the first acceptance target.

## Source and destination

Use the complete normalized disk copy as the sole source:

```text
/orcd/data/abodner/002/cody/LLC_patch/face_1/
  LLC4320_face1_i0-4320_j0-4320.zarr
```

It has the same `(10311, 205, 4320, 4320)` shape and
`(1, 205, 720, 720)` chunks as the seedfund scratch mirror. Reading disk while
writing the new cache to scratch separates the input and output traffic. The
source is opened read-only and is never modified.

Write the recut stores to:

```text
/orcd/scratch/orcd/014/codycruz/llc_emulator/LLC_patch/
  752-36-tile-face1/
```

The source already has all 205 required prognostic channels and the four
boundary channels `oceTAUX`, `oceTAUY`, `oceQnet`, and `oceFWflx`. The recut
must not drop `W` and must not append another `oceFWflx`.

The logical uncompressed time-varying payload grows from about 80.4 TB to 87.7
TB because of the overlaps. A quick representative-chunk check found LZ4 level
5 about 15-16% larger than the source's Zstd level 3, while encoding roughly
six times faster and decoding roughly two to three times faster. For the urgent
scratch copy, use LZ4 level 5: it minimizes build CPU time and should make
training decompression cheaper. Allow roughly 25-30% more physical space than
the present source after combining the halo duplication and codec difference.

## Reuse the existing recutter

Adapt, rather than replace:

- `scripts/recut_llc_patch_caches.py`
- `JOBS/other/recut_llc_patch_caches.sh`

The Python recutter already has the important algorithm: for one timestamp it
reads each source cache once, scatters by absolute LLC coordinates, and writes
independent destination chunks. With the full-face store as its only source,
all 36 source chunks are decoded once per array and used to construct all 36
destinations. Parallelize by **time**, not by output tile; spatial jobs would
reread the same source chunks for neighbouring tiles.

Required recutter changes:

1. Generate all windows from `face_tile_windows(1)` rather than carrying a
   hand-written four-tile list.
2. Preserve all 205 prognostic and four boundary channels.
3. Give every destination a zero-padded row-major tile ID in its filename,
   for example `tile00` through `tile35`.
4. Record tile ID, face, absolute bounds, channel names, codec, source path,
   and build version in attributes and a root manifest.
5. Remove any stale completion marker during initialization.
6. Let fill jobs write only disjoint time chunks. They must never rewrite
   metadata or consolidate metadata concurrently.
7. Add a verify-only mode which publishes `.complete.json` atomically only
   after every check passes.

### Ordering is a correctness requirement

`DataConfig` currently expands a cache directory with lexically sorted
filenames. Face rank assignment treats tile IDs as a row-major 6 x 6 grid.
Filenames whose first varying field is an unpadded `i` coordinate do **not**
sort into that order; they can silently destroy the intended contiguous rank
blocks.

The urgent build should prefix stores with `tile00` through `tile35`, where
the ID is the index returned by `face_tile_windows(1)`. In addition, add a
startup assertion/test that the loaded catalog has those windows in exact
row-major order. A later cleanup can make the directory loader sort by the
recorded `(face, j_start, i_start)` attributes instead of relying on names.

## Fast production job

Use an independent CPU Slurm chain, so it survives the interactive shell and
any H200 job being stopped:

```text
init -> 32-way time-array fill -> verify
```

Production resource choice per fill task:

- partition/account: the existing `mit_normal` CPU setup;
- 16 CPUs, used by the Blosc codec;
- 96 GB RAM;
- 12-hour limit; and
- 32 disjoint contiguous time ranges, throttled to 24 concurrent tasks.

Eight GB is not viable. One decoded 205-channel float16 face is about 7.65 GB
and the 36 destination tile buffers add about 8.35 GB before Python, Zarr,
codec, and filesystem workspaces. The expected live payload is already over
16 GB. Forty-eight GB should normally be sufficient, but 96 GB makes an
unexpected codec copy a performance event rather than an out-of-memory failure.

Each task receives about 322-323 of the 10,311 timestamps. At most 24 tasks
produce 24 sequential source streams and up to 384 codec threads; array tasks
may run on different nodes. Their output keys are disjoint, so partial Slurm
scheduling is safe. `MAX_CONCURRENT` is an environment override if observed
filesystem load or queue placement argues for a different cap.

**Cap concurrent readers near 32, not hundreds.** The CPUs inside each task are
for Blosc's in-memory codec work; the Python recutter issues one source chunk
read at a time. `scripts/_bench/face_read.py` measured
the cliff directly: 4 ranks x 8 threads read a face in 2.91 s where 4 x 15 took
7.85 s on identical work -- 2.7x slower from concurrency alone, and the same
collapse that is currently destroying the rental run with 90-228 s read tails.
Do not add a per-task source-read pool: 24 tasks already provide 24 concurrent
streams, while compression is thread-parallel and local.

**Read from the disk copy, write to scratch.** Reading
`/orcd/data/abodner/002/cody/LLC_patch/face_1/...` and writing
`/orcd/scratch/orcd/014/...` puts the 37 TB of reads and the 40 TB of writes on
different filesystems, so they do not contend for the same bandwidth. Doing
both on scratch halves the effective throughput for no benefit, and the disk
copy is the pristine original.

**Do not run this against the filesystem a training job is reading.** The
binding constraint below is bandwidth, and the rental train is already
I/O-starved. Running both concurrently starves both. Either pause training for
the recut, or confine the recut's reads to disk and the train's to scratch and
accept that writes still land on scratch.

Before the production array, run a tiny separate-root smoke build covering a
few timestamps and all 36 tiles. This is a correctness check, not a parameter
sweep. It should prove the full geometry and codec path in minutes. If the
smoke is clean, submit the production chain immediately.

### What actually sets the wall clock

Not CPU. The job moves **37 TB in and about 40 TB out, 77 TB total**, and at a
sustained aggregate of 2.7 GB/s -- the best figure measured on this store --
that is an **8-hour floor before any compute**. For contrast the compression
side, 86 TB at roughly 0.3 GB/s per core is comfortably hidden by the aggregate
16-core tasks. Increasing beyond 16 CPUs per task is not expected to move the
wall clock; aggregate bandwidth is the limiting resource.

A time subset is the only lever that shortens it materially, and it is small
here: `train_time` plus `val_time` span 2011-09-13 to 2012-10-14, about 9,500
of the 10,311 timestamps, so dropping the unused tail saves roughly 8%. If a
partial store would let training start sooner, shard by time and order the
array so the earliest training months land first -- the store is usable for
the opening epochs before the tail finishes.

The older four-tile recut took about four hours with eight time shards and
settled around 10.8-11.8 seconds per timestamp per shard. This build writes
nine times as many target tiles and reads a whole face, while using four times
as many time shards. A safe expectation is still **multiple hours**, roughly
4-8 hours depending on queue placement and filesystem load. Finishing in less
than two hours would require sustained aggregate compressed I/O on the order
of 10 GB/s or more plus compression and metadata work; it should not be
promised. The job is restartable by failed array index, so speed does not need
to come at the cost of an unverifiable store.

## Verification gate

The verify job must fail without writing `.complete.json` unless all of the
following hold:

1. Exactly 36 tile stores exist and their IDs are `00..35`.
2. Catalog order exactly matches `face_tile_windows(1)`.
3. Every tile reports shape 752 x 752 and the expected absolute `x`/`y`
   coordinates and LLC bounds.
4. Prognostic shape/chunks are `(10311, 205, 752, 752)` and
   `(1, 205, 752, 752)`.
5. Boundary shape/chunks are `(10311, 4, 752, 752)` and
   `(1, 4, 752, 752)`.
6. Every one of the 742,392 time-varying chunks exists
   (`36 * 10311 * 2`).
7. Channel order, times, normalization metadata, masks, `XC`, `YC`, and `rA`
   match the source.
8. Source slices and recut values are bitwise identical at representative
   train, validation, first, middle, and final timestamps for corner, edge,
   and interior tiles.
9. Duplicated seam cells are bitwise identical between neighbouring stores.
10. `build_tile_catalog` recovers a 4320 x 4320 face with only the expected
    32- and 48-cell overlaps.

The marker should contain the source path, source metadata fingerprint, tile
windows, time count, channel counts, codec, job IDs, and verification time.
Training scripts must require this marker before selecting the new directory.

## Trainer changes

The trainer already has the correct fundamental path for pre-cut stores:
passing a directory as `data.data_location` expands it into one replay source
per `.zarr`, and `_build_group_frame_reader` deliberately falls back to the
per-tile reader because there are no shared source chunks left to stream.
Therefore this does not need a new data-loader architecture.

The rental script does need a deliberate layout switch:

1. Set `data.data_location` to the verified 36-store directory.
2. Omit `data.llc_tiles`; those windows mean "cut 36 views from one store" and
   are wrong for already-cut stores.
3. Require `.complete.json` before launch.
4. Log every tile's path, ID, bounds, and chunks once at startup, then summarize
   `one prognostic + one boundary chunk per transition`.
5. Keep the current packed-store path available as an explicit fallback, not an
   automatic fallback after a failed marker check.

For the immediate synchronous six-H200 run, keep the current face topology,
optimizer/resume state, curriculum, and offloaded validation unchanged. Only
the training data layout changes. The H100 validator may continue to use the
old packed full-face store: its current face-validation path requires the
group reader, and the validation companion already overrides the data path and
supplies `data.llc_tiles`. Extending face validation to separate tile stores is
useful later but is not on the launch-critical path.

The rank-local branch can use the same directory after synchronous training is
stable. It must pass the same ordering/one-chunk assertions, but its optimizer
and timestamp-sampling questions remain separate from this storage change.

## Tests before committing the rental

Add small synthetic tests for:

- all 36 clamped windows, including 48-cell boundary seams;
- deterministic directory-to-row-major tile ordering;
- a pre-cut tile read touching exactly one prognostic and one boundary chunk;
- synchronous face replay over separate stores;
- rank-local block replay over separate stores; and
- rejection of a directory without a valid completion marker in the rental
  wrapper.

Then run two real-data checks:

1. A cold/read benchmark of the six-rank synchronous assignment from the new
   directory, compared with the packed face store.
2. A short synchronous preflight that crosses at least two replay refresh
   boundaries. The acceptance signal is sustained iteration time and bounded
   `data_load_time`, not a fast first 24 steps before a refresh.

Only after that preflight should the rental resume from its emergency
checkpoint. Rank-local performance testing can happen independently afterward.

## Definition of done

The storage work is complete when:

- the independent Slurm chain has ended successfully;
- `.complete.json` exists and the verifier log is clean;
- the trainer sees 36 row-major stores and one spatial chunk per array/tile;
- a refresh-crossing synchronous preflight has no large read-tail regression;
  and
- the rental and its validator companion have explicit, intentional data paths.

Until those conditions hold, keep the old packed face cache intact and do not
delete or modify it in place.

## Implementation status (2026-10-02)

The build path is implemented and preflighted:

- `JOBS/other/1-face_recut.sh` is the new 16-CPU/96-GB batch worker. The older
  `recut_llc_patch_caches.sh` was deliberately left unchanged.
- `JOBS/other/submit_1-face_recut.sh` submits the independent init, bounded
  32-way fill, and verify dependency chain.
- `scripts/recut_llc_patch_caches.py` now emits stable indexed store names,
  supports exhaustive verification, and publishes `.complete.json` atomically.
- The synchronous rental launcher selects the pre-cut directory only when the
  completion marker exists and omits `data.llc_tiles` for that layout.
- The synthetic end-to-end recut and missing-chunk failure tests pass, as do the
  existing tiling and chunk-reader suites (101 tests total).
- A dry run against the real disk source confirms one source store, 205 + 4
  channels, 10,311 timestamps, and complete coverage of all 36 requested
  windows. The production output directory is still absent; no build job has
  been submitted yet.

### Mandatory 48-hour capability gate

Do not launch the production recut from the metadata dry run alone. First run
`JOBS/other/submit_1-face_recut-capability.sh`. It creates a persistent,
fully verified miniature at
`752-36-tile-face1-48h-test`: all 36 real 752 x 752 stores, but only
source indices `0:48` (48 hourly timestamps). Compact slicing is important;
the miniature has time shape 48 rather than a 10,311-row array whose later
chunks are absent.

After that job writes `.complete.json`, run
`JOBS/other/test_1-face_recut_async.sh`. The test uses the
`Ocean_Emulator-asynchronous-train_blending` worktree with seven healthy H200
ranks, rank-local 3 x 3 blocks, offloaded validation, four replay rows, and 14
optimizer steps. Fourteen steps cross the first configured refresh boundary
at step 10, so the test covers initial seeds, ordinary transitions, a reseed,
DDP, and the per-epoch EMA snapshot. Its train and validation windows each use
one day of the miniature.

The async worktree needed one startup correction found by this review:
offloaded validation must not require the packed group-frame reader. Separate
pre-cut stores intentionally have no such reader. Focused rank-local/offload
tests pass after gating that requirement to inline validation, and a synthetic
36-store probe reconstructs all 16 row-major 3 x 3 blocks.

This makes the pre-cut layout usable by **both training topologies in the async
worktree**: `blend_scope=face` for the synchronous six-rank fallback and
`blend_scope=rank` for the seven-rank experiment. It is not yet a promise that
both Git worktrees can launch it. The main worktree still requires its packed
group reader for inline face validation; backporting offloaded validation or
adding a separate-store inline validator is a distinct change. The rental
launcher already executes the async worktree, so this does not block either
operational topology.

Production submission is authorized only if the GPU capability job exits zero,
writes `ema_ckpt_ep0001.pt`, reports 36 replay sources / 16 spatial blocks, and
shows bounded post-refresh `data_load_time`. A mere successful initialization
is not enough.
