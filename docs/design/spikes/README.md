<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC4320 storage spikes

These bounded experiments reduce ambiguity in the LLC4320 data-engineering
design. Local spikes are synthetic. The scripts under
[`cluster/`](cluster/) read a bounded part of the production dataset through
Slurm but never modify it; all output goes to user scratch.

The measured run is summarized in
[`ORCD-RESULTS-2026-09-03.md`](ORCD-RESULTS-2026-09-03.md).

## ORCD real-data benchmark suite

The Slurm suite uses an immutable fixture at
`/orcd/scratch/orcd/008/merose/llc_sharding_spikes_20260903`. It contains two
times of `U` and `Theta` for face 1 over the source-chunk-aligned
`j/i=720:3600` region, plus two complete face-1 `Eta` planes. The manifest
records source object paths, sizes, modification times, fixture SHA-256 hashes,
and read durations. The decoded fixture occupies about 6.92 GB.

The suite is a dependency graph:

```text
codec tournament
├── spatial-inner tournament ──> outer-shard tournament
├── depth/spatial tournament ──> balanced general-workload ranking
└── surface-layout tournament
```

Each candidate validates exact float32 equality and records physical bytes,
file count, encoding time, memory, four independent Cody-tile reads, a combined
union read, and general-purpose spatial/depth selections. Surface candidates
also record an arbitrary crop and a full-face scan. The balanced ranking gives
equal weight to training tiles, union access, one-level analysis, a 17-level
subset, and an all-depth crop; it is evidence for review, not an automatic
production-schema decision.

The first real-data smoke test used `U`, one time, Blosc Zstd level 1 with
bitshuffle, a `72 x 72` logical spatial chunk, and a `1440 x 1440` shard. It
stored 1.068 GB from 1.692 GB of decoded fixture values (ratio 0.631), encoded
in 5.07 seconds, read Cody's four tiles serially in 1.22 seconds, and read their
union in 1.19 seconds. Exact equality passed. Warm-cache timing on a single
scratch fixture is not a throughput projection.

The completed codec family is `21922525`. The downstream run used
spatial inner `21923800`, outer shard `21923802`, surface `21923803`,
depth/spatial `21923804`, halo controls `21923808`, time fixture `21924235`,
time packing `21924236`, constant-volume volume shards `21924852`, retained
Xarray/Icechunk pilot `21924952`, concurrency `21925248`, and surface
time/space `21925498`, plus dependent selectors. The sparse all-face pilot
`21926005`, real requeue/ledger run `21927118`, and successful independent
topology oracle `21927585` also completed. Job IDs document this run only;
reruns should use the scripts rather than assuming these results still exist.

The metadata-only inventory job `21928256` recorded all 67 source arrays,
their shapes, dimensions, dtypes, chunks, codecs, fill values, attributes, and
symlink-resolved paths without reading array payloads. Its JSON result is the
starting point for the production release manifest.

The 52-interface-level `W` depth bake-off `21928816` selected logical depth 17
inside a nominal physical depth of 68. This preserves the general volume
logical unit while avoiding a pathological second one-level shard that a
physical depth of 51 would create.

The complete-layout projection `21929213` mapped all 67 arrays to the selected
layout families. It projects at most 4,541,414 physical data objects and about
3.108 billion compressed inner chunks; the latter remain inside shard indexes
and must not become individual conversion tasks.

## Icechunk failure recovery

[`icechunk_recovery_spike.py`](icechunk_recovery_spike.py) ran locally and on
ORCD with Icechunk 2.2.0. It verified that a killed uncommitted writer remains
invisible, the prior snapshot remains immutable, an idempotent retry can
commit, and a stale concurrent commit fails with `ConflictError`. Icechunk
warned that local-filesystem storage is unsafe for concurrent committers, which
confirms the design requirement for one serialized commit coordinator. Disk
full, manifest splitting, completion-ledger recovery, and garbage collection
remain for the multi-worker pilot.

The follow-up fork/merge spike (`21924500`) used eight serializable worker
sessions, retried a simulated capacity failure, merged disjoint shards through
one coordinator, and committed an exact result. It created four time-split
manifest files, preserved the initial snapshot, garbage-collected the failed
worker's orphan chunk and unreachable transaction state, and verified that the
release snapshot remained readable. The requeue/ledger spike (`21927118`)
subsequently survived a real Slurm requeue, rejected duplicate claims,
recovered a stale claim, verified artifact hashes, kept a dead coordinator's
uncommitted state invisible, and reconstructed one exact release commit.

## Zarr sharding read spike

[`zarr_sharding_read_spike.py`](zarr_sharding_read_spike.py) models a 3 by 3
tile region at 1:10 spatial scale. It reads the center tile plus a halo and
records storage requests. Zarr 3.3.0 produced:

| Configuration | Physical objects | Objects touched | Read calls | Bytes returned |
| --- | ---: | ---: | ---: | ---: |
| 72-square source-like chunks | 9 | 9 | 9 | 537,144 |
| Independent 18-square chunks | 144 | 36 | 36 | 145,068 |
| 18-square chunks in 72-square shards | 9 | 9 | 17 | 219,642 |

The sharded case used suffix requests for shard indexes and byte-range requests
for neighboring inner chunks. It transferred about 59% fewer bytes than the
source-like case while retaining nine physical data objects. It transferred
about 51% more bytes than independent small chunks because Zarr coalesced some
adjacent ranges. Stored compressed bytes were about 8% higher than the
source-like case for this synthetic smooth, 40%-NaN dataset.

These measurements use an instrumented in-memory store. They establish codec
behavior, not expected ORCD NFS throughput.

## Source physical-size estimate

[`estimate_source_physical_bytes.py`](estimate_source_physical_bytes.py) avoids
an unbounded `du` crawl across the LLC4320 hierarchy. It scans static arrays
exactly and samples 32 complete, evenly-spaced time-chunk subtrees for every
time-dependent array. A one-CPU `mit_quicktest` run estimated:

| Scope | Compressed payload | Allocated bytes | Files |
| --- | ---: | ---: | ---: |
| Complete root, following `Theta`/`Salt` symlinks | 1.0317 PB | 1.0323 PB | 24.32 million |
| Eight arrays currently used by Samudra | 755.2 TB | 755.6 TB | not reported separately |

The aggregate payload sampling standard error was about 0.21 TB. This is not an
authoritative quota or billing measurement and does not quantify systematic
sampling or filesystem-accounting bias.

Run it on one Slurm compute node and capture the JSON result:

```bash
srun --partition=mit_quicktest --time=00:10:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G \
  python3.12 docs/design/spikes/estimate_source_physical_bytes.py \
  /orcd/data/abodner/003/LLC4320/LLC4320 --samples 32
```

## Icechunk round trip

[`icechunk_sharding_roundtrip.py`](icechunk_sharding_roundtrip.py) confirms that
Icechunk 2.2.0 can commit, snapshot, reopen, and partially read the same Zarr v3
sharded representation through both Zarr Python and Zarrista's asynchronous,
Rust-backed Icechunk reader. The test uses temporary local-filesystem storage.
It also emits Icechunk's expected warning that filesystem storage is unsafe for
concurrent committers.

## Rust codec and sharding compatibility

[`rust_zarr_compatibility/`](rust_zarr_compatibility/) writes equivalent Zarr v3
sharded arrays with Zarr Python and reads them with `zarrs` 0.21.2, matching the
version used by Samudra PR #800. It verifies both Blosc Zstd with bitshuffle and
the Zarr v3 core Zstd codec. Both returned the expected 25,600-value subset.

For the smooth integer-ramp fixture at compression level 5, the physical data
objects occupied 74,278 bytes with Blosc Zstd/bitshuffle and 550,171 bytes with
core Zstd. This demonstrates why codec portability and compression efficiency
must be tested separately; it is not an LLC4320 compression estimate.

Run the spike in a disposable directory:

```bash
spike_tmp=$(mktemp -d /tmp/llc-rust-compat.XXXXXX)
uv run --isolated --with 'zarr>=3.3,<4' --with numpy \
  python docs/design/spikes/rust_zarr_compatibility/make_fixture.py \
  "$spike_tmp/fixture.zarr"
cargo run --manifest-path \
  docs/design/spikes/rust_zarr_compatibility/Cargo.toml -- \
  "$spike_tmp/fixture.zarr"
```
