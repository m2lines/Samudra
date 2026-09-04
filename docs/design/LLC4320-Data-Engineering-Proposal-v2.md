<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC4320 Data Engineering Proposal, v2

Status: draft for design review
Updated: September 3, 2026

## Executive Summary

LLC4320 is large enough that its current Zarr v2 storage layout, rather than GPU
compute, can determine training throughput. Surface fields are compressed as
one global object per hour, and Cody's current `752 x 752` predicted tile
intersects nine independently compressed `720 x 720` source chunks. Both
patterns move and decompress substantially more data than a training sample
uses. This needs to
be addressed now because four-tile overlap experiments are under way and
Samudra is gaining a persistent Rust loader intended to remove data-wait stalls;
that loader cannot recover bytes that the physical layout forces it to read.

This proposal covers a new, immutable, generally usable scientific
representation of the existing dataset: its arrays and dimensions, logical
chunks, physical shards, codecs, transactional build and publication, failure
recovery, and validation. Cody's training experiments provide an important
empirical workload, but they do not define the archive schema. It does not
authorize changes to the source, prescribe a model architecture or final halo
width, or make a lossy float16 cache the scientific record. Conventional
Xarray access is the required initial reader and public contract; the native
Rust loader is a preferred follow-on optimization for Samudra.

The proposed solution is to rewrite the required arrays as float32 Zarr v3,
packing moderately sized, independently compressed inner chunks into larger
shards and publishing pinned snapshots through Icechunk. A tile plus any
supported halo can then read only intersecting inner chunks without duplicating
halo values or creating millions of physical files. The exact chunk, shard,
and codec configuration remains gated on bounded ORCD experiments. If the full
rewrite is too slow or large, the fallback is to virtualize the existing 3-D
chunks and materialize improved surface chunks plus a halo sidecar.

## Problem

Loading the current LLC4320 Zarr store is slow because its physical units do
not match training selections.

The latest tracked experiment predicts four `752 x 752` patches. Each contains
a `720 x 720` core extended by 16 cells on every side, so adjacent tiles share
a 32-cell predicted-and-blended overlap. This is not yet a context-only halo:
the branch explicitly reserves halo fields for future work but supplies none.
The storage design must serve the current overlap reads efficiently while
remaining capable of serving a later halo without another full rewrite.

### Source dataset

The source is the read-only, unconsolidated Zarr v2 store at:

```text
/orcd/data/abodner/003/LLC4320/LLC4320
```

It contains 10,311 hourly timesteps, 13 LLC faces, `4320 x 4320` horizontal
points per face, and 51 center depth levels. A complete metadata inventory
found 67 arrays totaling 2.731912 PB decoded. Logical size is the decoded array
size, not compressed bytes physically stored on disk.

A bounded filesystem inventory estimated the complete root at 1.0317 PB
decimal (0.9163 PiB) of compressed file payload and 1.0323 PB decimal of
filesystem-allocated space across approximately 24.32 million files. This
follows the `Theta` and `Salt` symlinks, so it includes their bytes on `002`,
and it includes source arrays outside Samudra's current training set. The eight
arrays currently in scope (`U`, `V`, `Theta`, `Salt`, `Eta`, `oceQnet`,
`oceTAUX`, and `oceTAUY`) account for an estimated 755.2 TB decimal of
compressed payload.

The inventory scanned static arrays exactly and sampled 32 complete,
evenly-spaced time-chunk subtrees for each time-dependent array. Its aggregate
sampling standard error was about 0.21 TB, but that does not cover systematic
filesystem-accounting or sampling bias. Obtain authoritative server-side usage
and quota figures before allocating production capacity; do not treat the
estimate as a billing or deletion number.

`Theta` and `Salt` are symbolic links into
`/orcd/data/abodner/002/shared_datasets/LLC4320/LLC4320`. The remaining main
arrays are under the `003` mount. Any virtual representation must preserve both
absolute source locations, and jobs may encounter independent bottlenecks on
the two storage servers.

The metadata-only public-inventory job found 65 arrays rooted on `003` and the
two symlinked arrays on `002`. The complete families are:

| Family | Count and arrays | Representative shape | Current chunks |
| --- | --- | --- | --- |
| Time-varying 3-D | 5: `U`, `V`, `Theta`, `Salt`, `W` | `(10311, 51 or 52, 13, 4320, 4320)` | one time, one face, `720 x 720`, full depth |
| Time-varying surface | 17: `Eta`, `KPPhbl`, `PhiBot`, six sea-ice fields, `SSH`, `SSH_notides`, and six ocean flux/stress fields | `(10311, 13, 4320, 4320)` | `(1, 13, 4320, 4320)` |
| Static 3-D masks/fractions | 6: `hFacC/S/W`, `mask_c/s/w` | `(51, 13, 4320, 4320)` | full depth, one face, `720 x 720` |
| Static horizontal geometry | 19: coordinates, angles, depth, metrics, and cell areas | `(13, 4320, 4320)` | one face, `720 x 720` |
| One-dimensional coordinates/reference fields | 20, including time/iteration, horizontal indexes, vertical coordinates, and hydrostatic references | varies | generally complete 1-D arrays |

The surface arrays are chunked globally and hourly, not per face. One chunk
contains all 13 faces. Sample compressed `Eta` chunks were 366-372 MB. A
`720 x 720` crop therefore reads and decompresses a global hour before
discarding almost all values.

The 3-D arrays are already stored in model-tile-sized chunks, but a tile plus
halo may intersect as many as nine chunks. A representative ocean-containing
3-D chunk was tens of MB compressed; its decoded size is about 101 MiB.

### Chunk-count consequence

Changing a surface array to logical chunks of `(1, 1, 720, 720)` would produce:

```text
10,311 times * 13 faces * 6 * 6 tiles
= 4,825,548 logical chunks per variable
```

Storing each logical chunk as a file would create millions of new files per
variable. Zarr v3 sharding is intended for this situation: small independently
readable chunks can be packed into larger physical objects. See the
[Zarr sharding documentation](https://zarr.readthedocs.io/en/latest/user-guide/performance/#sharding).

### Overlap and future halo access

Each current `752 x 752` tile deliberately straddles the `720 x 720` source
grid and intersects nine prognostic chunks. The complete four-tile group spans
`1472 x 1472` and intersects 16 distinct source chunks per variable and time.
The actual byte amplification is content- and compression-dependent, but the
current layout cannot read a subregion of a compressed Zarr v2 chunk
independently.

Cody's branch reads all 51 levels of `U`, `V`, `Theta`, and `Salt`. An ordinary
transition needs two prognostic times and the current boundary time; replay and
seed refreshes request single times. Logical time chunks should therefore stay
at one, even if a physical shard packs multiple independently compressed times.
Cross-face tile groups and true halos are not implemented yet, so face-edge and
corner behavior still require an independent correctness oracle.

### Evidence from Cody's experiment branch

The branch was reviewed at commit `aa336ee7a17ecf8c8ad127081e5b1fa4519cbcbd`;
the latest tracked four-tile launch configuration is preserved in its parent
commit `ed7ab891beac51d198b15487aa47fc719de274a6`. The relevant findings are:

- four same-face tiles cover `1424:2176` or `2144:2896` on each spatial axis;
  each is `752 x 752`, with a `720 x 720` nominal core and 32 cells of overlap
  between adjacent predicted regions;
- the complete group union is `1472 x 1472`; reading it once and scattering is
  therefore a legitimate alternative to four independent storage requests;
- `U`, `V`, `Theta`, and `Salt` use all 51 levels, while `Eta` and surface
  forcing are scalar planes; ordinary transitions require two prognostic times
  and one current forcing time;
- the current Python path reads the four tile datasets serially within one
  request while requests can be prefetched by a thread pool;
- the native raw reader already requests a face/window subset and deduplicates
  inflation of an all-depth source chunk, but it documents the current global
  surface chunk as the dominant amplification problem; and
- Cody's packed cache defaults to float16 Blosc LZ4 level 5 with byte shuffle.
  This is an important throughput control, not an acceptable archival dtype.

These observations support smaller independently compressed spatial units and
moderate physical sharding. They do not support encoding the four tiles as the
public archive: the branch itself rejects cross-face groups until an explicit
rotation operator exists, and its packed-cache dimensions are training
specific.

## Decision summary

The preferred durable representation is a native Zarr v3 sharded dataset in an
Icechunk repository, subject to representative ORCD performance, capacity, and
Xarray acceptance spikes. [Sharding](https://zarr.readthedocs.io/en/latest/user-guide/performance/#sharding) separates the independently readable unit
(an inner Zarr chunk) from the physical storage and write unit (a shard). It can
therefore serve a tile plus halo without reading nine full neighboring tiles
and without storing duplicate halo values.

This is more future-proof than a training-specific halo sidecar and removes the
need for a DataTree layout. The tradeoff is substantial: existing compressed
Zarr v2 chunks cannot be subdivided through virtual references. The 3-D
prognostic fields must be decoded and rewritten to gain independently readable
inner chunks.

The decision is therefore gated:

1. Prototype sharded and halo-sidecar layouts on representative production
   chunks, including an LLC face seam.
2. Measure cold and warm read throughput, compression ratio, write throughput,
   manifest size, file count, concurrent training-loader throughput, arbitrary
   spatial/depth subsets, time-series access, and full-face scans on ORCD
   storage.
3. Gate publication on exact correctness and sufficient training throughput
   through Zarr Python/Xarray (Plan A).
4. Validate that the schema and codec preserve a credible path to the native
   Samudra Rust loader (Plan B).
5. Adopt native sharding if it meets the agreed performance and storage bounds.
6. Retain virtual prognostic chunks plus a native halo sidecar as the fallback
   if a full rewrite is operationally or economically unacceptable.

### Measured decisions to date

The bounded real-data tournament has resolved several choices. These remain
subject to the approved-destination capacity and concurrency gate:

| Decision | Selected candidate | Evidence |
| --- | --- | --- |
| Durable strategy | native Zarr v3 sharding, with halo sidecar only as fallback | Supports general spatial/depth subsets without duplicated topology-specific values; real-data range-decoded smoke passed. |
| Dtype | float32 | Exact scientific archive; Cody's float16 cache remains a derivative-only option. |
| Volume codec | Blosc Zstd level 5 + bitshuffle | Smallest of seven real-data candidates at 3.563 GB for 6.768 GB decoded; exact Python and pinned-`zarrs` reads passed. |
| Volume logical chunk | one time, 17 levels, one face, `120 x 120` | Best balanced result across full-depth training and arbitrary 1/17/51-level crops while remaining within 0.7% of the comparable full-depth chunk's physical bytes. |
| Volume physical envelope | two times, all 51 levels, one face, `1440 x 1440` | Won every timed workload in the constant-volume time/space tournament and used 18 fixture objects, versus 34 for one-time `2160²`; projected at about 603,000 objects per volume variable. |
| `W` depth adaptation | logical depth 17 in nominal physical depth 68; otherwise the volume layout | A real 52-level `W` bake-off kept one data object per time/spatial envelope, used the fewest bytes, and won the balanced full-/single-/17-depth read score. A physical depth of 51 would create a second one-level shard. |
| Surface layout | logical `(time=1, face=1, 360, 360)` in physical `(time=24, face=1, 1080, 1080)` | Best balanced result across one-time tile/full-face and 24-time point/crop workloads, with about 89,000 projected objects per surface variable. |
| Static 3-D masks/fractions | logical `(k=17, face=1, 120, 120)` in physical `(k=51, face=1, 1440, 1440)` | Matches the selected arbitrary-depth/spatial access units; all-face mask corners passed exact reads. |
| Static horizontal geometry | logical `(face=1, 360, 360)` in physical `(face=1, 1080, 1080)` | Uses the surface spatial layout without an artificial time axis; all-face geometry corners passed exact reads. |
| Complete-archive object projection | at most 4,541,414 physical data objects across 67 arrays | Deterministic ceiling from every source shape and selected shard family, before repository metadata and without omitting all-fill shards; substantially below the source's approximately 24.32 million files. |
| Commit protocol | forked disjoint workers, one merge/commit coordinator | Eight-worker fork/merge, failure retry, snapshot isolation, split manifests, and garbage collection passed. |
| Plan A format/protocol read | pinned Icechunk snapshot through Xarray | An eight-time, all-depth, real-`Theta` pilot passed exact full-array and subset equality using the selected codec, logical chunks, and physical shards. |
| Prototype read concurrency | eight outer readers with Zarr async concurrency one | At 16 readers, p95 latency rose 25-55% versus eight on each tested mount/store. Retune on the approved destination; this is not a production limit. |

The bounded schema and all-face topology are now selected; public-destination
and production-scale gates remain before a rewrite is authorized.

The same projection contains approximately 3.108 billion independently
compressed inner chunks. Those entries live inside shard indexes rather than
as separate filesystem files, but they make inner-chunk-at-a-time conversion
or task scheduling unacceptable. Writers must decode and produce complete
physical shards as the bounded work unit.

### Intended users and workload contract

The release is a cluster-wide scientific archive, not a private Samudra cache.
It must retain source variable names, dimensions, coordinates, dtypes, fill
semantics, and scientifically relevant attributes in a conventional root-level
Xarray dataset. Users must not need Samudra, a DataTree, or knowledge of Cody's
tile convention to open it.

Schema selection gives equal design standing to these workload classes:

1. contiguous arbitrary spatial crops at one, several, or all depth levels;
2. complete face and multi-face analysis;
3. short and long time-series access for bounded spatial regions;
4. Cody's four overlapping `752 x 752` full-depth training tiles, both as four
   independent requests and as one `1472 x 1472` union read; and
5. future tile-plus-halo reads, including face seams and LLC corners.

No single benchmark is sufficient. The production choice must be on the
measured Pareto frontier for these workloads, fit the capacity envelope, and
avoid pathological read amplification for any common class. A Samudra-specific
float16/channel-packed representation may still be built later as a derivative
cache with its own lifecycle; it is not part of this archive.

## Constraints

### Safety and correctness

- Never write to or delete the source Zarr store.
- Treat the source as an immutable dependency until a separately authorized
  migration and retention decision is made.
- Do not expose partially populated arrays as a published dataset.
- Preserve float32 values in the durable scientific dataset. Float16 may be
  evaluated later as an explicitly lossy training cache, not silently used as
  the archival representation.
- Validate LLC face orientation, corner topology, staggered `U`/`V` grids,
  masks, fill values, NaNs, coordinates, attributes, and time ordering.

### Storage

- Minimize duplicate physical bytes and temporary stores.
- Measure physical compressed bytes; do not use `Dataset.nbytes` as a disk
  estimate.
- Avoid millions of small files where shards can provide the same logical read
  granularity.
- The source `003` filesystem reported about 33 TB free out of 700 TB during
  discovery. It must not be assumed to have room for the destination.
- Select and confirm a destination, quota, backup policy, and garbage-collection
  policy before materializing production data.

### Performance

- Optimize for sustained concurrent loader throughput and GPU utilization,
  not only single-slice latency.
- Keep inner chunks large enough for useful compression and decompression
  throughput. Zarr recommends at least approximately 1 MB uncompressed as a
  starting point for Blosc-backed chunks.
- Bound nested concurrency. Zarr estimates total I/O concurrency as roughly
  Dask threads times Zarr asynchronous concurrency; uncontrolled multiplication
  can overload shared storage.
- Read each source chunk at most once per conversion attempt and produce output
  in source order where practical.

### ORCD operation

- Run computation through Slurm compute nodes, never on login nodes.
- Use `mit_quicktest` for short smoke tests and `mit_normal` or `pi_abodner` for
  longer CPU jobs. At discovery time, `pi_abodner` exposed three nodes, 448
  CPUs total, and a 6-day 6-hour maximum runtime.
- Use bounded, resumable work units. Preemptible jobs must checkpoint and be
  safe to requeue.
- Treat the storage server as the likely throughput ceiling. More workers help
  only until aggregate source bandwidth saturates.
- Use node-local temporary storage for logs, manifests under construction, and
  spill where appropriate; do not stage multi-terabyte copies casually.

### Compatibility

- Icechunk currently requires Zarr Python 3. Samudra currently pins `zarr<3`,
  and the data subproject's conda environment pins Zarr 2.18.2.
- Build the conversion tool in a separate, reproducibly locked environment.
- Treat array-format compatibility and store-protocol compatibility as separate
  gates. A reader may understand Zarr v3 sharding and its codecs but still be
  unable to open an Icechunk session.
- Add an LLC canonicalizer and an Icechunk-aware native reader before treating
  the new repository as training-ready.
- Preserve the existing root-level Xarray dataset contract initially. A
  DataTree adds organization but no inherent I/O improvement.
- `Eta` is used in both Samudra's prognostic and boundary variable sets; store
  it once rather than duplicating it into semantic groups.

### Consumer priority and compatibility matrix

The production contract is Xarray-first. It should also preserve a credible
native Rust path.

| Priority | Reader path | Sharded Zarr v3 | Icechunk | LLC readiness | Role |
| --- | --- | --- | --- | --- | --- |
| Plan A | Zarr Python/Xarray | yes | yes, through an Icechunk session store | current Python LLC canonicalization exists | Required production reader, correctness oracle, and initial performance gate. |
| Plan B | Samudra PR #800 Rust loader | yes for a plain filesystem store; locally verified with its `zarrs` 0.21.2 dependency | no; it constructs `FilesystemStore` directly | explicitly out of scope in the PR | Preferred native optimization path; requires LLC semantics and an Icechunk storage adapter. |
| Compatibility probe | Zarrista | yes | yes, through its async Icechunk-session bridge; locally verified | low-level array API only | Independent evidence that the representation is consumable through `zarrs`. |

PR #800 establishes a useful seam: Python owns canonical semantics, sampling,
masking, normalization, and DDP schedules, while Rust owns persistent reads,
rollout-wide plane deduplication, bounded prefetch, decompression, and pinned
buffers. Its reported quarter-degree experiment reduced mean epoch iteration
time from 22.31 to 4.13 seconds, mainly by removing cold-load stalls. The LLC
reader should preserve that separation. It should push tile-plus-halo subsets
into Rust rather than materializing a face or global plane and cropping in
Python.

Icechunk does not force Python-only access. `zarrs_icechunk` implements an
Icechunk store for Rust `zarrs`, and Zarrista already uses this route. It is
nevertheless a real integration dependency, not transparent filesystem access. The current
`zarrs_icechunk` API is asynchronous, while PR #800 uses synchronous
`FilesystemStore`, and Icechunk session serialization is version-coupled.
Pin and test the Rust and Python Icechunk versions together.


## Strategic alternatives

### A. Virtual prognostic data plus a halo sidecar

Keep the existing `720 x 720` 3-D chunks as virtual references, materialize
better boundary chunks, and store precomputed halos separately.

Advantages:

- Avoids rewriting the dominant 3-D fields.
- Fastest path to a usable training layout.
- Source-oriented halo extraction can read each source tile once.

Disadvantages:

- Encodes assumptions about tile size, halo width, and LLC topology.
- Requires at least a central-data read plus sidecar reads.
- A source-oriented ring may still require up to eight neighboring ring reads
  and runtime rotations; a destination-ready ring duplicates/rearranges data.
- The new repository remains dependent on the original files.

### B. Native Zarr v3 sharding

Rewrite training arrays into smaller logical chunks packed into larger shards.
The logical chunks are independently compressed and byte-range readable, while
the shard controls physical file count and write granularity.

Advantages:

- No duplicated halo representation.
- Supports multiple halo widths and future tile/access patterns.
- Standard Zarr indexing; no DataTree or custom ring reconstruction required.
- Can improve depth-subset access if depth chunking is also redesigned.

Disadvantages:

- Requires a complete decode/re-encode of the 3-D variables; virtualization
  cannot subdivide their existing compressed blobs.
- Smaller compression boundaries may increase physical bytes.
- More inner chunks increase shard-index, range-request, and decompression
  overhead.
- Partial reads depend on efficient byte-range behavior from ORCD's NFS path.
- Writing a shard is the minimum safe independent write unit, which constrains
  parallelism and memory.

### Recommendation

Adopt alternative B. The representative ORCD spikes showed no halo-sidecar
performance advantage, selected bounded general-purpose read units, and passed
exact Xarray, topology, concurrency, and recovery checks. Alternative A remains
the bounded-cost fallback only if the approved-destination preflight makes a
complete rewrite operationally infeasible; it may also serve as an explicitly
temporary training cache while the durable sharded rewrite is produced.

Within alternative B, prefer an Icechunk-native repository over an ordinary
Zarr directory. The required Plan A Xarray gate passed; Plan B Rust integration
is useful but is not a publication prerequisite. Icechunk supplies pinned
snapshots, transactional publication, virtual-to-native migration, and rollback
that a 2.7 PB rewrite particularly needs. A plain sharded Zarr v3 directory is
the portability fallback: build it under a private staging path, validate a
completion manifest, make it immutable, and publish it through an atomic
pointer. Do not maintain both as full physical copies. If needed, an Icechunk
catalog can virtually reference the immutable plain store.

## Proposed target representation

Keep one root Xarray-compatible dataset with source variable names and source
scientific semantics. Use Icechunk snapshots, branches, and tags for lifecycle
rather than encoding lifecycle into a DataTree. The cluster-wide archival
release should materialize all 67 inventoried source arrays so it is a
self-contained scientific archive rather than an eight-array Samudra cache. A
smaller publication is permitted only if it is explicitly named and documented
as a versioned subset, not as the LLC4320 archive replacement.

```text
/
├── 5 time-varying 3-D fields        native sharded Zarr v3
├── 17 time-varying surface fields   native sharded Zarr v3
├── 6 static 3-D masks/fractions      native sharded Zarr v3
├── 19 horizontal geometry fields     native sharded Zarr v3
└── 20 coordinates/reference fields   native Zarr v3
```

The bounded schema below is selected from the real-data tournaments. It remains
provisional only with respect to destination behavior: retune it if the
approved production filesystem behaves materially differently from scratch.

### Candidate 3-D layout

The Cody-branch review moved the leading physical shard from `720 x 720` to
`1440 x 1440`. An isolated two-time spatial tournament favored one-time
`2160 x 2160`, but the more representative constant-volume time/space
tournament selected two-time `1440 x 1440` while retaining the other envelopes
as controls:

```text
leading shard: (time=2, k=51, face=1, j=1440, i=1440)
inner chunk:   (time=1, k=17, face=1, j=120,  i=120)
```

`W` has 52 interface levels. Keep the 17-level logical depth unit, but use a
nominal physical depth of 68 so its four inner depth chunks occupy one shard.
Zarr truncates the edge to the 52-level array shape. The measured
`(logical=17, physical=68)` candidate avoided the otherwise universal
one-level second shard and outperformed 13- and 26-level logical controls on
the balanced depth-read score.

The outer-shard tournament is:

| Physical shard | Role | Shards touched by one `752²` tile | Shards touched by the `1472²` group |
| --- | --- | ---: | ---: |
| `(1, 51, 1, 720, 720)` | current-envelope control | 9 | 16 |
| `(1, 51, 1, 1440, 1440)` | medium-envelope control | 4 | 9 |
| `(1, 51, 1, 2160, 2160)` | wide-space control | 4 | 4 |
| `(2, 51, 1, 1440, 1440)` | selected candidate | 4 | 9 |

The corresponding physical object counts per 3-D variable are approximately
4.83 million, 1.21 million, 0.54 million, and 0.60 million. Do not combine
multi-time packing with `2160²` until writer memory and recovery behavior are
measured. The follow-up comparison held decoded shard volume roughly constant
across `(time, space) = (1, 2160), (2, 1440), (4, 1080), (8, 720)` using eight
real times. The selected `(2, 1440)` candidate was fastest for four independent
tiles, their union, a point time series, and a cropped time series. It also
reduced fixture object count from 34 for `(1, 2160)` to 18, with essentially
identical physical bytes.

The tournament tested spatial inner chunk sizes `s in {60, 72, 90, 120, 180}`.
Each divides 720 and 1440. With all 51 float32 levels their decoded sizes are
approximately 0.70, 1.01, 1.58, 2.80, and 6.31 MiB. The joint depth/spatial
tournament selected `k=17`, `s=120`, whose decoded logical chunk is about
0.93 MiB. It had the best equal-workload geometric-mean score among candidates
within 1.15 times the smallest measured physical size.

An inner `32 x 32` chunk is not valid inside a `720 x 720` shard because 32
does not divide 720. Very small valid choices such as 16 or 24 are also well
below 1 MiB when depth is not included, and would create hundreds or thousands
of inner chunks per spatial shard. Halo width and inner chunk width do not need
to be equal.

The selection deliberately rejects `k=51` as the logical chunk even though
current training loads every level: it would force a single-level scientific
selection to decompress all levels. Since 51 factors as `3 * 17`, the spike
tested depth chunks of `3`, `17`, and `51` jointly with spatial widths `72`,
`90`, and `120`. `k=17`, `s=120` cut the tested single-level crop from about
39 ms for comparable `k=51`, `s=120` to 18 ms, while its four-tile median was
1.77 seconds and its physical bytes were only 0.19% larger. Keep the physical
shard at all 51 levels so full-depth training still touches one depth envelope.

The Rust loader changes how these candidates should be measured, not the basic
sharding choice. `zarrs` can retrieve a requested array subset and decode only
the intersecting inner chunks, but PR #800 currently asks for full OM4 planes.
The LLC implementation must make tile, halo, face, and depth selections part of
the native read request. Otherwise the reader would erase the storage benefit
by loading a whole face and cropping afterward.

### Candidate surface layout

Surface variables need independent spatial reads without millions of files.
They were tuned independently because a scalar `72 x 72` chunk is only about
20 KiB decoded. The first tournament selected a `360 x 360` logical spatial
chunk from widths `180`, `240`, and `360`. A 24-time full-face fixture then
compared roughly constant-volume physical envelopes and selected
`(time=24, face=1, 1080, 1080)`:

```text
leading shard: (time=24, face=1, j=1080, i=1080)
inner chunk:   (time=1,  face=1, j=360,  i=360)
```

It balanced a one-time `752 x 752` tile, a full-face scan, a 24-time point, and
a 24-time crop. Its 24-time point was fastest, its tile and full-face reads
were within about 1% of the fastest candidates, and its cropped time series was
22% slower than the fastest candidate. It projects to approximately 89,000
physical objects per surface variable, compared with about 151,000 for
`(time=8, 1440, 1440)`.

The earlier shard-shape alternatives remain useful context:

1. `(1 time, 13 faces, 4320, 4320)`: preserves roughly one object per hour and
   matches the current source read unit, but creates a large shard and coarse
   write task.
2. `(1 time, 1 face, 4320, 4320)`: about 13 times as many objects, with more
   manageable independent writes.
3. `(1 time, 1 face, 720, 720)`: simple alignment with training tiles but about
   4.8 million physical objects per variable; likely unacceptable on NFS.

The selected surface layout need not match the 3-D spatial inner width.

### Compression and ordering

Keep float32 and make Blosc Zstd plus bitshuffle the baseline codec pipeline.
This matches the source codec family, is readable by Zarr Python, and is enabled
in the `zarrs` dependency used by PR #800. It is a Zarr v3 extension codec, so
the release gate must test every intended reader rather than relying only on
format conformance.

Benchmark at least these pipelines on exactly the same representative values:

| Pipeline | Purpose |
| --- | --- |
| Blosc Zstd + bitshuffle, levels 1, 3, and 5 | Primary candidates; likely strongest float32 size/throughput balance. |
| Blosc LZ4 + bitshuffle | Decode-throughput candidate if storage capacity permits. |
| Blosc LZ4 level 5 + byte shuffle | Exact control used by Cody's active float16 packed caches, tested here with float32. |
| Zarr v3 core Zstd, levels 1, 3, and 5 | Maximum core-codec portability control, without bitshuffle. |

Measure compressed bytes, encode throughput, single-thread and concurrent Rust
decode throughput, and tile-plus-halo latency. The local synthetic compatibility
fixture stored 74,278 bytes with Blosc Zstd/bitshuffle versus 550,171 bytes with
core Zstd at level 5; that ratio is deliberately not extrapolated to LLC4320.
Both decoded correctly through `zarrs` 0.21.2. PR #800 also notes that its
bundled C-Blosc has optimized x86-64 bitshuffle but a scalar ARM bitshuffle
path, so benchmark on the architecture that will actually load the data.

Use the sharding codec's CRC32C-protected index. Separately benchmark a payload
checksum if end-to-end digests and Icechunk integrity checks do not provide the
desired corruption detection. Test shard-index placement at the start and end
of the object; the local spike observed suffix reads with the default end
placement, followed by byte-range reads for the selected inner chunks.

Configure Icechunk manifest splitting before creating production arrays. Each
large array still has hundreds of thousands of physical shards; a single
manifest per array can produce slow startup, high memory use, and expensive
incremental commits. Split primarily along time and then evaluate face/tile
boundaries. The billions of logical inner chunks reside in shard indexes, not
as individual Icechunk manifest entries.

## Storage calculations

The 1.85 PB figure used in the halo discussion is logical decoded size, not a
compressed disk measurement.

For tile width `T` and halo width `h`:

```text
internal ring pixels = T^2 - (T - 2h)^2 = 4h(T - h)
external halo pixels = (T + 2h)^2 - T^2 = 4h(T + h)
```

For `T=720` and `h=32`:

- Cody's `4 * h * T = 92,160` approximation double-counts internal corners
  or omits external corners depending on interpretation.
- The exact internal ring is 88,064 pixels, 16.99% of a tile.
- The exact external halo is 96,256 pixels, 18.57% of a tile.
- Applied to 1.85 PB, the external halo is about 344 TB logical before land,
  dtype conversion, or compression.

If 40% of pixels are land, 60% remain, so float16 would give roughly 103 TB
before compression. The approximately 65 TB estimate instead assumes about
60% land, leaving 40%, and then halves float32 to float16. The land fraction
and compressed result must be measured by face, variable, level, and chunk
layout rather than assumed globally.

For `h=80`, the approximately 150 TB estimate is consistent with an internal
ring, 60% land, and float16. It is not the general storage requirement.

Sharding removes the separate halo allocation but requires rewriting the full
selected source. Output physical size cannot be derived from logical size; the
prototype must measure compression ratios on ocean, coastal, and land-heavy
chunks.

## Icechunk lifecycle and transaction design

### Baseline

1. Create a new Icechunk repository at the approved destination.
2. Configure absolute local-filesystem virtual chunk containers for both source
   mounts.
3. Virtually ingest the source schema and chunks without changing the source.
4. Record source paths, metadata hashes, chunk inventory method, code commit,
   dependency lock, and creation time.
5. Commit and tag this snapshot as the virtual source baseline.

Virtual ingestion itself may require listing or statting tens of millions of
source chunk files because the source is unconsolidated and virtual references
need locations and lengths. This metadata crawl must be measured and made
restartable. Dense chunk coordinates may be generated deterministically, but
missing chunks and physical lengths still require validation.

### Build branch

1. Create a private branch from the baseline.
2. Create final-shape native arrays using the chosen chunk, shard, codec, and
   manifest configuration.
3. Populate disjoint complete shards in bounded batches.
4. Commit progress only to the private branch.
5. Persist an explicit completion ledger keyed by array and shard coordinate.
6. Make retries idempotent: recompute or verify one entire shard, never perform
   an overlapping partial-shard write.

Intermediate branch snapshots are restart points, not valid datasets.

### Publication

1. Verify complete shard coverage and run all scientific validation.
2. Benchmark the immutable candidate snapshot with the Samudra loader.
3. Create an immutable release tag only after all gates pass.
4. Update consumers explicitly to that tag or snapshot; do not make readers
   follow a mutable build branch.
5. Retain the virtual baseline and source dependency until a separately
   approved retirement plan exists.

For a cluster-wide release, place repository data and its catalog/pointer in an
ORCD-approved shared namespace with inherited read and directory-traverse
permissions. Validate access from an unprivileged account that is not a member
of the build owner's private group. Publish a minimal `xr.open_zarr` example,
the pinned snapshot/tag, variable inventory, calendar/time conventions,
estimated read costs, and a support/ownership contact. Build credentials and
write permissions must not be required by readers.

### Commit coordination

Icechunk filesystem storage is not safe with concurrent committers. Parallel
workers may compute and write disjoint shards, but one coordinator must gather
their changes and advance the branch. Use Icechunk's distributed Xarray/Dask
write APIs, forked sessions, and a single merge/commit path.

Across separate Slurm jobs, serialize branch commits. Each job reopens the
latest build snapshot, claims a disjoint batch from the completion ledger,
writes it, validates it, and commits through one coordinator. Never allow
independent job-array elements to commit concurrently to the same filesystem
repository.

### Consumer contract

Every release must publish a machine-readable contract containing the Icechunk
snapshot ID and tag, Zarr metadata hashes, schema version, chunk/shard shapes,
codec JSON, expected arrays, source provenance, and compatible reader versions.
Consumers open a pinned snapshot, never a mutable branch. The native Samudra
loader should expose a storage-neutral `CanonicalReader`: one implementation
for plain filesystem Zarr and one for Icechunk. The latter should accept a
serialized read-only session or equivalent explicit repository/snapshot
configuration; it must not silently follow `main`.

If Icechunk session serialization remains the bridge across Python and Rust,
lockstep version compatibility is part of the data release. Prefer a native
Rust repository open from explicit storage configuration and snapshot ID when
that API is stable, since it removes Python session serialization from worker
startup.

The public contract also needs a stable discovery pointer. The pointer may move
only when a new immutable release has passed validation; notebooks and
production jobs should record the resolved snapshot ID for reproducibility.

## I/O-aware execution design

No compute strategy can exceed the aggregate bandwidth and metadata capacity
of the `002` and `003` NFS servers. The goal is to approach that ceiling without
causing contention for other users.

- Partition conversion work by existing source chunks and complete output
  shards, not by destination samples that reread neighbors.
- Read each source chunk once, decode once, and write every inner chunk for its
  destination shard in the same task.
- Sweep concurrency conservatively (for example 1, 2, 4, 8, then 16 readers)
  and stop increasing it when aggregate throughput flattens or tail latency
  worsens.
- Balance work involving `Theta`/`Salt` on `002` against arrays on `003` rather
  than assuming one shared bandwidth pool.
- Coordinate with ORCD/storage administrators before the production crawl and
  sustained rewrite. Ask for recommended reader counts, maintenance windows,
  quota, inode limits, and whether a storage-local or data-transfer facility is
  appropriate.
- Avoid Dask task graphs with millions of tasks at once. Generate bounded time
  or shard batches and checkpoint after each successful batch.

## Validation

### Structural

- Exact expected arrays, dimensions, shapes, dtypes, codecs, chunks, and shards.
- Expected count of populated shards and inner chunks.
- No missing shard coordinates except explicitly verified all-fill regions.
- Manifest sizes and split boundaries within configured limits.
- Coordinates and attributes preserved intentionally.

### Numerical

- Exact decoded equality to source for float32 output, including NaN placement
  and signed values.
- Full validation while each source chunk is already decoded, producing a
  source/output digest or equivalent per-shard evidence.
- Independent rereads of deterministic stratified samples: every variable,
  ocean/coast/land, shallow/deep, first/middle/last time, every face, face
  edges, and LLC corners.
- Separate validation for staggered `U`, `V`, `oceTAUX`, and `oceTAUY` axes.

### Failure recovery

- Kill a worker during a shard write and verify no published snapshot changes.
- Kill the coordinator before commit and verify the previous snapshot remains
  readable.
- Requeue/restart a job and verify completed shards are not rewritten unless
  verification fails.
- Simulate disk-full and corrupt/incomplete output detection on the prototype.

### Performance

- Cold-cache and warm-cache tile-plus-halo reads.
- Interior, face-edge, and face-corner tiles.
- Single reader and realistic concurrent training readers.
- Read bytes, request count, decompression CPU, wall time, throughput, and p95
  latency.
- End-to-end DataLoader throughput and GPU starvation time.
- Conversion throughput and projected full-run duration at the storage
  saturation point.
- Decode the same selections through Zarr Python, Samudra's pinned `zarrs`
  version, and Zarrista; compare values, bytes read, wall time, and CPU time.

## Completed local spikes

The reproducible spikes live under [`spikes/`](spikes/README.md).

A bounded, one-CPU Slurm inventory estimated 1.0317 PB decimal of compressed
payload for the complete source root and 755.2 TB for the eight pilot arrays.
It followed the `Theta` and `Salt` symlinks and avoided an unbounded
login-node metadata crawl.

A 1:10-scale Zarr 3.3.0 experiment compared a center tile plus halo:

| Representation | Physical objects | Objects touched | Read calls | Bytes returned |
| --- | ---: | ---: | ---: | ---: |
| Source-like monolithic chunks | 9 | 9 | 9 | 537,144 |
| Independent small chunks | 144 | 36 | 36 | 145,068 |
| Small chunks in tile-sized shards | 9 | 9 | 17 | 219,642 |

The sharded representation transferred about 59% fewer bytes than nine whole
chunks while keeping the low physical-object count. It used suffix reads for
shard indexes and byte-range reads for neighboring chunks. Its compressed size
was about 8% larger in this synthetic smooth, 40%-NaN example.

An Icechunk 2.2.0 spike successfully committed, snapshotted, reopened, and
partially read the Zarr v3 sharded array through both Zarr Python and Zarrista's
Rust-backed asynchronous Icechunk reader.

A second compatibility spike wrote equivalent Blosc Zstd/bitshuffle and core
Zstd sharded arrays with Zarr Python, then read identical subsets through
`zarrs` 0.21.2, the exact major/minor dependency used by PR #800. Both decoded
correctly. For the deliberately smooth fixture, physical bytes were 74,278 and
550,171 respectively. These tests prove API and codec behavior only; local
temporary storage does not predict ORCD NFS performance.

## Completed ORCD spikes

A read-only Slurm extraction created an immutable real-data fixture in user
scratch: two times of `U` from `003`, two times of `Theta` from `002`, both at
all 51 levels over a source-aligned `2880 x 2880` region of face 1, and two
complete face-1 `Eta` planes. Its manifest includes exact source paths, sizes,
modification times, and fixture hashes. This gives the schema tournaments real
ocean, coast, land, two source mounts, and Cody's complete four-tile footprint;
it does not cover LLC seams or temporal compression behavior beyond two times.

The first real-data sharding smoke test stored one `U` time with `72 x 72`
logical spatial chunks in `1440 x 1440` shards using Blosc Zstd level 1 and
bitshuffle. It wrote 1.068 GB for 1.692 GB of decoded fixture values, a 0.631
physical-to-decoded ratio, and passed exact equality. Encoding took 5.07
seconds; warm reads of four independent Cody tiles and their union took 1.22
and 1.19 seconds. These timings establish that range-decoded sharding works on
ORCD scratch, not an expected production throughput.

The destination-ready halo control stored the four interior Cody tiles for two
times of `U` and `Theta`. A 16-cell external halo produced `752 x 752` tiles,
duplicated 9.09% of decoded values, occupied 1.031 GB, and read the four tiles
in a 1.82-second median. A 32-cell halo duplicated 18.57%, occupied 1.122 GB,
and read in 2.01 seconds. The selected sharded logical layout read the
equivalent current four-tile workload in 1.77 seconds without duplicated values
and also supports arbitrary selections. On this fixture, the halo sidecar has
no performance or storage advantage; retain it only as a full-rewrite fallback.

The selected volume schema was then materialized into a retained Icechunk pilot
containing eight real `Theta` times, all 51 levels, and a `2880 x 2880` region.
It encoded 13.54 GB decoded to 6.119 GB in 52.0 seconds. Xarray 2026.7.0 opened
the immutable Icechunk 2.2.0 snapshot, exact full-array equality passed, the
four-tile median was 1.45 seconds, the union median was 1.13 seconds, and an
eight-time point selection took 15 ms. This resolves the bounded Plan A
format/protocol question, but not all-face topology, the complete variable
inventory, public permissions, or production-destination concurrency.

The follow-up sparse all-face pilot materialized two times of the eight dynamic
arrays, all 13 faces, four `720 x 720` corner regions per face, masks,
horizontal geometry, and vertical coordinates. Pinned Xarray opened snapshot
`XEDFKS47N3C85MB1RTR0`; all 52 populated face corners matched their sources
exactly, and staggered dimensions were preserved. The archive occupied 17.715
GB in 648 physical objects, encoded in 382 seconds, and peaked at about 40 GiB
RSS. This passes the broad dataset-shape and Xarray gates while deliberately
leaving cross-face rotation to an independent topology oracle.

That independent xgcm oracle applied the published LLC/ECCO 13-face connection
table to scalar and staggered-vector differences on source and pilot. Exact
equality passed for 23,163 populated X-boundary scalar results, 23,254
Y-boundary scalar results, 23,156 X-vector results, and 23,240 Y-vector
results. The selected representation therefore preserves face rotations and
vector-component exchange for the tested boundaries; Cody's same-face
training geometry was not used as the topology oracle.

A bounded 1/2/4/8/16-reader sweep compared source `U` on `003`, source `Theta`
through its `002` symlink, and the sharded Icechunk pilot on scratch. Cache state
was uncontrolled, so the absolute and apparent first-pass throughput are not
cold-cache claims. Sixteen readers produced the highest aggregate decoded
throughput, but its p95 latency increased by about 51%, 55%, and 25% over eight
for those three cases. Use eight outer readers and Zarr async concurrency one as
the conservative prototype starting point. The approved production destination
must repeat the sweep with disjoint coordinates before setting conversion or
public-reader guidance.

The Icechunk recovery spike also passed on ORCD with Icechunk 2.2.0: a killed
uncommitted write was invisible, the previous snapshot remained immutable, an
idempotent retry committed, and a stale concurrent session failed with
`ConflictError`. The filesystem backend warned that concurrent committers are
unsafe. The single-committer design is therefore a decision; multi-worker
fork/merge was then tested with eight serializable worker sessions. A simulated
capacity failure produced no worker result and remained invisible, retry and
single-coordinator merge succeeded exactly, time splitting produced four
manifests, and garbage collection removed the orphan chunk while the release
snapshot remained readable. A subsequent durable-ledger job survived a real
Slurm requeue, rejected duplicate claims, recovered a stale claim, verified
artifact hashes, kept a dead coordinator's uncommitted state invisible, and
reconstructed one exact release snapshot. The single-committer/fork-merge
protocol, time-oriented manifest splitting, and durable completion ledger are
therefore selected.

## Remaining release gates

The real-data schema tournaments, halo comparison, recovery injection,
completion-ledger requeue, concurrency sweep, and sparse all-face Xarray pilot
are complete. Do not begin the full rewrite until the following remaining
gates pass:

1. **Approved-destination preflight:** Repeat disjoint-coordinate concurrency
   and write tests on the actual public destination. Confirm byte and inode
   quota, group-readable permissions, directory traversal, backup policy,
   expected object count, and projected conversion duration. Scratch results
   cannot answer these operational questions.
2. **Native Rust reader check (non-blocking):** Extend the PR #800 reader seam
   with LLC subset requests and an Icechunk-backed implementation. Read the
   same pilot using persistent readers, bounded prefetch, and the pinned
   `zarrs`/Icechunk versions. This establishes the Plan B path but does not
   replace the Xarray acceptance gate.

The approved-destination work must run through Slurm and must not write beside
or beneath the source dataset.

## Open questions

1. Which filesystem or object store will hold the Icechunk repository, and what
   are its byte and inode quotas?
2. Must the original store remain independently available indefinitely?
3. What tile size survives Cody's physics, stability, and speed experiments?
4. What halo widths and depth subsets must the storage serve efficiently?
5. What loader throughput and GPU-utilization threshold defines success?
6. What maximum physical storage growth and conversion duration are acceptable?
7. Does ORCD NFS deliver efficient concurrent byte-range reads within large
   shard files?
8. Who owns publication, tag promotion, rollback, retention, and garbage
   collection decisions?
9. Is Icechunk a required storage protocol for the first Rust loader release,
    or may a validated plain Zarr v3 publication serve as a temporary bridge?

## References

- [ORCD getting started](https://orcd-docs.mit.edu/getting-started/)
- [ORCD scheduler overview](https://orcd-docs.mit.edu/running-jobs/overview/)
- [Icechunk concepts](https://icechunk.io/en/stable/understanding/concepts/)
- [Icechunk virtual datasets](https://icechunk.io/en/stable/guides/virtual/)
- [Icechunk Zarr virtualization](https://icechunk.io/en/stable/guides/zarr/)
- [Icechunk distributed writes](https://icechunk.io/en/stable/guides/dask/)
- [Icechunk filesystem limitations](https://icechunk.io/en/stable/guides/storage/#filesystem-storage)
- [Icechunk performance and manifest splitting](https://icechunk.io/en/stable/guides/performance/)
- [Zarr sharding](https://zarr.readthedocs.io/en/latest/user-guide/performance/#sharding)
- [Cody's LLC experiment branch](https://github.com/m2lines/Samudra/tree/llc_cpu_working)
- [Reviewed branch commit](https://github.com/m2lines/Samudra/tree/aa336ee7a17ecf8c8ad127081e5b1fa4519cbcbd)
- [Latest tracked four-tile launch parent](https://github.com/m2lines/Samudra/tree/ed7ab891beac51d198b15487aa47fc719de274a6)
- [Samudra Rust-loader PR #800](https://github.com/m2lines/Samudra/pull/800)
- [Zarrista](https://developmentseed.org/zarrista/latest/)
- [`zarrs_icechunk`](https://github.com/zarrs/zarrs_icechunk)
