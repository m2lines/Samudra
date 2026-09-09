<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC4320 General Archive and Remote-Streaming Experiment

Status: draft for review

## Summary

Empire AI provides enough GPU compute for large LLC4320 experiments but this
proposal assumes that our allocation can retain only 100 TiB, far less than
the 2.732 PB decoded LLC4320 archive. Copying the complete archive to every
training system is therefore not viable.

Build one generally useful, immutable Zarr v3 repository on MIT-managed
storage. Co-design its physical shards with the current 720-cell training
patches without encoding halos, overlapping samples, or a four-tile sampler
into the archive. Expose an experimental subset through an operator-approved
HTTPS or S3-compatible endpoint. Empire AI workers will read complete
shards or only the internal logical chunks needed for a patch and its halo.
Node caching and replay reuse are optimizations to measure, not assumptions
required to make the base layout viable.

This is not a proposal to serve `/orcd/data/abodner/003` directly, nor to make
an ad hoc web server on an ORCD login or Slurm compute node. The origin must be
a separately materialized, read-only repository behind an approved data
service. The production source remains immutable and private to ORCD.

The v2 experiments remain the local-filesystem baseline, but their winning
`time=2, 1440 x 1440` candidate changed time and space together and therefore
did not independently establish either choice for WAN access. V3 adopts this
working decision for time-varying 3-D fields:

```text
logical chunk:  (time=1, depth=17, face=1, j=120, i=120)
physical shard: (time=1, depth=51, face=1, j=720, i=720)
codec:          Blosc Zstd level 5 + bitshuffle
```

`time=1` makes every state independently addressable and removes pair-boundary
effects from adjacent-time training. A 720-square shard corresponds to one
model core, is a useful cache and recovery unit, and does not assume that every
consumer uses Cody's 2-by-2 patch group. The smaller internal chunks preserve
efficient depth slices, crops, and halo range reads. The prototype will try to
falsify this decision with `60 x 60` logical chunks and `1440 x 1440` physical
shards as the principal controls; it will not rerun a broad unconstrained
layout search.

The first goal is not a large training run. It is a small remote corpus and a
staged benchmark that tells us whether one GPU node, and then several nodes,
can keep GPUs busy without exceeding an agreed MIT egress budget. A model
training launch remains a separate approval step.

## Decision requested

Approve engineering of a bounded remote-streaming prototype, conditional on:

1. ORCD approving the storage location and the read-only HTTPS or
   S3-compatible serving path;
2. Empire AI confirming outbound HTTPS/S3 access from compute nodes;
3. an agreed MIT egress and concurrency ceiling; and
4. tens of GiB of stable MIT-side prototype capacity for the first fixture,
   expandable only after the range-read gate passes.

Do not authorize a full LLC4320 rewrite from this document.

## Goals

- Determine whether remote Zarr reads can hide beneath Samudra compute at
  useful GPU scale.
- Make the experiment representative of Cody's four overlapping spatial
  patches and replay-buffer behavior.
- Minimize bytes crossing the WAN and avoid duplicate requests across GPUs on
  one node.
- Preserve an ordinary Xarray/Zarr view for inspection, analysis, and training.
- Use immutable version identifiers so node caches are safe and reproducible.
- Produce enough evidence to decide between remote streaming, rolling staging,
  and a larger remote storage allocation.

## Non-goals

- Completing the full 2.7-PB conversion before remote access is validated.
- Publishing all 67 LLC4320 arrays in the first remote prototype.
- Serving the source NFS tree over the public internet.
- Assuming that internal InfiniBand or NVLink performance predicts WAN
  performance.
- Requiring Samudra's experimental Rust loader for the first result.
  TensorStore is the performance acceptance path, while Xarray/Zarr remains
  the labeled correctness path.
- Claiming GPUDirect Storage. Current Zarr Python GPU buffers can place final
  arrays in device memory, but reads and codec execution still pass through
  host memory.

## Background

### Source and training workload

The source is the read-only, unconsolidated Zarr v2 hierarchy at:

```text
/orcd/data/abodner/003/LLC4320/LLC4320
```

The metadata inventory found 10,311 hourly times, 13 faces, `4320 x 4320`
horizontal cells, 51 center levels, 67 arrays, and 2.731912 PB decoded. The
source occupies an estimated 1.0317 PB of compressed payload in approximately
24.32 million files. `Theta` and `Salt` resolve through symlinks to the `002`
filesystem; the other 65 arrays resolve to `003`.

Cody's current experiment uses four `752 x 752` same-face tiles arranged as a
2-by-2 group. Each tile is a 720-cell base plus 16 cells on each side, so
adjacent predictions overlap by 32 cells. The group union is `1472 x 1472`.
The overlap is predicted and blended; it is not a context-only halo.

The core fields are `U`, `V`, `Theta`, and `Salt` at all 51 levels, with `Eta`
and surface forcing. An ordinary one-step transition needs two prognostic times
and current forcing. Replay requests and seed refreshes have different miss
patterns, so the benchmark must replay the observed request trace rather than
assume that every optimizer step reads two new states.

### Storage constraint

This proposal accepts 100 TiB as a hard remote-cluster storage budget. That is
large enough for a cache or a selected training window, but not a complete
archive. Based on the measured 755.2 TB compressed payload of the eight
Samudra-oriented arrays, 100 TiB is roughly equivalent to 62 days of the whole
globe or, ignoring face-to-face compression differences, approximately one to
two complete-face histories. This is useful capacity, but it must be managed as
a versioned working set rather than treated as the source of record.

Public Empire AI information describes storage accounting through 100 TB and
separate management above that threshold. It also documents substantial
internal storage and network hardware, but neither establishes available WAN
bandwidth from an allocated Empire AI compute node to MIT. The actual
MIT-to-Empire-AI path must be measured from scheduled nodes; login-node or
data-transfer-node specifications are not substitutes.

### Relevant remote-Zarr precedents

Remote Zarr is an established pattern, especially when compute and object
storage are in the same cloud region. WeatherBench 2 publishes cloud-optimized
scientific datasets in Zarr on Google Cloud Storage. The WIND training project,
for example, lazily loads those arrays through Xarray and `gcsfs` and supports
training without first downloading the complete dataset; its documentation also
recommends colocating compute in `us-central1` for higher throughput.

Those examples establish software feasibility, not LLC feasibility. LLC
examples are much larger per sample, our proposed origin and GPUs cross
institutional networks, and Cody's four-tile transition can require gigabytes
before cache reuse. The relevant question is therefore measured end-to-end
utilization, not whether Xarray can open a remote URL.

### What v2 established

The ORCD experiments made the following durable findings:

| Question | V2 result | V3 implication |
| --- | --- | --- |
| Codec | Blosc Zstd level 5 plus bitshuffle was smallest of seven real-data candidates and approximately 2.3% smaller than matching source objects. | Use it as the first WAN candidate; retain LZ4 as a CPU-latency control. |
| General logical volume chunk | `(time=1, k=17, face=1, 120, 120)` balanced full-depth training and arbitrary depth crops. | Keep as the scientific-access control, but test full-depth chunks because WAN request latency is more expensive. |
| General physical volume shard | `(time=2, k=51, face=1, 1440, 1440)` won a constant-volume local tournament in which time and spatial extent changed together. | Treat it as useful evidence, not an isolated result. V3 defaults independently to `time=1, 720 x 720` and keeps 1440 as a control. |
| Surface layout | Logical `(1,1,360,360)` in physical `(24,1,1080,1080)` was the best local balance. | Retain as a later control. Surface object count and time-series access require their own decision; do not silently apply the 3-D volume layout. |
| Halo sidecar | A 16-cell sidecar duplicated 9.09% and was slower than the general sharded layout; 32 cells duplicated 18.57%. | Do not precompute halos for the canonical remote arrays. |
| Correctness | Pinned Xarray reads, all-face corners, staggered dimensions, and an independent xgcm topology oracle passed exactly. | Reuse the schema and validators; remote transport must not change scientific semantics. |
| Recovery | Fork/merge, failure retry, garbage collection, a persistent ledger, and a real Slurm requeue passed. | Build the remote corpus transactionally and publish only an immutable snapshot. |
| Concurrency | Eight outer readers was the conservative ORCD starting point; 16 increased tail latency. | Do not transfer this number to WAN clients. Measure concurrency independently at each origin/destination pair. |

### V3 focused layout and native-reader tournament

On 2026-09-08, candidates A--E were encoded from the same real LLC `Theta`
fixture: 8 times, all 51 levels, and a `2880 x 2880` region. The fixture is
13,536,460,800 decoded bytes. Every candidate passed full-array exact equality
with Zarr Python 3.3.0, zarrs-python 0.2.3, and TensorStore 0.1.85.

| ID | Logical / physical distinction | Stored GiB | Physical objects | Encode seconds |
| --- | --- | ---: | ---: | ---: |
| A | `17 x 120²` logical in `time=1, 720²` physical | 5.699 | 128 | 90.6 |
| B | `17 x 60²` logical in `time=1, 720²` physical | 5.672 | 128 | 91.0 |
| C | `51 x 120²` logical in `time=1, 720²` physical | 5.652 | 128 | 66.4 |
| D | `17 x 120²` logical in `time=1, 1440²` physical | 5.699 | 32 | 64.2 |
| E | `17 x 120²` logical in `time=2, 720²` physical | 5.699 | 64 | 36.3 |

Logical boundaries affected compressed size by less than 1%. Physical-only
changes A, D, and E produced effectively identical compressed bytes. Larger
physical shards reduced object count and made this single-process bulk encoder
faster, but conversion throughput is not sufficient reason to optimize the
published access layout.

For candidate A, median local read times were:

| Reader | Aligned 720² | 752² halo | 1472² union | Four separate 752² tiles |
| --- | ---: | ---: | ---: | ---: |
| Zarr Python | 0.246 s | 0.278 s | 1.118 s | 1.026 s |
| zarrs-python | 0.207 s | 0.301 s | 1.130 s | 1.026 s |
| TensorStore | 0.189 s | 0.216 s | 0.736 s | 0.767 s |

Across eight non-point representative workloads, TensorStore was approximately
28% faster than Zarr Python for A by geometric mean. zarrs-python was
approximately equal overall: it improved the aligned core but not the larger
halo and union selections. This makes TensorStore the current performance
reader and zarrs-python the Xarray-compatible native control.

Under TensorStore, D was approximately 7--10% faster than A for the aligned
core, halo, and four-separate-tile reads, but only 2% faster for the union. E
was approximately 7--14% faster for the aligned core, halos, and separate-tile
reads, 4% faster for the union, and only 1% faster for an aligned adjacent-time
pair. These are local, warm-filesystem measurements; they do not measure WAN
range latency, transferred bytes, or cache churn.

Decision: retain A as the general-archive prior. B's smaller logical chunks
saved only 0.47% of stored bytes and had reader-dependent performance. C helped
some full-depth workloads but penalized shallow and point access. D and E stay
as the two required WAN controls because native local performance shows that
they are credible, not because they have displaced `time=1, 720²`. The WAN
HTTP/S3 trace is the deciding experiment for physical shard size and time
grouping.

The complete trial data and raw JSON are in
[`spikes/remote_streaming/results/2026-09-08`](spikes/remote_streaming/results/2026-09-08/README.md).

## Feasibility model

### Per-example lower bound

One float32 state for the four prognostic variables over the `1472 x 1472`
union is:

```text
4 variables * 51 levels * 1472 * 1472 cells * 4 bytes
= 1.647 GiB decoded
```

The four separate `752 x 752` tiles contain 1.719 GiB because they duplicate
their overlaps. A normal transition requires approximately two such states.
Applying the measured 0.526 compressed-to-decoded ratio gives a best-case
payload near 1.8 GiB per transition before surface fields, chunk
over-read, metadata, retries, or TLS overhead.

This is an intentionally simple lower bound, not a throughput prediction. If
each DDP rank consumes a unique transition and receives no cache hits, the
approximate origin demand is:

```text
required origin throughput
  = ranks * remote-miss bytes per step / step duration
```

Illustratively, using 1.8 GiB per miss:

| Independent ranks | 1-second steps | 2-second steps | 5-second steps |
| ---: | ---: | ---: | ---: |
| 1 | about 15 Gb/s | about 7.5 Gb/s | about 3 Gb/s |
| 8 | about 120 Gb/s | about 60 Gb/s | about 24 Gb/s |
| 32 | about 480 Gb/s | about 240 Gb/s | about 96 Gb/s |

Float16 can roughly halve decoded bytes, but its compressed ratio must be
measured rather than assumed. Replay reuse, node-level request sharing, and
caching can reduce origin misses by much more than dtype conversion. These
numbers make cache behavior part of the architecture, not an optional tuning
step.

### Latency and request count

The selected `17 x 120 x 120` logical volume chunk is approximately 0.93 MiB
decoded and about half that size compressed on the fixture. A full-depth
selection addresses three depth chunks at each spatial position. These chunks
are stored inside a much larger shard: clients may retrieve a complete aligned
720-square shard, or coalesce only the byte ranges intersecting a crop or halo.
The number of HTTP requests is therefore a reader and coalescing question, not
necessarily one request per logical chunk.

Zarr Python 3.3 added automatic coalescing of nearby ranges in a shard. Its
defaults merge gaps up to 1 MiB while capping a merged request at 16 MiB. This
can exchange modest read amplification for fewer WAN round trips, but the
effect depends on codec ordering, requested variables, and the remote server's
range implementation. The prototype must record actual GETs and bytes rather
than infer them only from array shapes.

## Proposed architecture

```text
immutable LLC4320 source on ORCD
              |
              | bounded Slurm conversion
              v
private build branch on MIT origin storage
              |
              | exact validation + immutable release
              v
approved read-only HTTPS/S3 endpoint
              |
              | authenticated byte-range GETs over WAN
              v
optional compressed-range/shard cache per GPU node
              |
              v
node-level fetch coordinator and prefetch queue
              |
              v
CPU decode -> pinned host tensors -> asynchronous GPU copy
              |
              v
four-tile model step and replay buffer
```

### MIT origin

The preferred origin is an S3-compatible object store approved by ORCD, or an
Open Storage Network allocation reachable through HTTPS. OSN provides the S3
access model, protected read-only credentials, and open-access buckets. For
this experiment use protected read-only credentials and TLS.

Icechunk supports S3-compatible repositories and read-only repositories served
as static files over HTTP. If ORCD approves an HTTPS static-serving path, the
MIT filesystem repository can be opened with Icechunk's HTTP storage backend.
Do not assume that a normal ORCD filesystem path is remotely readable, and do
not run a persistent server on a login node.

For the immutable experiment, either of these representations is acceptable:

1. an Icechunk repository on S3-compatible storage, opened at an explicit
   snapshot; or
2. an Icechunk filesystem repository published as read-only static HTTPS
   content, again opened at an explicit snapshot.

If neither service path can be approved, stop the streaming experiment and use
the rolling-staging fallback. Globus is appropriate for bulk staging, not
random Zarr reads in the training hot path.

### Remote caches

Start with direct reads and add each cache layer separately so its benefit is
observable:

- **L1 decoded/prefetch cache:** process memory holding ready tensors for the
  next few steps and replay entries.
- **L2 node cache:** compressed ranges or complete immutable shard objects on
  node-local NVMe. All GPU ranks on the node share this cache through one fetch
  coordinator or a lock-safe content-addressed directory.
- **L3 shared working set:** optional shards staged in the remote cluster's
  100 TiB allocation. Populate it explicitly or on demand, with eviction keyed
  by snapshot and object identifier.

Never let eight ranks on one node independently fetch the same shard. Deduplicate
in-flight requests. For Cody's grouped sampler, concurrently fetch the four
aligned 720-square core shards plus the required ranges from their neighbors,
assemble the `1472 x 1472` union, and scatter or create views after decode.

Immutable snapshot IDs must be part of every cache key. A release change uses
a new namespace; it never mutates cached bytes in place.

### Reader path

Use TensorStore as the performance path for the remote benchmark. It natively
supports Zarr v3, `sharding_indexed`, Blosc, and concurrent I/O, and it was the
fastest reader in the focused local tournament. Use Xarray with the
zarrs-python Rust codec pipeline as the labeled-array compatibility path, and
plain Zarr Python as the independent correctness baseline. Zarrista remains a
promising direct zarrs API for explicit subchunk and shard-index-cache control;
add it after the TensorStore HTTP result if that control is needed.

Open the repository and snapshot once per long-lived worker, preload bounded
coordinate and relevant time-manifest metadata, and retain persistent HTTP
connections. Do not reopen the repository per sample.

Start with one outer data-fetch pipeline per GPU node, not one unconstrained
pipeline per rank. Bound:

- in-flight samples;
- total HTTP requests;
- bytes resident in compressed and decoded queues;
- codec worker threads; and
- host-to-device copies.

Zarr's async concurrency and the DataLoader worker count multiply. Sweep them
as a joint configuration and enforce a node-wide semaphore.

Samudra's experimental Rust loader should consume the same immutable snapshot
and cache protocol after the standard native readers establish feasibility. It
is not a prerequisite for creating or publishing the experiment corpus.

## Prototype fixture

### Scope

Build a small spatial-and-temporal subset using exactly the schema, dtype, and
layout proposed for the general archive, for example
`llc4320-general-streaming-fixture-v1`. It is a test fixture, not a
training-specific representation and not a commitment to convert the full
archive.

Include:

- `U`, `V`, `Theta`, and `Salt`, all 51 levels;
- `Eta`, `oceQnet`, `oceTAUX`, and `oceTAUY`;
- required masks, coordinates, and normalization metadata;
- representative regions containing ocean, coast, land, and at least one
  aligned 2-by-2 group of 720-square cores with its exterior halo; and
- consecutive times sufficient to exercise ordinary transitions, shard-time
  boundaries, seed refreshes, and replay; plus
- a two-time sparse seam/corner canary across all 13 faces, reusing the v2
  all-face regions and topology oracle.

Begin with 8 consecutive hours, four complete `720 x 720` core shards, and the
neighboring physical shards containing the logical chunks needed for 16- and
32-cell exterior halos. Add the small all-face seam/corner canary separately.
This should produce a fixture in the tens-of-GiB range while exercising
adjacent times, exact shard alignment, halo range reads, the four-patch union,
and partial-shard access. Expand to 96 hours only after the layout and
transport measurements justify it.

Encode candidates from identical source values and retain measurements and
manifests. Candidate stores are disposable; the winning fixture is immutable.

### Canonical representation and fallback control

The decision candidate retains canonical source variable names, dimensions,
and float32 values. This directly exercises the future general archive and is
the representation that must pass Xarray correctness and usability tests.

An explicitly lossy, packed float16 representation may be measured later as a
fallback control if the canonical archive fails the bandwidth gate. It is not
part of the first fixture or the proposed public archive. Any future derivative
must have a distinct name and record its source snapshot, channel order,
casting rule, masks, codec, and generator commit.

Do not duplicate the four overlapping tiles in the retained origin. Preserve a
spatial field and select the union at read time. A sample-chunked or duplicated
tile store may be included as a performance control, but it should win only if
the measured utilization improvement justifies its fixed geometry and storage
amplification.

## Candidate remote layouts

### Volume arrays: working decision

Use this layout for the first canonical fixture:

```text
logical chunk:  (time=1, depth=17, face=1, j=120, i=120)
physical shard: (time=1, depth=51, face=1, j=720, i=720)
codec:          Blosc Zstd level 5 + bitshuffle
```

One float32 3-D variable in a complete physical shard is approximately 101 MiB
decoded. At the fixture's measured compression ratio it is approximately 53
MiB, subject to variable, depth, and land-mask content. That is large enough to
amortize WAN setup while remaining a practical unit for concurrency, caching,
retry, and general spatial access.

Run a focused matrix that varies one design dimension at a time:

| Question | Candidate | Control | What decides |
| --- | --- | --- | --- |
| Spatial logical granularity | `120 x 120` | `60 x 60` | Halo/crop transferred bytes versus request, index, and decode overhead. |
| Depth logical granularity | `17` | `51` | General depth-slice cost versus full-depth ready-tensor latency. |
| Spatial physical envelope | `720 x 720` | `1440 x 1440` | General access, cache/retry size, and concurrency versus four-tile throughput. |
| Physical time extent | `1` | `2`, then `4` only if justified | Adjacent-transition bytes, pair-boundary effects, cache reuse, and recovery size. |

Physical `time=1` is the default, not merely one tournament entry. With
two-time shards aligned as `[0,1]`, `[2,3]`, and so on, half of ordinary
adjacent transitions cross two physical objects while half reside in one;
random/replay access also makes the benefit workload-dependent. Grouping time
is accepted only if an isolated experiment demonstrates a material gain.

A 720-aligned core plus a 16- or 32-cell context region reaches neighboring
physical shards. Small internal chunks bound the bytes read from those
neighbors, although they do not eliminate topology-aware selection or object
opens. Conversely, a four-core group with an exterior halo can touch more
720-square objects than 1440-square objects. The trace must therefore report
both requested object count and returned bytes.

Also test Zarr range coalescing with explicit gap and maximum-read budgets,
including the shipped defaults, rather than treating it as an implementation
detail.

### Surface arrays

Surface arrays need fewer bytes but can create many small physical objects.
Start the fixture with logical `(time=1, face=1, j=120, i=120)` inside physical
`(time=1, face=1, j=720, i=720)` shards so the first end-to-end trace has simple
state alignment. This is a fixture default, not yet the full-archive decision.
After the volume path is understood, compare it with the v2 surface result:

```text
logical:  (time=1, face=1, j=360, i=360)
physical: (time=24, face=1, j=1080, i=1080)
```

Add a `720 x 720` logical spatial control. Surface data is a small fraction of
the prognostic payload, so do not optimize it at the expense of more important
volume experiments.

### Packed float16 fallback control

For the packed `205`-channel prognostic array, test full-channel logical chunks
with spatial widths 120 and 240:

```text
logical candidates: (time=1, face=1, channel=205, j={120|240}, i={120|240})
physical controls:  (time=1, face=1, channel=205, j={720|1440}, i={720|1440})
```

At float16, the 120 and 240 logical candidates decode to approximately 5.6 and
22.5 MiB. A `1440 x 1440` packed shard decodes to about 1.58 GiB, so compare it
with the 720 envelope for memory, recovery, and cache behavior. Range reads
should avoid loading the complete shard, but the experiment must verify that
the actual client and endpoint do so.

### Codecs and dtype

Test these combinations on identical values:

- float32, Blosc Zstd level 5 plus bitshuffle;
- float32, Blosc LZ4 level 5 plus byte shuffle, matching Cody's codec family;
- float16, Blosc Zstd level 5 plus bitshuffle; and
- float16, Blosc LZ4 level 5 plus byte shuffle.

Measure stored and transferred bytes separately. The best local codec may not
be the best remote codec: Zstd can save WAN bandwidth while LZ4 can save CPU
time. Record decode CPU, not merely wall time.

## Spike plan

The spikes form this dependency graph:

```text
endpoint and policy preflight
        |
        +--> WAN range microbenchmark on Empire AI CPU node
                                  |
                                  +--> real Zarr range trace
                                               |
                                               +--> CPU ready-tensor benchmark

immutable source corpus
        |
        +--> focused logical/physical layout and codec tournament
                        |
                        +--> publish winning 8-hour fixture
                                      |
                                      +------> real Zarr range trace
                                                   |
                                                   +--> optional cache benchmark
                                                                |
                                                                +--> one GPU-node loader benchmark
                                                                             |
                                                                             +--> 2/4/8-node scaling
                                                                                          |
                                                                                          +--> short training run
```

### Work that can start before the production endpoint exists

Run these bounded tracks in parallel:

1. **MIT layout fixture:** build the `time=1, 720 x 720` physical candidate
   and its focused controls from identical real LLC values. Measure encoded
   bytes, object and index counts, encode/decode CPU, and exact equality.
2. **MIT access-amplification trace:** serve the fixture locally within one
   scheduled job and record the actual ranges produced by aligned `720 x 720`,
   halo-bearing `752 x 752`, grouped `1472 x 1472`, shallow-depth, point-series,
   and deliberately misaligned selections. This validates the Zarr plan and
   instrumentation without claiming WAN performance.
3. **Empire AI CPU preflight:** from a scheduled CPU node, record filesystem
   capacity, Python/runtime availability, DNS/proxy behavior, and outbound
   HTTPS support. Run the client against a harmless public range-capable object
   to validate measurement code, not to estimate MIT throughput.
4. **Empire AI staged decode control:** bulk-copy a few immutable candidate
   objects using an approved transfer path, then benchmark range planning,
   decompression, Xarray selection, normalization, and tensor construction
   locally. This separates codec/CPU limits from the future WAN result.

A short-lived authenticated server on an MIT Slurm node may be used for a
bounded reachability diagnostic only. ORCD compute addresses may be private,
and either failure or success is non-representative of an approved production
service. Do not tunnel through either login node and do not report such a test
as attainable training throughput.

### Spike 0: endpoint and policy preflight

Resolve before writing the corpus:

- approved MIT storage and serving service;
- authentication method and credential lifetime;
- TLS and byte-range support;
- whether `HEAD`, conditional GET, and concurrent range GET are supported;
- outbound access and proxy requirements on Empire AI compute nodes;
- MIT egress cap and acceptable experiment windows;
- origin monitoring and abuse/rate controls; and
- whether the remote shared filesystem may be used as an automatic cache.

This spike is blocked on administrator input. An SSH tunnel or login-node web
server is not an acceptable substitute.

### Spike 1: WAN range microbenchmark

Write deterministic objects at 512 KiB, 2 MiB, 8 MiB, 16 MiB, 64 MiB, and
approximately 512 MiB. From an allocated compute node, measure:

- DNS, connect, TLS, and first-byte latency;
- sequential and random aligned ranges;
- 1/2/4/8/16/32/64 concurrent requests;
- aggregate throughput and p50/p95/p99 latency;
- retries, throttling, and connection reuse;
- complete-object versus range transfer bytes; and
- at least a two-hour stability run at the proposed egress ceiling.

Run from an Empire AI CPU node first and repeat from the intended GPU-node
partition before the GPU loader gate. Use unique ranges for cold-like passes
and immediate repeats for warm passes; do not claim control of shared server
or kernel caches.

### Spike 2: remote layout tournament

Create every candidate from one immutable real-data corpus. Instrument the
store below Xarray/Zarr so each trial records object key, requested byte range,
returned bytes, status, and duration.

Replay these workloads:

- four independent `752 x 752` tiles;
- one `1472 x 1472` union and in-memory scatter;
- two adjacent prognostic times plus current forcing;
- seed refreshes;
- Cody's measured replay request sequence;
- randomized times and spatial positions; and
- deliberate shard-boundary and face-edge patches.

Rank by end-to-end ready-tensor time and transferred bytes. Stored size and
object count are constraints, not the sole score.

The minimum focused matrix is:

| ID | Logical volume chunk | Physical volume shard | Role |
| --- | --- | --- | --- |
| A | `(1,17,1,120,120)` | `(1,51,1,720,720)` | Working decision. |
| B | `(1,17,1,60,60)` | `(1,51,1,720,720)` | Lower halo/crop amplification. |
| C | `(1,51,1,120,120)` | `(1,51,1,720,720)` | Lower full-depth range count. |
| D | `(1,17,1,120,120)` | `(1,51,1,1440,1440)` | Four-core-group physical control. |
| E | `(2,17,1,120,120)` | `(2,51,1,720,720)` | Isolated physical-time control. |

Candidate E exists to quantify the rejected prior, not to give `time=2` equal
standing. Add physical `time=4` only if E materially outperforms A and its
pair-boundary behavior is acceptable.

### Spike 3: cache semantics and failure

Implement a content-addressed cache keyed by release, object, offset, and
length, or cache complete shard objects when the measured access trace favors
that strategy. Verify:

- concurrent ranks deduplicate one in-flight miss;
- partial and interrupted files never become hits;
- checksums or validators detect corruption;
- eviction respects the configured byte ceiling;
- restart reuses valid immutable entries;
- changing the snapshot never returns stale bytes;
- origin timeout falls back to bounded retry without hanging a DDP rank; and
- all ranks fail noisily and coherently when the retry budget is exhausted.

Compare no cache, per-process cache, shared node cache, and optional shared
100-TiB cache. Report origin bytes per optimizer step for each.

### Spike 4: training-shaped loader benchmark

Run Samudra's real preprocessing, normalization, masks, channel construction,
pinning, and asynchronous device copies, but replace the model with controlled
sleep or matrix work at several step durations. This isolates whether I/O can
be hidden under plausible compute without spending a full training allocation.

Measure one GPU node first. Sweep DataLoader workers, Zarr async concurrency,
prefetch depth, cache size, and union-versus-four-read behavior. Capture CPU,
memory, network, disk, GPU copy-engine activity, ready-queue depth, and time
blocked awaiting a batch.

### Spike 5: scale and actual training

Only after the earlier gates pass:

1. scale the loader benchmark to 2, 4, and 8 nodes using disjoint sample
   schedules;
2. repeat with schedules that deliberately share shards to validate node and
   shared-cache reuse;
3. run a short Cody-equivalent training job; and
4. compare loss and step behavior with the current local-cache path.

Launching the actual model training job requires explicit approval.

## Acceptance criteria

### Correctness

- Float32 canonical reads equal the source exactly, including NaNs and masks.
- Float16 values equal the documented source-to-float16 cast bit-for-bit.
- Tile union and four independent selections produce identical owned regions.
- Time ordering, variable/level-to-channel mapping, staggered axes, and
  normalization metadata are exact.
- Only a pinned immutable release is readable by training jobs.

### Performance

For the one-node training-shaped benchmark:

- median GPU-ready batch delivery is faster than the target model step;
- p95 data-wait time is no more than 10% of step wall time after startup;
- GPU utilization is at least 90% during a sustained, representative window;
- the ready queue does not repeatedly drain;
- retries and failed requests are below the agreed operational threshold; and
- origin throughput remains below the approved MIT ceiling.

For 2/4/8-node scaling:

- aggregate training-shaped throughput is at least 80% of ideal linear scaling
  through the largest approved test;
- origin bytes grow with cache misses, not blindly with GPU count;
- p95 data wait remains within the one-node acceptance envelope; and
- no node or rank creates an unbounded number of concurrent requests.

Use measured model step time and miss rate to calculate the final bandwidth
gate. The illustrative 1.8-GiB figure must not become an acceptance constant.

### Operations

- The source store is never modified or directly exposed.
- The origin can enforce read-only credentials and an aggregate rate limit.
- Every request is attributable through service metrics without logging secret
  material.
- Cache and origin failures have bounded retries and deterministic DDP failure.
- The prototype, logs, and caches fit their declared quotas.

## Decision tree after the prototype

1. **Remote origin meets utilization and egress gates:** proceed incrementally
   with the general immutable archive and keep remote storage as an optional
   managed cache.
2. **Remote origin works only with high cache reuse:** retain the architecture,
   but schedule rolling working sets into the 100 TiB allocation ahead of each
   training phase. WAN reads refill the cache asynchronously rather than gate
   ordinary steps.
3. **WAN range latency or policy fails:** use Globus/DTNs to stage versioned
   training windows in bulk. Do not continue tuning Zarr chunks around a
   transport that cannot support random reads.
4. **Canonical arrays fail but packed float16 passes:** publish the packed store
   as a training derivative with explicit provenance; keep the canonical MIT
   archive separate.
5. **Neither representation sustains the target:** request more colocated
   storage, reduce the active data window, or change the training sampling
   strategy before scaling GPUs.

## Risks and mitigations

| Risk | Consequence | Mitigation |
| --- | --- | --- |
| WAN bandwidth scales with DDP ranks | GPU starvation and load on MIT | Node-level fetch coordination, cache-aware sampler, replay reuse, explicit egress cap. |
| Too many small ranges | Latency dominates bandwidth | Full-depth chunks, larger spatial controls, Zarr coalescing, persistent connections. |
| Too-coarse chunks | Excess bytes and cache churn | Measure transferred bytes and amplification at tile boundaries; retain smaller control. |
| Every rank fetches the same shard | Multiplicative waste | One shared node cache and in-flight request deduplication. |
| Remote service is not approved | Prototype cannot run safely | Make endpoint preflight the first hard gate; use rolling Globus staging as fallback. |
| Float16 projection is mistaken for archive data | Scientific misuse | Distinct name, schema, provenance, casting declaration, and lifecycle. |
| Replay benchmark is synthetic | Wrong bandwidth estimate | Capture and replay Cody's actual request trace and miss sequence. |
| HTTP server ignores ranges | Whole-shard transfers | Verify status/range bytes in Spike 1 and fail the endpoint gate. |
| Cache serves stale or partial data | Silent corruption | Immutable snapshot keys, atomic insertion, validators/checksums, failure injection. |
| CPU decode becomes bottleneck | Network appears slower than it is | Separate network, decode, preprocessing, and GPU-copy timings; compare Zstd and LZ4. |
| MIT origin disrupts other users | Operational harm | Administrator-approved service, rate limit, monitoring, scheduled scale tests. |

## Open questions

1. Which MIT service can provide sustained, authenticated HTTPS/S3 range reads
   without an ad hoc server?
2. What are the measured WAN paths and policy limits from Empire AI CPU and GPU
   compute nodes, rather than its login nodes or internal fabric?
3. What is Cody's current median and p95 model step time per GPU?
4. What fraction of optimizer steps trigger a new source-state read after
   replay and seed reuse?
5. Will DDP ranks intentionally share times/faces, allowing node-level shard
   reuse, or sample independently?
6. Is the 100 TiB allocation node-local, shared persistent storage, or both,
   and what are its purge and inode policies?
7. Is a float16 packed training derivative acceptable as the primary remote
   representation, with float32 retained only at MIT?
8. What MIT aggregate egress rate and experiment duration are acceptable?
9. Which Empire AI GPU type and partition is the first training target?

## References

- [Zarr remote storage backends](https://zarr.readthedocs.io/en/latest/user-guide/storage/)
- [Zarr sharded range-read coalescing](https://zarr.readthedocs.io/en/latest/user-guide/examples/sharding_coalescing/)
- [Zarr concurrency guidance](https://zarr.readthedocs.io/en/main/user-guide/performance/)
- [Zarr GPU buffers and current host-codec limitation](https://zarr.readthedocs.io/en/stable/user-guide/gpu/)
- [TensorStore Zarr v3 driver](https://google.github.io/tensorstore/driver/zarr3/)
- [zarrs-python Rust codec pipeline](https://github.com/zarrs/zarrs-python)
- [Zarrista shard-index cache](https://developmentseed.org/zarrista/latest/api/shard-cache/)
- [Icechunk storage backends, including S3 and read-only HTTP](https://icechunk.io/en/stable/storage/)
- [Icechunk performance and manifest preloading](https://icechunk.io/en/stable/guides/performance/)
- [Open Storage Network S3-compatible access](https://openstoragenetwork.github.io/docs/dataset-access/)
- [MIT ORCD storage services](https://orcd-docs.mit.edu/services/storage-services/)
- [MIT ORCD transfer guidance](https://orcd-docs.mit.edu/filesystems-file-transfer/transferring-files/)
- [Empire AI hardware, storage accounting, and transfers](https://www.cuit.columbia.edu/empire-ai)
- [WeatherBench 2 cloud-optimized datasets](https://github.com/google-research/weatherbench2)
- [WIND remote Zarr training data path](https://github.com/ml-jku/wind)
- [Cody's LLC experiment branch](https://github.com/m2lines/Samudra/tree/llc_cpu_working)
- [Samudra Rust-loader PR #800](https://github.com/m2lines/Samudra/pull/800)
