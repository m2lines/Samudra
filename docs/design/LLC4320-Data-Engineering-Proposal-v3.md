<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC4320 Remote-Streaming Training Store

Status: draft for review

## Summary

Empire AI provides enough GPU compute for large LLC4320 experiments but this
proposal assumes that our allocation can retain only 100 TiB, far less than
the 2.732 PB decoded LLC4320 archive. Copying the complete archive to every
training system is therefore not viable.

Build a small, immutable, training-oriented Zarr v3 repository on MIT-managed
storage and expose it through an operator-approved HTTPS or S3-compatible
endpoint. Torch or Empire AI workers will read only the byte ranges needed for
their spatial patches. A bounded cache on each GPU node, optionally backed by
the remote cluster's shared 100 TiB allocation, will retain compressed ranges
or complete shards. The replay loader will reuse cached states and predictions
rather than repeatedly crossing the WAN.

This is not a proposal to serve `/orcd/data/abodner/003` directly, nor to make
an ad hoc web server on an ORCD login or Slurm compute node. The origin must be
a separately materialized, read-only repository behind an approved data
service. The production source remains immutable and private to ORCD.

The v2 experiments remain the local-filesystem baseline, but they do not settle
the remote layout. WAN latency changes the optimum: a logical chunk that was
appropriate for arbitrary scientific access may generate too many range
requests for full-depth training. V3 therefore adds a remote-layout tournament
and treats the following as hypotheses rather than final choices:

- pack complete depth into each logical training chunk;
- test `120`, `240`, and `360` spatial widths;
- retain approximately `1440 x 1440` physical shards and two-time grouping;
- retain Blosc Zstd level 5 plus bitshuffle as the first codec;
- compare canonical per-variable arrays with a clearly labeled float16 packed
  training projection; and
- fetch the four-tile `1472 x 1472` union once per node and scatter tile views
  in memory.

The first goal is not a large training run. It is a small remote corpus and a
staged benchmark that tells us whether one GPU node, and then several nodes,
can keep GPUs busy without exceeding an agreed MIT egress budget. A model
training launch remains a separate approval step.

## Decision requested

Approve engineering of a bounded remote-streaming prototype, conditional on:

1. ORCD approving the storage location and the read-only HTTPS or
   S3-compatible serving path;
2. Empire AI and Torch confirming outbound HTTPS/S3 access from compute nodes;
3. an agreed MIT egress and concurrency ceiling; and
4. approximately 1 TiB of stable MIT-side prototype capacity.

Do not authorize a full LLC4320 rewrite from this document.

## Goals

- Determine whether remote Zarr reads can hide beneath Samudra compute at
  useful GPU scale.
- Make the experiment representative of Cody's four overlapping spatial
  patches and replay-buffer behavior.
- Minimize bytes crossing the WAN and avoid duplicate requests across GPUs on
  one node.
- Preserve an ordinary Xarray/Zarr view for inspection and debugging, even if
  the winning training projection also exposes packed arrays.
- Use immutable version identifiers so node caches are safe and reproducible.
- Produce enough evidence to decide between remote streaming, rolling staging,
  and a larger remote storage allocation.

## Non-goals

- Replacing the complete, generally usable LLC4320 archive proposed in v2.
- Publishing all 67 LLC4320 arrays in the first remote prototype.
- Serving the source NFS tree over the public internet.
- Assuming that internal InfiniBand or NVLink performance predicts WAN
  performance.
- Requiring the experimental Rust loader for the first result. Xarray/Zarr is
  the acceptance path; the Rust loader is a follow-on optimization.
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
bandwidth from an allocated GPU node to MIT. Torch documents 100-Gb/s Ethernet
on its data-transfer nodes and recommends those nodes and Globus for bulk
transfers. A DTN specification is not evidence that Torch GPU nodes can sustain
equivalent random range reads from MIT.

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
| General physical volume shard | `(time=2, k=51, face=1, 1440, 1440)` won every measured local workload. | Retain as the initial physical envelope; test changes only after measuring real range requests. |
| Surface layout | Logical `(1,1,360,360)` in physical `(24,1,1080,1080)` was the best local balance. | Retain initially; surface bytes are secondary to the four full-depth fields. |
| Halo sidecar | A 16-cell sidecar duplicated 9.09% and was slower than the general sharded layout; 32 cells duplicated 18.57%. | Do not precompute halos for the canonical remote arrays. |
| Correctness | Pinned Xarray reads, all-face corners, staggered dimensions, and an independent xgcm topology oracle passed exactly. | Reuse the schema and validators; remote transport must not change scientific semantics. |
| Recovery | Fork/merge, failure retry, garbage collection, a persistent ledger, and a real Slurm requeue passed. | Build the remote corpus transactionally and publish only an immutable snapshot. |
| Concurrency | Eight outer readers was the conservative ORCD starting point; 16 increased tail latency. | Do not transfer this number to WAN clients. Measure concurrency independently at each origin/destination pair. |

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

The v2 logical volume chunk is approximately 0.93 MiB decoded and about half
that size compressed on the fixture. A full-depth selection must retrieve
three depth chunks at each spatial position. Across four variables and many
spatial positions, a training example can create hundreds or thousands of
logical range reads.

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
remote shared cache (optional, <= 100 TiB)
              |
              v
one compressed-range/shard cache per GPU node
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

Use three distinct cache layers and report each separately:

- **L1 decoded/prefetch cache:** process memory holding ready tensors for the
  next few steps and replay entries.
- **L2 node cache:** compressed ranges or complete immutable shard objects on
  node-local NVMe. All GPU ranks on the node share this cache through one fetch
  coordinator or a lock-safe content-addressed directory.
- **L3 shared working set:** optional shards staged in the remote cluster's
  100 TiB allocation. Populate it explicitly or on demand, with eviction keyed
  by snapshot and object identifier.

Never let eight ranks on one node independently fetch the same shard. Deduplicate
in-flight requests and fetch the `1472 x 1472` group union once when the four
tiles share a time and face. Scatter or create views after decode.

Immutable snapshot IDs must be part of every cache key. A release change uses
a new namespace; it never mutates cached bytes in place.

### Reader path

Plan A uses pinned Xarray, Zarr Python, and Icechunk versions. Open the
repository and snapshot once per long-lived worker, preload bounded coordinate
and relevant time-manifest metadata, and retain persistent HTTP connections.
Do not reopen the repository per sample.

Start with one outer data-fetch pipeline per GPU node, not one unconstrained
pipeline per rank. Bound:

- in-flight samples;
- total HTTP requests;
- bytes resident in compressed and decoded queues;
- codec worker threads; and
- host-to-device copies.

Zarr's async concurrency and the DataLoader worker count multiply. Sweep them
as a joint configuration and enforce a node-wide semaphore.

The Rust loader is Plan B. It should consume the same immutable snapshot and
cache protocol after Plan A establishes feasibility. It is not a prerequisite
for creating or publishing the experiment corpus.

## Prototype corpus

### Scope

Build a training-specific repository, clearly named as a derivative, for
example `llc4320-samudra-streaming-pilot-v1`. It is not the public 67-array
archive.

Include:

- `U`, `V`, `Theta`, and `Salt`, all 51 levels;
- `Eta`, `oceQnet`, `oceTAUX`, and `oceTAUY`;
- required masks, coordinates, and normalization metadata;
- one representative face containing ocean, coast, and land; and
- consecutive times long enough to exercise ordinary transitions, seed
  refreshes, and replay; plus
- a two-time sparse seam/corner canary across all 13 faces, reusing the v2
  all-face regions and topology oracle.

For the first retained origin, use 96 consecutive hours and one complete face.
The source-byte estimate is roughly 0.5 TiB for the eight dynamic arrays,
although the actual converted size must be measured. This fits a 1 TiB origin
budget while providing 95 adjacent transitions and many spatial locations.

Before retaining that copy, run the layout tournament on a smaller corpus:

- 256 consecutive times;
- one face;
- the complete `1472 x 1472` four-tile union plus enough neighboring cells to
  avoid an artificial alignment advantage; and
- identical values for every candidate.

Encode candidates serially and delete or archive losers only after their
measurements have been captured. Do not keep several half-terabyte candidate
stores indefinitely.

### Canonical and packed views

Compare two schema families:

1. **Canonical:** source variable names and dimensions, retaining float32.
   This is easiest to inspect and most directly exercises the future general
   archive.
2. **Training projection:** explicitly lossy float16 arrays, optionally packed
   as `prognostic(time, face, channel, j, i)` and
   `boundary(time, face, channel, j, i)`, with a manifest mapping every channel
   to source variable and level. This follows Cody's successful cache pattern
   and reduces HTTP requests.

The packed projection is acceptable only as a reproducible derivative. It must
record source snapshot/provenance, channel order, casting rule, masks, codec,
and exact generator commit. It must never be presented as the scientific
archive.

Do not duplicate the four overlapping tiles in the retained origin. Preserve a
spatial field and select the union at read time. A sample-chunked or duplicated
tile store may be included as a performance control, but it should win only if
the measured utilization improvement justifies its fixed geometry and storage
amplification.

## Candidate remote layouts

### Volume arrays

Keep physical shards fixed initially so the logical tournament isolates WAN
request behavior:

```text
physical shard: (time=2, depth=51, face=1, j=1440, i=1440)
codec:          Blosc Zstd level 5 + bitshuffle
```

Test:

| Candidate | Logical chunk | Purpose |
| --- | --- | --- |
| V2 general baseline | `(1,17,1,120,120)` | Measures the cost of general depth access over WAN. |
| Full-depth control | `(1,51,1,120,120)` | Cuts full-depth request count by approximately three without changing spatial granularity. |
| WAN-balanced | `(1,51,1,240,240)` | Fewer, multi-megabyte ranges with moderate crop amplification. |
| WAN-coarse | `(1,51,1,360,360)` | Low request count and greater spatial amplification. |

The leading hypothesis is the WAN-balanced canonical layout:

```text
logical chunk:  (time=1, depth=51, face=1, j=240,  i=240)
physical shard: (time=2, depth=51, face=1, j=1440, i=1440)
```

Its decoded float32 logical chunk is about 11.2 MiB per variable. It is large
enough to amortize a remote request while preserving substantially finer
spatial access than a complete 720-cell tile. This is a hypothesis to test, not
a decision imported into metadata before the tournament.

Also test Zarr range coalescing with explicit gap and maximum-read budgets,
including the shipped defaults, rather than treating it as an implementation
detail.

After selecting a logical layout, compare physical time grouping of 2 and 4
with spatial envelopes of 1440. The logical chunks remain one time each. A
larger time shard should win only if it reduces metadata/object overhead
without causing whole-object transfers, cache churn, or poor recovery units.

### Surface arrays

Begin with the v2 result:

```text
logical:  (time=1, face=1, j=360, i=360)
physical: (time=24, face=1, j=1080, i=1080)
```

Add a `720 x 720` logical spatial control. Surface data is a small fraction of
the prognostic payload, so do not optimize it at the expense of more important
volume experiments.

### Packed float16 control

For the packed `205`-channel prognostic array, test full-channel logical chunks
with spatial widths 120 and 240:

```text
logical candidates: (time=1, face=1, channel=205, j={120|240}, i={120|240})
physical controls:  (time=2, face=1, channel=205, j={720|1440}, i={720|1440})
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
        +--> WAN range microbenchmark on Torch
        |             |
        |             +--> repeat on Empire AI
        |
immutable source corpus
        |
        +--> logical chunk / codec / packing tournament
                        |
                        +--> publish winning 96-hour snapshot
                                      |
                                      +--> reader and cache benchmark
                                                    |
                                                    +--> one GPU-node loader benchmark
                                                                  |
                                                                  +--> 2/4/8-node scaling
                                                                                |
                                                                                +--> short training run
```

### Spike 0: endpoint and policy preflight

Resolve before writing the corpus:

- approved MIT storage and serving service;
- authentication method and credential lifetime;
- TLS and byte-range support;
- whether `HEAD`, conditional GET, and concurrent range GET are supported;
- outbound access and proxy requirements on Torch and Empire AI compute nodes;
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

Repeat from Torch and Empire AI. Use unique ranges for cold-like passes and
immediate repeats for warm passes; do not claim control of shared server or
kernel caches.

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

1. **Remote origin meets utilization and egress gates:** proceed with a larger
   immutable training projection and treat remote storage as a managed cache.
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
2. What are the measured WAN paths and policy limits from Torch and Empire AI
   compute nodes, rather than their DTNs or internal fabrics?
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
9. Does the eventual job target Torch H200 nodes, Empire AI Alpha H100 nodes,
   or Empire AI Beta B200 nodes first?

## References

- [Zarr remote storage backends](https://zarr.readthedocs.io/en/latest/user-guide/storage/)
- [Zarr sharded range-read coalescing](https://zarr.readthedocs.io/en/latest/user-guide/examples/sharding_coalescing/)
- [Zarr concurrency guidance](https://zarr.readthedocs.io/en/main/user-guide/performance/)
- [Zarr GPU buffers and current host-codec limitation](https://zarr.readthedocs.io/en/stable/user-guide/gpu/)
- [Icechunk storage backends, including S3 and read-only HTTP](https://icechunk.io/en/stable/storage/)
- [Icechunk performance and manifest preloading](https://icechunk.io/en/stable/guides/performance/)
- [Open Storage Network S3-compatible access](https://openstoragenetwork.github.io/docs/dataset-access/)
- [MIT ORCD storage services](https://orcd-docs.mit.edu/services/storage-services/)
- [MIT ORCD transfer guidance](https://orcd-docs.mit.edu/filesystems-file-transfer/transferring-files/)
- [Torch hardware](https://services.rt.nyu.edu/docs/hpc/spec_sheet/)
- [Torch data-transfer nodes](https://services.rt.nyu.edu/docs/hpc/storage/data_transfers/)
- [Empire AI hardware, storage accounting, and transfers](https://www.cuit.columbia.edu/empire-ai)
- [WeatherBench 2 cloud-optimized datasets](https://github.com/google-research/weatherbench2)
- [WIND remote Zarr training data path](https://github.com/ml-jku/wind)
- [Cody's LLC experiment branch](https://github.com/m2lines/Samudra/tree/llc_cpu_working)
- [Samudra Rust-loader PR #800](https://github.com/m2lines/Samudra/pull/800)
