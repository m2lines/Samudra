<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC4320 ORCD Spike Results, September 3, 2026

These are bounded engineering measurements, not production service-level
claims. Jobs ran through Slurm, read the LLC4320 source without modifying it,
and wrote only to `/orcd/scratch/orcd/008/merose`. Shared-NFS cache state was
not controllable.

## Fixtures

- Two times of `U` and `Theta`, all 51 levels, face 1, `j/i=720:3600`, plus two
  full face-1 `Eta` planes: 6.92 GB decoded.
- Eight times of `Theta`, all 51 levels, face 1, `j/i=720:3600`: 13.54 GB
  decoded.
- Twenty-four times of face-1 `Theta` and `Eta` for time-packing tests.

Fixture manifests record source paths, chunk metadata, modification times,
read durations, and SHA-256 hashes.

A separate metadata-only full-store inventory found 67 arrays and 2.731912 PB
decoded. It classified five time-varying 3-D fields, 17 time-varying surface
fields, six static 3-D masks/fractions, 19 static horizontal geometry fields,
and 20 one-dimensional coordinate/reference fields. Sixty-five resolve to the
`003` filesystem; symlinked `Theta` and `Salt` resolve to `002`. Decision: a
cluster-wide replacement release contains all 67 arrays; an eight-array output
must be labeled as a Samudra-oriented subset.

Applying the selected family layouts to all 67 shapes projects an upper bound
of 4,541,414 physical data objects, before repository metadata and without
omitting all-fill shards. This is below the source's approximately 24.32
million files. The layouts contain about 3.108 billion independently
compressed inner chunks inside those shards, so conversion scheduling must use
complete shards rather than individual inner chunks.

## Codec tournament

All candidates used logical `(time=1, k=51, 72, 72)` chunks in physical
`(time=1, k=51, 1440, 1440)` shards over the same two-time `U`/`Theta` fixture.

| Codec | Bytes (GB) | Decoded ratio | Four tiles (s) | Union (s) |
| --- | ---: | ---: | ---: | ---: |
| Blosc Zstd 5 + bitshuffle | 3.563 | 0.526 | 1.837 | 1.499 |
| Blosc Zstd 3 + bitshuffle | 3.702 | 0.547 | 1.796 | 1.678 |
| Blosc LZ4 5 + bitshuffle | 3.755 | 0.555 | 1.719 | 1.744 |
| Blosc Zstd 1 + bitshuffle | 3.764 | 0.556 | 1.702 | 2.038 |
| Blosc LZ4 5 + byte shuffle | 3.809 | 0.563 | 1.765 | 1.659 |
| Core Zstd 3 | 4.420 | 0.653 | 1.620 | 1.732 |
| Core Zstd 1 | 4.425 | 0.654 | 1.802 | 1.703 |

Decision: Blosc Zstd level 5 with bitshuffle. It was the smallest candidate,
about 2.3% smaller than the corresponding 3.646 GB of existing source objects,
and passed exact Python and pinned `zarrs` compatibility checks.

## Logical volume chunks

Spatial widths `60`, `72`, `90`, `120`, and `180` were first tested at full
depth. A joint tournament then tested `k={3,17,51}` and spatial
`s={72,90,120}`. Candidates more than 1.15 times the smallest physical size
were excluded, then an equal-workload geometric mean ranked four-tile, union,
single-depth, 17-depth, and all-depth crops.

Decision: `(time=1, k=17, face=1, j=120, i=120)`. It occupied 3.587 GB versus
3.580 GB for comparable `k=51`, reduced a single-depth crop from about 39 ms to
18 ms, and retained 1.77-second four-tile and 1.72-second union medians.

## Physical volume shards

All candidates used the selected logical chunk and codec on eight real
`Theta` times. The alternatives held decoded shard volume roughly constant.

| Time × spatial envelope | Objects | Bytes (GB) | Four tiles (s) | Union (s) | Point series (s) | Crop series (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1 × 2160²` | 34 | 6.120 | 0.881 | 0.717 | 0.006 | 0.158 |
| `2 × 1440²` | 18 | 6.119 | 0.767 | 0.703 | 0.004 | 0.142 |
| `4 × 1080²` | 20 | 6.119 | 0.901 | 0.868 | 0.005 | 0.168 |
| `8 × 720²` | 18 | 6.119 | 0.871 | 0.872 | 0.005 | 0.202 |

Decision: `(time=2, k=51, face=1, j=1440, i=1440)`. It won every timed
workload in this tournament and projects to approximately 603,000 physical
objects per volume variable.

`W` has 52 interface levels, so a separate 0.863 GB decoded fixture compared
logical depth 13 in a physical depth-52 shard, logical depth 17 in depth-51
and depth-68 shards, and logical depth 26 in depth 52. Logical 17/physical 68
won the balanced full-depth, single-depth, and 17-depth read score, occupied
568.838 MB, and used one data object. Reusing physical depth 51 created a
second one-level shard. Decision: use nominal physical depth 68 for `W` while
retaining the selected logical depth 17 and volume time/spatial layout.

## Surface layout

The spatial-inner tournament selected `360²`. A 24-time full-face `Eta`
tournament then compared physical envelopes:

| Time × spatial envelope | Objects | Tile (s) | Full face (s) | Point series (s) | Crop series (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `2 × 4320²` | 14 | 0.009 | 0.111 | 0.010 | 0.035 |
| `8 × 2160²` | 14 | 0.027 | 0.117 | 0.006 | 0.029 |
| `8 × 1440²` | 29 | 0.010 | 0.118 | 0.006 | 0.027 |
| `24 × 1080²` | 18 | 0.008 | 0.113 | 0.005 | 0.033 |
| `24 × 720²` | 38 | 0.009 | 0.131 | 0.008 | 0.030 |

All occupied 0.890 GB from 1.792 GB decoded because logical compression
boundaries were identical. Decision: logical `(1, 1, 360, 360)` chunks in
physical `(24, 1, 1080, 1080)` shards. It is the best balanced choice and
projects to about 89,000 objects per surface variable.

## Halo control

| External halo | Tile width | Decoded duplication | Bytes (GB) | Four tiles (s) |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 752 | 1.091× | 1.031 | 1.824 |
| 32 | 784 | 1.186× | 1.122 | 2.009 |

The selected general sharded layout served the current four-tile workload in
1.77 seconds without duplicated values. Decision: no precomputed halos in the
primary archive; retain a sidecar only as a full-rewrite fallback.

## Icechunk and Xarray

- Killed uncommitted writes were invisible; prior snapshots remained immutable;
  retry succeeded; stale commits raised `ConflictError`.
- Eight forked sessions wrote disjoint shards and merged through one
  coordinator. A simulated capacity failure was retried. Four split manifests
  were created, garbage collection removed the orphan, and the release
  snapshot remained exact.
- An eight-time real-data Icechunk pilot encoded 13.54 GB to 6.119 GB in 52.0
  seconds. Xarray 2026.7.0 opened snapshot
  `574X0GAPMDP4DTYPNNK0`; full-array and subset equality passed.

Decision: disjoint forked workers, one merge/commit coordinator, time-oriented
manifest splitting, pinned consumer snapshots, and Xarray as the publication
gate.

The sparse all-face pilot then materialized two times of `U`, `V`, `Theta`,
`Salt`, `Eta`, `oceQnet`, `oceTAUX`, and `oceTAUY`; all 13 faces; four
`720 x 720` corner regions per face; masks, horizontal geometry, and vertical
coordinates. Snapshot `XEDFKS47N3C85MB1RTR0` occupied 17.715 GB in 648
physical objects. It preserved staggered dimensions, opened through pinned
Xarray, and passed exact source equality at all 52 populated face corners. The
write took 382 seconds and peaked at about 40 GiB RSS. This validates the
public dataset shape and reader contract, not cross-face rotation by itself.

Job `21927118` completed a real Slurm requeue. Its durable ledger rejected an
active duplicate claim, retained a completed task across restart, recovered a
stale in-progress claim, verified every artifact checksum, and rejected a
duplicate completed claim. A coordinator deliberately died before commit; its
changes remained invisible, and the restarted coordinator reconstructed and
committed exact output as snapshot `TJ9XZX4HAYDF37266XYG`.

An independent xgcm oracle applied the published LLC/ECCO 13-face connection
table to source and snapshot `XEDFKS47N3C85MB1RTR0`. Exact equality passed for
23,163 populated X-boundary scalar results, 23,254 Y-boundary scalar results,
23,156 X-component staggered-vector results, and 23,240 Y-component results.
This closes the seam-rotation and vector-component-exchange gate for the sparse
pilot.

## Concurrency

One-node sweeps used 1, 2, 4, 8, and 16 outer readers with Zarr async
concurrency fixed at one. Sixteen readers gave peak decoded throughput, but p95
latency rose by about 51% for source `U` on `003`, 55% for source `Theta` on
`002`, and 25% for Icechunk scratch versus eight readers. Cache reuse makes the
absolute throughput unsuitable as a production estimate.

Decision: start the next prototype at eight outer readers. Repeat with disjoint
coordinates on the approved destination before setting production guidance.

## Remaining gates

- approved cluster-wide destination, quota, permissions, backup, and owner;
- projected conversion duration at the approved source/destination limits; and
- non-blocking real-data native Rust/Icechunk reader integration.
