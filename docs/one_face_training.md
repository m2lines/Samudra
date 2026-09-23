# Full-face (1-face) replay training

## Context

Today the trainer can advance **one tile** (720² or 752²) per replay row, or a
**2×2 cluster** of 752² tiles blended at their seams — but a cluster lives
entirely on one rank, because a replay row is stored as `[T, C, H, W]` and its
tiles are flattened into the batch dimension of a single forward pass.

We want to advance a **whole LLC face** — all 36 tiles of `face_1` — as one
synchronously-blended unit, sharded across several GPUs. This is the third task
in an eventual multitask curriculum (1-tile → 4-tile cluster → 1-face → region →
globe), so the design has to make "which tiles are in a group and which rank
owns them" a *parameter*, not a structural assumption.

Data: `/orcd/data/abodner/002/cody/LLC_patch/face_1/LLC4320_face1_i0-4320_j0-4320.zarr`
— a `llc-train-ready-v1` packed cache, `prognostic [10311, 205, 4320, 4320]`
float16 chunked `[1, 205, 720, 720]`, `boundary [10311, 4, 4320, 4320]`, plus
`XC`/`YC`/`rA` and per-channel masks. Pre-normalized, land = NaN.

---

## 1. Tile geometry — uniform 752², no mixed shapes

The original sketch gives interior tiles 752², corner tiles 736² and edge tiles
752×736. **Do not do this.** Two things break:

* `TileBlender.__init__` (`src/ocean_emulators/tiling.py:779`) raises
  `"TileBlender needs uniformly shaped tiles"` — its `weights` buffer is a
  stacked `[T, 1, H, W]` tensor.
* Three shapes cannot share one batched forward, so every chunk becomes three
  forwards.

**Instead, clamp the tile origin inward at the face boundary:**

```python
start(c) = min(max(0, 720 * c - 16), 4320 - 752)
# -> [0,752) [704,1456) [1424,2176) [2144,2896) [2864,3616) [3568,4320)
```

* All 36 tiles are **752 × 752**. One shape, one batch, blender unchanged.
* Coverage of the 4320² face is exact; no data is invented and nothing is padded
  with synthetic values.
* Interior seams overlap by 32 (16 + 16) as designed. The four **face-edge**
  seams overlap by 48. `TileGroupLayout.overlaps` is already keyed per
  `(tile_id, side)` and `_tile_window` ramps each side over its own width, so
  the partition of unity still holds at every seam.
* `ownership_boxes` splits each band at its midpoint, so ownership tiles the
  face exactly: 728 + 712 + 720 + 720 + 712 + 728 = 4320 per axis.
* The interior windows are **byte-identical** to the ones the existing
  `752-4-tile` cache already uses (`i1424-2176`, `i2144-2896`), so this is the
  validated convention extended, not a new one.

The U-Net builds one down/up stage per `ch_width` pair and `ch_width` becomes
`[in_channels] + [448,576,640,768]`, i.e. **4 downsamples**, so inputs must be
divisible by 16. 752 = 16 × 47. `group_norm_groups: 32` divides 448/576/640/768.

**The "slice/copy action" in the original sketch is free.** Because each tile's
read window already *includes* its overlap, tiles drawn from truth agree
exactly in their shared cells by construction. That is precisely the invariant
`TileBlender.blend` documents as its precondition, so seeding needs no extra
step.

New pure function in `tiling.py`:

```python
def face_tile_windows(face, *, extent=4320, tile=720, overlap=16)
    -> list[tuple[int, int, int, int, int]]   # (face, i0, i1, j0, j1)
```

These feed `data.llc_tiles`, which **already exists**
(`src/ocean_emulators/config.py:211`) and cuts N windows out of one store —
`DataConfig.build` turns each into its own replay source and
`Trainer._build_tile_catalog` (`train.py:4094`) already builds the catalog from
`replay_windows`. No data-layer surgery is needed to *address* the 36 tiles.

### Measured land distribution (face 1, surface mask)

```
0.995 1.000 1.000 1.000 1.000 1.000
1.000 1.000 1.000 1.000 1.000 1.000
1.000 1.000 1.000 0.785 0.653 1.000
1.000 1.000 1.000 0.424 0.104 0.689
0.924 1.000 1.000 0.336 0.000 0.736
1.000 0.761 0.407 0.062 0.000 0.315
```

Face is 78.2% ocean. **Two tiles are 100% land**, one more is 6% wet, eight are
under 50%. All 36 are advanced identically — dropping any breaks blender
coverage, and compute is identical for land and ocean anyway. The real issue
land creates is loss normalization, handled in §6.

---

## 2. Data path — chunk-streaming tile reader

This is the binding constraint, not GPU partitioning.

A tile window `[720c-16, 720c+736)` starts 16 cells *before* chunk boundary `c`
and ends 16 cells *after* boundary `c+1`, so it spans **three** chunk columns.
Read naively an interior tile touches 3×3 = 9 chunks; the clamped end tiles
span only two columns, so a face costs 256 chunk decodes to deliver 36 chunks
of distinct data — **7.1× amplification**.

One face timestep is 36 chunks ≈ **5 GB compressed / 7.8 GiB f16**.

**Design: decode each chunk exactly once and scatter it into every tile buffer
that overlaps it.**

```
for each chunk the rank's tiles intersect:
    decode once  ->  [205, 720, 720]
    for each local tile overlapping it:
        copy the intersecting sub-rectangle into that tile's buffer
    drop the chunk
```

* A rank owning a contiguous 3×3 block of tiles needs 4×4 = 16 chunks for 9
  tiles. Across 4 ranks that is 64 decodes vs 36 — **1.8×**, down from 7.1×.
* Peak host memory per in-flight frame is the *tiles*, not the face:
  9 × 232 MB ≈ 2.1 GB plus a few chunks of thread scratch. The whole face is
  never materialized.
* Disk-level sharing between ranks comes free from the OS page cache — all
  ranks read the same files at the same timestamp, and the node has ~2 TB.

This is why tiles are assigned to ranks as **contiguous 3×3 blocks**: chunk
locality. (Wet-fraction balancing is *not* needed for this; see §6.)

### Measured (`scripts/_bench/face_read.py`, node4005, 2026-09-22)

One whole face-frame of truth — 36 tiles × 205 channels × 752², float16 — with
the ranks reading concurrently and the page cache dropped before each frame:

| config | per face-frame | chunk decodes | throughput |
|---|---:|---:|---:|
| 4 ranks × 8 threads, **chunked** | **2.91 s** | 64 | 2.70 GiB/s |
| 4 ranks × 8 threads, naive | 14.50 s | 256 | 0.54 GiB/s |
| 4 ranks × 4 threads, chunked | 4.16 s | 64 | 1.87 GiB/s |
| 4 ranks × 15 threads, chunked | 7.85 s | 64 | 0.99 GiB/s |
| 9 ranks × 6 threads, chunked | 5.53 s | 100 | 1.41 GiB/s |

**Chunk-streaming is 5.0× faster than the naive per-tile read** — more than the
1.8× decode ratio alone, because the naive path also re-reads the same
compressed bytes off NFS.

Two things that were not obvious before measuring:

* **Total concurrency has a sharp optimum near 32 readers.** 4×8 is 2.7×
  faster than 4×15 on identical work. Read threads must be sized against the
  store, not the core count: ranks × threads should land near 32, not 60.
* **More ranks read _slower_.** Nine ranks hold 2×2 tile blocks, which need 100
  chunk decodes rather than 64, and 9×6 also overshoots the concurrency
  optimum. The extra GPUs buy less than their compute scaling suggests.

Caveat: node4005 was shared (108/192 CPUs held by other jobs, load ~184), so
these are contended numbers — pessimistic for a dedicated node, but that is
also the realistic condition.

Where it goes: a new reader alongside the existing per-tile `DataSource` path,
used by `_ReplayPrefetchPipeline._load_raw_batch` (`train.py:566`) when the
group is face-sized. Keep the existing per-tile path intact for the 1-tile and
2×2 tasks.

**Known gap:** `DataConfig.build` sets `NativeStoreSpec.packed_prefix` only for
the *boundary* store, so `loader_backend: rust` cannot currently read a packed
prognostic cache. The Rust reader itself handles packed caches
(`rust/llc_load/src/lib.rs:63`). Start on `loader_backend: cpu`; wiring
`packed_prefix="prognostic"` is a small follow-up worth doing, since the Rust
reader releases the GIL and the prefetch workers are threads, not processes.

---

## 3. Sharding — what each rank owns

| GPUs | tiles/rank | chunks/step | compute/face-advance |
|------|-----------:|------------:|---------------------:|
| 4    | 9          | 3 × 3       | ~7.7 s |
| 6    | 6          | 2 × 3       | ~5.1 s |
| **9**| **4**      | **1 × 4**   | **~3.4 s** |
| 12   | 3          | 1 × 3       | ~2.6 s |

Compute extrapolated from the measured 3.4 s / 110 GB peak for 4 × 752² tiles
on one H200. Folding in the measured read cost above — the two overlap, since
the reader runs in a prefetch thread:

| GPUs | compute | truth read | **step** |
|------|--------:|-----------:|---------:|
| 4    | ~7.7 s  | 2.91 s     | **~7.7 s** (compute-bound) |
| 9    | ~3.4 s  | 5.53 s     | **~5.5 s** (I/O-bound) |

So G=9 is only ~1.4× faster than G=4, not the 2.3× its compute suggests: past
four ranks the read loses more to block fragmentation and concurrency than the
GPUs gain. Using G=9 means retuning the read first — fewer threads per rank,
and possibly 1×4 tile strips rather than 2×2 blocks so a rank's chunk footprint
stays closer to its tile count.

36 divides by 4, 6, 9, 12 — **not 8**, and the h200 nodes are 8-GPU. On one node
the clean options are 4 or 6; 9 or 12 needs two nodes (the seam exchange is only
~0.1–1 GB/step, fine over IB).

`tiles_per_rank` is **derived** from `world_size` with an assert, never
hardcoded — G becomes a launch-time knob and the globe case follows the same
code. (Note for later: 468 tiles does not divide by 32; its divisors near there
are 26 and 36.)

**Intra-step chunking.** Each rank runs `ceil(tiles_per_rank / chunk_size)`
forward/backward chunks per face advance, accumulating gradients, and the chunk
count **must be identical on every rank** or DDP deadlocks. Reuse the existing
pattern in `train_one_epoch_replay` (`train.py:1990`): `ddp_model.no_sync()` for
every chunk but the last. `chunk_size = 3` for G = 4 (~83 GB), `4` for G = 9.

---

## 4. Replay row = one face

A row addresses a whole face; each rank stores only **its own tiles' shard** of
that row.

* **Rows stay on pinned CPU**, as they are today
  (`ReplayBuffer._prepare_entry`, `replay.py:152`), in bf16 because
  `use_bfloat16: true`.
* Per-rank shard = 9 × 232 MB = **2.09 GB**. H2D + D2H per step ≈ 4.2 GB, which
  over PCIe Gen5 is ~0.1 s against a ~7.7 s step — **~1%**, and it is already
  overlapped on `self.replay_copy_stream` with `ready_event` (`train.py:1268`,
  `2754`, `649`). Moving rows to HBM would buy ~1% of wall clock and cost
  2.09 GB of HBM per row. Not worth it.
* **`buffer_size: 8`** → 16.7 GB/rank pinned, 67 GB/node. Leaves ample room for
  the in-flight truth frames (§2) and the loader.
* **`checkpoint_buffer: false`**, reseed from gold on restart. Reseeding 8 face
  rows costs ~8 face reads ≈ 40 GB ≈ under a minute. Rows restart at
  `lead_step = 0`, but a 48 h job is ~20 k steps, so the buffer re-equilibrates
  long before it matters.

**Rank coordination.** All ranks must pick the *same* row every microbatch. The
mechanism already exists for `dp_ctx`: seed `replay_generator` identically on
every rank instead of `rand_seed + 104729 * get_rank()` (`train.py:1261`). Every
rank then runs `_plan_replay_batch_local` and derives an identical plan with no
broadcast.

**One hazard:** `_replay_state_diverged` (`train.py:2726`) is data-dependent and
per-rank, so one rank could reseed a row while others advance it, silently
desyncing the buffers. This needs an `all_reduce(MAX)` on the diverged flag
before `apply_replay_prefetch_updates` acts on it.

---

## 5. Blend — seam-only exchange

`blend_before_backward: false`, so the blend runs **after** `backward`, under
`no_grad`, on detached predictions. Nothing crosses the autograd graph and no
collective is needed inside the loss.

Because the quintic weights are exactly 1 outside the overlap and form a
partition of unity, **the blend changes only the overlap bands** — a seam-only
exchange is *mathematically identical* to `TileBlender.blend`, not an
approximation.

Per tile, per neighbour, bf16, 205 channels:

* edge band: 205 × 32 × 752 × 2 B ≈ 9.9 MB
* corner square: 205 × 32 × 32 × 2 B ≈ 0.42 MB

With contiguous 3×3 blocks only the block perimeter crosses ranks — roughly
130 MB per rank per step, well under 0.1 s.

New in `tiling.py`, beside `TileBlender`:

```python
class HaloExchangePlan:   # static, built once from layout + tile->rank map
class DistributedTileBlender:
    def blend(self, local_tiles) -> local_tiles   # same contract as TileBlender.blend
```

Implement with `torch.distributed.batch_isend_irecv` against a precomputed
static plan — deterministic, no dynamic allocation. Reuse `ramp_profile`,
`TileGroupLayout` and `_overlap_width`; `_seam_pairs` in `tiled_eval.py:119` is
a *diagnostic* helper that deliberately skips corner squares, so it is not
directly reusable for the exchange.

Predictions from earlier chunks are held detached in bf16 until all chunks
finish (9 × 232 MB = 2.1 GB on GPU), then blended once per face advance.

**Acceptance test:** `DistributedTileBlender` must reproduce
`TileBlender.blend` to bitwise equality at `world_size=1`, and to float
tolerance at `world_size=4` under gloo/CPU.

---

## 6. Loss

### A batch-size bug found on the way (fixed)

`_weighted_channel_mean` scaled its denominator by `loss.shape[0]` whether or
not the weight already carried a batch axis. `wet` and `spatial_weight` do not,
so the scaling is right for them. `sample_weight` — the per-tile land mask a
mixed-tile batch needs — *does*, so its sum already covered the batch and the
extra factor divided the loss by the batch size a second time:

```
constant error of 1, sample_weight all ones:  B=1 -> 1.0   B=4 -> 0.25   B=9 -> 0.11
```

`gradient_z_l1_loss` already guarded against exactly this (its `expand` carries
a comment saying so); `decomposed_mse/mae/mse_mae` and `gradient_h` did not.

This was **live in every 2×2 cluster run** — those logs show `Per-tile wet masks
differ ...`, which is what turns `sample_weight` on. Consequences for runs
already trained:

* the base metric and `gradient_h` were 4× too small, so the effective learning
  rate was ~4× below nominal;
* `gradient_z` was *not* scaled down, so `lambda_z` was effectively 4× larger
  relative to the other terms than configured.

Fixed by scaling only when the weight has no batch axis. Every single-tile run
is bit-identical; 2×2 loss values shift by 4×, so they are no longer comparable
to older wandb curves. It matters more at 36 tiles, where the factor would have
been the chunk size — and would have silently switched off whenever the tiles
happened to share a land mask.

### Global normalization (implemented)

`_weighted_channel_mean` normalized by the **local** wet count. With DDP
averaging per-rank losses, the objective became
`(1/R) sum_r [wet-weighted mean over rank r's tiles]`, so a rank holding the
face's two dead tiles had its few wet cells weighted up by the ratio of the
wet counts -- about 3.4x for face 1 under contiguous blocks.

The same defect broke chunking, which is what forced the fix: a chunk mean
cannot be added to another chunk's unless both divide by the same thing.

**Both go away with a denominator computed from the masks up front.** It does
not depend on the predictions, so each chunk can divide by it immediately and
backward straight away; the chunk contributions are then shares of one mean
and simply sum. Make that denominator face-wide and the cross-rank distortion
goes too. `gradient_z` needed its own, because it normalizes by valid level
PAIRS rather than wet cells and is an average OF ratios rather than one ratio
-- so both its pair denominators AND its pair counts have to be held fixed, or
a pair that is all land in one chunk is dropped from that chunk's average and
kept in another's.

**Only the training loss is normalized this way.** A share is the right thing
to accumulate gradients from and the wrong thing to report, so `train_loss_fn`
carries the fixed denominators and `loss_fn` -- validation, checkpoint
selection, the autoregressive rollouts -- keeps the ordinary batch-derived
ones. Without that split `best_validation_ckpt` would be ranking on a constant
multiple of the metric it names.

Side effect: absolute *training* loss values are no longer comparable to
previous runs in wandb. Validation numbers still are.

### Channel weights

`GradientLossConfig` — what every LLC config actually uses — has **no
per-channel weighting**; `WeightedLoss` (where the 1.5 lives,
`utils/loss.py:498`) is never constructed on this path. So Eta is effectively
**1.0 today, not 1.5**.

Add a `channel_weights: dict[str, float]` to the loss config, applied as a final
per-channel scale wrapping whatever `build_loss_fn` returns
(`config.py:1182`), resolved by name against
`TensorMap.prognostic_var_names`. Set `Eta: 5.0`; everything else stays 1.0.
Applies to the gradient terms too, since they share `_weighted_channel_mean`.

### Masks

`self.tile_wet_masks` (`train.py:987`) currently stacks *all* sources:
36 × 205 × 752² bool = **4.2 GB of HBM**. Restrict it to the rank's own tiles
(9 → 1.04 GB). Same for `_grouped_val_weight` (`train.py:3271`), which would
otherwise materialize 36 × 205 × 752² **float32 = 16.7 GB** — chunk it.

---

## 7. Eval

`tiled_eval.py` already does exactly the right thing: seed every tile from
truth (overlaps agree by construction), step each tile, blend residuals with
the quintic operator at every AR step, scatter back, write a canonical stitched
zarr. It just needs to reach 36 tiles.

* **Single GPU, sequential tiles.** State ≈ 17 GB + inputs ≈ 17 GB + one tile's
  activations ≈ 27 GB → ~62 GB peak, which fits an H200. ~11 s per AR step, so
  a 480-step rollout is ~1.5 h.
* The only real change: `TiledEval.__init__` builds its catalog from a
  *directory of pre-cut caches* via `_open_raw_caches` (`tiled_eval.py:410`).
  Teach it the `data.llc_tiles` path using the same
  `tile_catalog_from_windows` branch `Trainer._build_tile_catalog` already has.
* Keeping eval single-process and independent of the training sharding is
  deliberate: a bug in the distributed blend cannot hide in both.

---

## 8. Config and launch

New `configs/samudra_llc/train_replay_1face.yaml` (from `train_replay-1.yaml`)
and `JOBS/train_llc_replay-1_face.sh` (from `train_llc_replay-3.sh`).

```yaml
batch_size: 1                     # one row = one face
gradient_accumulation_steps: 1    # see below
replay:
  enabled: true
  grouped: true
  buffer_size: 8
  blend_window: quintic
  blend_before_backward: false
  checkpoint_buffer: false
  steps_per_epoch: 2500
epochs: 50
learning_rate: 0.0006
scheduler: { type: cosine, target_epochs: 70 }
loss:
  type: [gradient_h, gradient_z]
  metric: mse_mae
  lambda_h: 0.1
  lambda_z: 0.1
  channel_weights: { Eta: 5.0 }
```

New env knobs in the job script:

```bash
TILE_OVERLAP="${TILE_OVERLAP:-16}"        # constant across epochs
FACE="${FACE:-1}"
TILES_PER_CHUNK="${TILES_PER_CHUNK:-3}"   # 3 for G=4, 4 for G=9
GPUS="${GPUS:-4}"
BOUNDARY_VARS_KEY="${BOUNDARY_VARS_KEY:-all_fw_noeta}"
VALID_MASK="${VALID_MASK:-true}"
```

Settled elsewhere: `pad: constant`, `pred_residuals: true`,
`ch_width: [448,576,640,768]`, `group_norm_groups: 32`, spatial features on
(XC/YC/rA → 4 channels). Input channels = 205 prognostic + 4 boundary + 4
spatial + 1 valid mask = **214**. The cache's `boundary_channel_names` are
exactly `[oceTAUX, oceTAUY, oceQnet, oceFWflx]` = `all_fw_noeta`. ✔

**Training from scratch** — no warm start. `wider_model-W-7` carries vertical
velocity as a prognostic, which must not leak into this model, and the wide 2×2
checkpoint used `boundary_vars_key: all` (Eta as forcing, not oceFWflx).

**Why `gradient_accumulation_steps: 1`.** A face advance is already 36 tiles.
With `accum: 4` the effective batch is 144 tiles against the 2×2 run's 16 — 9×
larger at the same LR 6e-4. At `accum: 1` it is 36 tiles, a 2.25× increase,
and one face advance equals one optimizer step.

**`steps_per_epoch: 2500`.** Epochs are the curriculum clock (`max_lead` and
`refresh_every_n_microbatches` transitions are indexed by epoch), so keep 50
epochs and shrink the epoch. At `buffer_size: 8` a row needs ~77 microbatches
to reach the lead cap and gets reseeded every ~8 × 83 microbatches late in
training, so 2500 gives the late epochs room. Budget: ~5.6 h/epoch at G=4
(~12 days for 50), ~2.8 h at G=9 (~6 days).

**`SCHEDULER_TARGET_EPOCHS`.** Keep **70**, not 50. `target_epochs` is `T_max`
of `CosineAnnealingLR` (`utils/schedule.py:98`), so 70 leaves LR at ~20% of peak
at epoch 50 — which is exactly the regime where EMA weights earn their keep. 50
anneals LR to zero and makes EMA close to redundant. It also keeps the LR
trajectory comparable to the 1-tile and 2×2 runs.

---

## 9. Designed-for extension (not built now)

* **Multitask replay.** A row already carries a `ReplayGroup`. Add a `task`
  field to `ReplayCursor` and let `replay_groups` hold heterogeneous groups —
  1-tile, 4-tile cluster, 36-tile face. `sample_replay_seed_cursor` then draws a
  task, and everything downstream (`datasets_for`, `replay_group_for`,
  `_blend_microbatch_predictions`) already dispatches per group.
  The two things that must *not* be hardcoded for this to work later, and are
  not in this plan: `tiles_per_rank` and the tile→rank map.
* **Globe.** `build_group_layout` explicitly raises on cross-face groups —
  cross-face halos need a rotation operator that does not exist. Everything
  else (windows, blender, sharding, seam plan) carries over.

---

## 10. Status

Done, with tests:

**Geometry and blending** (`tiling.py`)
* `face_tile_windows` / `tile_origins` — the clamped 36-window geometry, with
  coverage, ownership and seam-width properties (`tests/test_tiling.py`).
* `tile_window` lifted to module level so both blenders build identical weights.
* `halo_ops`, `contiguous_tile_blocks`, `DistributedTileBlender`, held to
  **bitwise equality** with `TileBlender` at world sizes 1, 4 and 9 for both
  windows (`tests/test_distributed_blend.py`). Matching bit for bit needed the
  numerator *and* the denominator accumulated in catalog order.

**Reader** (`chunk_reader.py`)
* `chunk_plan` + `GroupChunkReader` — decode each store chunk once, scatter it
  into every tile that overlaps it, drop it. Verified byte-identical to the
  per-tile read against a synthetic packed cache, and against the **real face
  cache** for a clamped corner tile and an interior tile
  (`tests/test_chunk_reader.py`, the latter marked `manual`).

**Sharding** (`face_parallel.py`)
* `assign_tiles`, `split_into_chunks`, `FaceParallelContext`,
  `face_group_is_shardable` — ownership, equal chunk counts, the collective
  divergence vote and the face-wide loss denominator, with 4- and 9-rank gloo
  tests (`tests/test_face_parallel.py`).

**Divergence** (`replay.py`)
* `diagnose_replay_state` / `describe_divergence`, wired into
  `apply_replay_prefetch_updates`: a diverged row names the offending tiles and
  their magnitudes, then the whole row is reseeded
  (`tests/test_replay_divergence.py`).

**Loss** (`utils/loss.py`, `config.py`)
* The `_weighted_channel_mean` batch fix and `channel_weights`
  (`tests/test_channel_weights.py`).

**Trainer wiring** (`train.py`, `config.py`, `datasets.py`)
* `face_parallel` on `TrainConfig`, with up-front rejection of the
  combinations that cannot work (no replay, ungrouped, alongside
  `domain_parallel`, or `blend_before_backward=true`).
* The replay seed is shared across ranks when face-parallel, so every rank's
  planner draws the same row and the same seed time.
* `_build_face_replay_groups`: a group whose `dataset_indices` is only this
  rank's tiles while its `layout` is the whole face. That one choice is what
  makes `datasets_for`, `_entry_states`, `_seed_transitions_for_slot`,
  `_slot_input_span` and `_batch_wet_weight` rank-local with no edits, and
  lets `_reconcile_group_prediction` call the distributed blender unchanged.
* `TrainData.slice_batch` and `_replay_forward_backward`: the chunked
  forward/backward, with DDP syncing once on the last chunk of an
  accumulation cycle. One chunk reproduces the old path exactly.
* The divergence vote is collective (`FaceParallelContext.agree`), so one
  rank's runaway tile reseeds the whole face on every rank.

**Benchmark**
* `scripts/_bench/face_read.py` and the numbers in §2.

**Loss normalization** (`utils/loss.py`, `face_parallel.py`)
* `weighted_channel_denominator` and `GradientZNorms`/`gradient_z_norms`:
  denominators computed from the masks up front. They do not depend on the
  predictions, so every chunk divides by the same constant and the chunks then
  simply **sum** -- exact to 6.5e-08 through mse_mae + gradient_h + gradient_z
  + channel weights, with tiles that have different land and uneven splits
  (`tests/test_chunked_step.py`).
* `FaceParallelContext.global_loss_norms` all-reduces them once at startup, so
  every rank divides by the FACE's counts. The score no longer depends on
  which tiles a rank holds, which is what removes the ~3.4x reweighting the
  land-heavy rank would otherwise get -- and it means land does not have to be
  balanced across ranks, leaving the assignment free to chase chunk locality.
* The `/world_size` lands in different places for the two terms, because the
  base metric is one ratio and `gradient_z` is an average OF ratios. See the
  method docstring.

**Config and launch**
* `configs/data/llc_face1.yaml`, `configs/samudra_llc/train_replay_1face.yaml`,
  `JOBS/train_llc_replay-1_face.sh` (with `SMOKE=true`), and
  `JOBS/eval/eval_1_face.sh`.
* The 36 windows are generated from `TILE_OVERLAP` by the job script rather
  than listed in YAML, so the overlap is one knob instead of 36 rows that can
  drift out of agreement with it.

**Eval** (`tiled_eval.py`)
* Teaches the tiled evaluator the `data.llc_tiles` path. Without it the
  evaluator reopens the one store per tile and every tile inherits the store's
  full extent, collapsing the catalog into 36 identical tiles.

## 11. Verification

What is covered, and by what:

| Property | Test |
|---|---|
| 36 uniform windows, exact coverage, ownership partitions the face | `test_tiling.py` |
| Sharded blend == `TileBlender`, **bitwise**, world 1/4/9, both windows | `test_distributed_blend.py` |
| Chunk reader == per-tile read, synthetic **and** the real face cache | `test_chunk_reader.py` |
| Equal chunk counts, collective divergence vote, rank-independent denominators | `test_face_parallel.py` |
| Chunk losses sum to the whole-batch loss, all three terms | `test_chunked_step.py` |
| Per-tile divergence reporting | `test_replay_divergence.py` |
| Channel weights; the batch-normalization fix | `test_channel_weights.py` |
| **A whole face through the real `Trainer`**, seams agreeing after write-back | `test_face_training_e2e.py` |
| Training scores shares, validation scores means | `test_face_training_e2e.py` |

The end-to-end test is the one that earns its keep: it is the 45x-shrunk
version of the run that found the timestamp-decoding mismatch between the
reader and `DataSource`, which no unit test saw because each side was
self-consistent. That version needed 100 GB and OOMed a login node; this one
takes 15 s.

**Not covered by any test: the real `Trainer` on more than one rank.** DDP
setup goes through `init_train_backend`, which passes `device_ids=[gpu]` and
so needs CUDA -- a CPU gloo run of the whole trainer is not available. The
pieces are each verified multi-rank and the whole is verified single-rank, but
the combination is first exercised by the GPU smoke run.

### Smoke run

```bash
sbatch --export=ALL,SMOKE=true JOBS/train_llc_replay-1_face.sh
```

`SMOKE=true` cuts it to 2 epochs x 12 steps with a 2-row buffer. Watch for:

* `Face-parallel rank 0/4: tiles [...], 3 chunk(s) of [3, 3, 3]` on each rank
  -- the chunk counts must be identical or DDP will hang rather than fail.
* `Group frame reader: 9 tiles, N chunk decode(s) per frame ... would decode M`
  with M/N around 4-5.
* `Face loss normalization installed`.
* `max gpu mem` under ~130 GB, `replay_diverged_*` at zero, and
  `data_load_time` not climbing.

### Full run

```bash
sbatch JOBS/train_llc_replay-1_face.sh
```

Defaults: 4 GPUs, 9 tiles each in 3 chunks of 3, buffer 8, 2500 steps x 50
epochs. `GPUS` must divide 36 -- 4, 6, 9 or 12, **not** 8. If you change
`GPUS`, keep `FACE_READ_THREADS x GPUS` near 32.

## Open risks

* ~~**I/O is the schedule.**~~ Measured: 2.91 s per face-frame at 4×8,
  comfortably under the ~7.7 s compute at G=4. The production reader is held
  byte-for-byte to the per-tile read, including against the real cache, but
  its *speed* in the live loop is still only inferred from the benchmark.
* **Read concurrency is now a real tuning knob.** 4×15 is 2.7× slower than
  4×8 on the same work, so a well-meant `DATA_NUM_WORKERS` bump can cost more
  than it buys.
* Pinning ~67 GB is routine, but pinned allocation is slow at startup; watch the
  first-epoch timings.
* A per-rank divergence reseed desyncing the buffers is silent if the
  `all_reduce` in §4 is missed. Assert cursor equality across ranks periodically.
