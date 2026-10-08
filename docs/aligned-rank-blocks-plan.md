# Chunk-aligned rank blocks plan

## Context

Rank-local training (`face_parallel.blend_scope=rank`) builds each 3x3 block
from the global face windows of `face_tile_windows(1)`, whose origins are
`0, 704, 1424, 2144, 2864, 3568`. A block's union is therefore 2192² in the
interior, so it straddles the 720² chunk grid by 16 px on every side:

| Block position | Blocks | Chunks per array today | After |
| --- | ---: | ---: | ---: |
| face corner | 4 | 16 | **9** |
| face edge | 8 | 20 | **9** |
| interior | 4 | 25 | **9** |
| uniform mean | 16 | 20.25 | **9** |

The outer chunk ring only contributes a 16-px sliver, but Blosc/zstd chunks
must be read and decoded whole. In the latest rental run, reading takes about
6.4 s of a 7.5 s step (p50), and read stalls dominate the tail. Cutting decodes
by **56%** is the main throughput lever.

The pre-cut 36-store recut (`docs/face-recut-plan.md`) is rejected.

## Decision

Keep every tile 752². Clamp the nine windows inside the chunk-aligned 2160²
block instead of inside the face. That is option 2. One axis, in
block-relative pixels:

```text
chunks      |   c   |  c+1  |  c+2  |
            |<------ 2160 ------->|
lo   [0 ......... 752)
mid         [704 ......... 1456)
hi                  [1408 ......... 2160)
seams lo|mid = 48, mid|hi = 48;  3 chunks/axis -> 9 per array
```

`face_tile_windows(face, extent=2160)` already returns `[0, 704, 1408]`
through `tile_origins`. No new clamping code is needed.

What changes for training:

- Every in-block seam is 48 px. Today interior seams are 32 and face-edge
  seams 48, and 12 of the 16 blocks already contain 48 seams. The blender
  derives widths per `(tile, side)`, so it needs no change.
- The block edge is the tile edge, with no 16-px read beyond the block. The loss
  already covers the full 752² window, so each tile's task is unchanged.
- The centre tile's window is unchanged. Along an axis where a window touches
  a face edge, its position is unchanged too.
- It is the same 16 blocks in the same order: new block `(r, c)` covers the
  tiles of old centre `(r+1, c+1)`.
- All 16 blocks share one geometry, so one blender replaces nine.
- Validation and inference are unchanged. The offload validator passes its own
  `face_tile_windows` and forces `blend_scope=face`, and `tiled_eval` uses
  face windows.

Costs and constraints:

- There are 144 replay sources (16 blocks x 9) instead of 36. Pinned wet masks
  grow from 4.2 to about 16.7 GB per rank, about 117 GB on a 7-rank node with
  2 TB of RAM.
- The loss denominator becomes the mean block wet count:
  `face_wide_loss_norms(face_tiles=144, rank_tiles=9)` already computes it.
  It is near-equal to today's 9/36 face share, so loss curves shift slightly.
- `blend_scope=rank` now requires `validation_mode=offload`, which production
  already uses.
- Resume is clean because the rental sets `--replay.checkpoint_buffer false`,
  so rows reseed in the new geometry.

## Phase 0: port the rank path into the main checkout, then clean up

Do this before any code changes. Cody makes the commits. The goal is for
`/home/codycruz/Ocean_Emulator` to be the only 1-face training checkout, so the
async worktree can be deleted.

**Gate before step 3:** no job may be running from the main checkout. Eval
25162709 runs `cd /orcd/home/002/codycruz/Ocean_Emulator` and lazily imports
from `src/`. Check `squeue -u codycruz` first. Steps 1 and 2 don't touch
`src/`, so they are safe while it runs.

1. **Recut WIP.**
   - Save `git diff scripts/recut_llc_patch_caches.py` and the untracked
     `tests/test_recut_llc_patch_caches.py` to the scratchpad as a backup.
   - Then `git checkout -- scripts/recut_llc_patch_caches.py` and delete the
     untracked test.
   - Add a one-line "superseded" banner to `docs/face-recut-plan.md`.
2. **Main's other WIP** is the multi-GPU `tiled_eval.py`,
   `TiledInferenceConfig.resume` and the `repack` Eta fix, which is identical
   in both checkouts. Cody commits it so the merge starts from a clean tree.
3. **Merge.** In main, run
   `git merge --no-ff --no-commit asynchronous-train_blending`, which brings
   the four commits `cb0a2dbb..fc39c5b3`.
   - Expected conflicts: `train.py`, `config.py`, `utils/schedule.py`,
     `tests/test_schedule.py`. Both sides added LR warmup and schedule code.
   - Keep main's `1e935a5a` rampup wiring and curriculum offset, plus async's
     offload pieces.
   - Cody commits the merge.
4. **Async WIP.**
   - Run `git -C <async> diff > wip.patch`, then `git apply --3way wip.patch`
     in main. Expect conflicts in `train.py` and `tests/test_replay.py`.
   - Drop the pre-cut reader from it: `_precut_read_executor`, its lock,
     `_precut_replay_reader` and its shutdown. The per-tile fallback goes back
     to plain sequential reads.
   - Cody commits.
5. **JOBS** (gitignored):
   - Set `REPO=` to the main checkout in `train_llc_1face_8xh200_rental.sh`,
     `-validation.sh` and `other/test_8xh200_preflight*.sh`.
   - Remove the `DATA_LAYOUT=precut` branch.
   - Delete `other/1-face_recut.sh`, `other/submit_1-face_recut*.sh` and
     `other/test_1-face_recut_async.sh`.
   - Keep the older 4-tile `other/recut_llc_patch_caches.sh`.
6. **Verify the port.**
   - The full suite passes in main.
   - The diff of each `src/` and `tests/` file against the async worktree shows
     only the main-only commits and the pre-cut removal.
   - Then `git worktree remove --force` the async worktree, after Cody
     confirms. Keep the branch until Cody deletes it.

## Phase 1: aligned block windows

All edits are small and reuse existing paths. Comments stay one line.

1. **`tiling.py`**: add `block_tile_windows` next to `face_tile_windows`.

   ```python
   def block_tile_windows(face, *, extent=4320, tile=720, overlap=16, block=3):
       """Windows of every chunk-aligned block x block tile group, block-major."""
       span = block * tile
       inner = face_tile_windows(face, extent=span, tile=tile, overlap=overlap)
       starts = range(0, extent - span + 1, tile)
       return [
           (f, i0 + bi, i1 + bi, j0 + bj, j1 + bj)
           for bj in starts
           for bi in starts
           for f, i0, i1, j0, j1 in inner
       ]
   ```

2. **`face_parallel.py`**, `build_block_replay_groups`:
   - Take consecutive `block**2` tiles of a block-major catalog.
   - Drop the `grid` parameter.
   - Delete `interior_centers` and `block_tiles`.
   - Trim the `_overlap_signature` docstring; the dedupe now finds one blender.
3. **`train.py`**:
   - `_build_face_replay_groups`: in rank mode, raise unless
     `validation_mode=offload`, then return `_build_block_groups(catalog)`.
     Delete the rank-plus-inline branch.
   - `_build_block_groups`: replace the `isqrt` square-face check with
     `len(catalog) % 9 == 0`.
   - `_build_group_frame_reader`: in rank mode, raise if any block's
     `GroupFrameReader.chunks_for(windows)` exceeds 18 (9 per array). This is
     the existing, unused helper. Log the per-block count once.
   - Fix comments that hard-code "36 masks" or "4.2 GB".
4. **Launcher**: when `blend_scope=rank`, build `LLC_TILES` with
   `block_tile_windows` instead of `face_tile_windows`.
5. **Docstrings**: update the 16-chunk 3x3 claim in the `chunk_reader.py`
   module docstring.
6. **Tests**:
   - `tests/test_tiling.py`: 144 windows. Each block's union is an aligned
     `3*tile` square, `chunk_plan` gives 9 chunks, every overlap is 48, and the
     centre windows match `face_tile_windows`.
   - `tests/test_face_training_e2e.py`: `_block_config` uses
     `block_tile_windows(1, extent=96, tile=16, overlap=2)` plus offload.
   - Replace the centre test with a consecutive-nines and alignment test.
   - Replace `test_validation_still_covers_the_whole_face` with
     "rank plus inline raises".
   - Update the denominator test to the 9/144 share.

## Verification gate

1. After Phase 0 and again after Phase 1,
   `uv run pytest -m "not manual and not cuda" -n auto` passes in main.
2. A geometry check on face 1 shows, for all 16 blocks: a 2160² union at
   720-multiple origins, 9 prognostic and 9 boundary chunks, overlaps `{48}`,
   and one shared blender.
3. Preflight, which needs Cody's go-ahead to launch:
   `JOBS/other/test_8xh200_preflight.sh` from main, crossing one replay refresh.
   The log must show:
   - 18 chunk decodes per frame per block;
   - 144 per-tile wet masks on the CPU (about 16.7 GB);
   - `data_load_time` p50 well under the 6.4 s baseline;
   - no mask or source errors.
4. The offload validator still reports 36 tiles with overlaps `[32, 48]`.

## Definition of done

- The main checkout is the only 1-face training setup, the launchers point at
  it, and the async worktree is removed.
- Rank blocks decode 9 chunks per array, as logged and asserted.
- The suite passes, and the preflight shows a reduced `data_load_time`.

## Implementation status (2026-10-07)

- Phase 0 step 1 done: recut WIP backed up and discarded, recut plan marked
  superseded.
- Step 2 committed (`58c5245e`). The eval gate cleared once 25162709 ended.
- Step 3 merge is staged with conflicts resolved:
  - Warmup: async's `_remove_lr_warmup` (strips the step factor before
    `scheduler.step()`), and its bounds-checked warmup fields.
  - Main's `curriculum_epoch_offset` is kept.
  - The non-GPU suite fails only the 152 tests that already fail on `58c5245e`.
- Waiting on Cody to commit the merge before step 4. The async WIP patch
  dry-runs cleanly, excluding the repack script, which is already in main.
