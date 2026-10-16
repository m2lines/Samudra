# Coarsen LLC4320 to 1/4° and 1° (plan, 2026-10-15)

## Context

The 1/48° tile emulator can't run on the whole globe on one GPU, so it sees no
information from outside its 9-tile window. The plan is a second, coarse U-Net
that runs on the whole globe at 1/4° on one GPU. Its predictions are injected
into the 1/48° U-Net at its 3rd dilation level, giving each tile the large-scale
state and information from outside the tile.

This document covers only the data step. It builds:

1. **Coarse copies** of LLC4320 at 1/4° and 1°, for every hourly step. They hold
   all prognostic and boundary variables, with velocities turned to east/north,
   plus the static grid fields the coarse model needs.
2. **A 720×720-tile rechunk of the 1/48° 2D fields.** The raw store keeps these
   as one whole-globe chunk per step, so a tile loader currently decompresses
   ~970 MB per variable to get one 2 MB tile. The round-robin tile training
   needs them stored per tile.

Both are made by one script, `scripts/coarsen_llc.py`. It is restartable at the
level of single time steps and runs many steps in parallel.

## Decisions (agreed 2026-10-15)

- **Coarse boxes are plain index blocks on each face.** A 12×12 block of 1/48°
  pixels makes one 1/4° cell, and a 48×48 block makes one 1° cell. A face
  becomes 360×360 at 1/4° and 90×90 at 1°. A 720 tile becomes 60×60 and 15×15,
  so no coarse cell crosses a tile edge. "1/4°" and "1°" are nominal, since LLC
  spacing varies with latitude.
- **Store native faces and stitch at load time.** Coarse arrays keep a `face`
  axis, shape (13, 360, 360). A small helper rotates faces 7–12 and stitches
  the 4 side-by-side facets into a lat/lon grid when data is loaded, which is
  cheap at coarse size. The multi-day pass therefore doesn't lock in an Arctic
  choice.
- **Velocities are east/north in every face**, including the Arctic cap. U/V
  and oceTAUX/oceTAUY are moved to cell centres and rotated with CS/SN before
  coarsening, using the method of `rotate_to_geo` / `interp_to_center` in
  `LLC_ocean_emulator/llc_cross_face.ipynb`.
- **Face edges copy the edge value.** The last column of U (and last row of V)
  of each face reuses its own value instead of the neighbouring face's, the
  same as the notebook's `boundary="extend"`. Inside a face, tile edges use the
  true neighbouring values.
- **Coarse values are float32 in physical units.** Land is NaN. Normalization
  happens at load time, using stats computed from the coarse data (the `stats`
  mode below).
- **Averaging weights are `rA` over wet pixels.** A coarse cell is ocean at a
  level if at least one of its 1/48° pixels is wet at that level. Its value is
  the rA-weighted mean of the wet pixels. Partial bottom cells count as full
  cells, as asked ("divide by rA").
- **The rechunk is an exact float32 copy** of Eta, oceTAUX, oceTAUY, oceQnet and
  oceFWflx, with one array per variable. It also stores the 1/48° `hFacC_2d`
  so the coarse and 1/48° models share it.
- **`hFacC_2d` is the open-water fraction of each column's bottom cell.** It is
  hFacC at the deepest wet level among the 51. A column wet through all 51
  levels gets 1, a partial bottom cell gets its hFacC, and land gets 0.
- **Missing forcing steps are filled from their neighbours.** The raw store
  lacks ~60 isolated hourly steps of oceQnet/oceTAUX/oceFWflx (59 of oceTAUY),
  all multiples of 12 and including t=0. Each is filled with the mean of steps
  t−1 and t+1 (t=0 copies t=1). The filled indices, per variable, are recorded
  in the attrs of both the coarse and rechunk stores.
- **Land fraction is per level and replaces the separate masks.** `wet_frac`
  (51, 13, n, n) is the coarse grid's own hFacC: the rA-weighted mean of the
  1/48° hFacC over the box at each level.
  - 1 = all water, 0 = all land, in between where coasts and shelves slope down
    through the levels or a box has partial bottom cells. Same direction as
    hFacC and `mask_c` (water = 1).
  - The binary coarse mask is `wet_frac > 0`, which is the "keep a box with at
    least one ocean pixel" rule. The old surface land fraction is
    `1 − wet_frac[0]`. Both are derived at load time, so no separate
    `land_frac`, `land_mask` or `mask_3d` arrays are stored.
  - At 1/48° the same field is just `hFacC`. The raw store already chunks it as
    (51, 1, 720, 720), one chunk per tile, so the rechunk doesn't copy it. Tile
    loaders read it from the raw store, and `mask_c` is `hFacC > 0`.
  - It is static and written once (344 MB at 1/4° in float32), so it stays
    float32.
- **Static fields are stored once**, not per time step.
- **3D and 2D stay in two arrays** (answering "can we combine them in one
  chunk?"). They come from two different passes (below). One array would force
  both passes to write the same chunk, which breaks independent restarts.
  Loading one step is two reads: 1.37 GB + 34 MB at 1/4°.
- **Position/area channels are built at load time.** The four channels
  (cos·cos, cos·sin, sin lat, log(rA/1e6)) are not stored. The loader builds
  them from the stored coarse XC, YC, rA with the existing
  `_packed_spatial_features` (`src/ocean_emulators/utils/data.py:146`), the same
  as the 1/48° model.

## Source and destination

```text
source (read-only, zarr v2, no consolidated metadata -> consolidated=False)
  /orcd/data/abodner/003/LLC4320/LLC4320
    3D  Theta, Salt (j,i), U (j,i_g), V (j_g,i)   [10311, 51, 13, 4320, 4320]  chunks (1,51,1,720,720)
    2D  Eta, oceQnet, oceFWflx, oceTAUX (i_g), oceTAUY (j_g)  [10311, 13, 4320, 4320]  chunks (1,13,4320,4320)
    grid XC YC rA dxF dyF Depth CS SN [13,4320,4320]; hFacC, mask_c [51,13,4320,4320]

coarse (scratch; reach it only from node4006-type nodes, never node4005)
  /orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_0p25deg.zarr
  /orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_coarse_1deg.zarr
  /orcd/scratch/seedfund/001/llc_emulator/coarse/_progress/{boundary,prognostic}/<t:05d>

rechunk (data disk)
  /orcd/data/abodner/002/cody/LLC_boundary_rechunk/LLC4320_2d_720.zarr
```

Time is 10311 hourly steps, indices **0–10310**, from 2011-09-13T00 to
2012-11-15T14. The train/val split matches the face-1 cache: steps 0–8807 are
train and 8808–10310 are val. Both are recorded in attrs.

## Output layout

**Coarse store** (one per resolution; n = 360 at 1/4°, 90 at 1°). All arrays
are float32 with NaN fill unless noted, and blosc zstd with bitshuffle, like the
raw store.

| array | shape | chunks | contents |
|---|---|---|---|
| `vars_3d` | (10311, 204, 13, n, n) | (1, 204, 13, n, n) | U_0..50 (east), V_0..50 (north), Theta_0..50, Salt_0..50 |
| `vars_2d` | (10311, 5, 13, n, n) | (1, 5, 13, n, n) | Eta, oceTAUX (east), oceTAUY (north), oceQnet, oceFWflx |
| `XC`, `YC` | (13, n, n) | whole | box centre (area-weighted mean of the unit vector, converted back to lon/lat) |
| `rA` | (13, n, n) | whole | box area = sum of all 1/48° rA in the box |
| `dxF`, `dyF` | (13, n, n) | whole | box width/height along the face's own i/j axes (mean row/column sum) |
| `Depth`, `hFacC_2d` | (13, n, n) | whole | rA-weighted mean over surface-wet pixels |
| `wet_frac` | (51, 13, n, n) | (51, 1, n, n) | coarse hFacC: rA-weighted box mean of 1/48° hFacC at each level; mask = `wet_frac > 0` |
| `time` | (10311,) int64 | (1024,) | copied from the raw store |
| `vars_{3d,2d}_{mean,std}` | (204,) / (5,) | whole | written later by `stats` mode |

Attrs: channel names, box size, source path, `velocity_frame: east/north`,
`face_edge_velocity: edge value copied`, `filled_time_steps` (per variable),
train/val split, and `built_by`.

**Rechunk store.** Each of Eta, oceTAUX, oceTAUY, oceQnet and oceFWflx is its
own array with the raw dims (time, face, j, i / i_g / j_g). Each has shape
(10311, 13, 4320, 4320), chunks (1, 1, 720, 720), float32, and the raw store's
compressor. `dimension_separator="/"` keeps directories small, and
`write_empty_chunks=False` skips all-land tiles. The store also holds static
`hFacC_2d` (13, 4320, 4320) with chunks (1, 720, 720), a copy of `time`, and
`filled_time_steps` in attrs. Values are raw and unrotated, the same as the
source. The 3D hFacC (the 1/48° `wet_frac`) is not copied, because the raw
store already chunks it per 720 tile.

**Sizes**

| output | per step (uncompressed) | total uncompressed | notes |
|---|---|---|---|
| 1/4° coarse | 1.41 GB (3D 1.37 GB + 2D 34 MB) | ~14.5 TB | 1.37 GB chunk is under blosc's 2 GB limit |
| 1° coarse | 88 MB | ~0.9 TB | |
| rechunk | 4.85 GB | ~50 TB (~20 TB compressed, estimated) | |

Free space: scratch 64 TB, `/orcd/data/abodner/002` 68 TB (94% used).
The rechunk is ~24M chunk files, minus all-land tiles. Check the inode quota
before running (Phase 0).

## How each value is computed

For one face and one tile, with `w = rA * mask_c[k]` (2D fields use level 0):

```text
num_q  = box_sum(x * w, 12)        # 60x60 per tile, NaN treated as 0, summed in float64
wsum_q = box_sum(w, 12)            # static, float64
x_q    = num_q / wsum_q            # NaN where wsum_q == 0 (all land)
x_1    = box_sum(num_q, 4) / box_sum(wsum_q, 4)   # 1° from the 1/4° sums: no second read
```

The sums stay float64 until the final float32 cast, so the 1° result matches a
direct 48×48 average up to float rounding.

**Guard against silent zeros.** A missing raw chunk reads back as NaN, and step
1 of the velocity recipe would turn it into 0. So every tile checks three things:

- Theta's wet boxes (`den > 0`) equal `wet_frac > 0`.
- Salt's NaNs equal Theta's.
- A U/V tile whose coarse boxes include any wet cell must have its chunk file
  on disk.

An all-NaN U/V tile is not itself an error. Isolated lakes have wet cells but
no open faces; Lake Victoria (face 1, tile (3600, 2880)) is one.

**Velocities** (U/V in the prognostic pass, oceTAUX/oceTAUY in the boundary
pass), for a whole face at a time:

1. Set NaN to 0. Velocity through a land face is zero.
2. Centre: `uc[i] = (U[i] + U[i+1]) / 2` and `vc[j] = (V[j] + V[j+1]) / 2`.
   At the face's last column/row, copy the edge value.
3. Rotate per pixel: `east = uc*CS - vc*SN` and `north = uc*SN + vc*CS`.
4. Coarsen with the centre wet mask, as for any other field.

**Static fields** come from the grid in `init` mode. `rA`, `dxF`, `dyF`, `XC`
and `YC` use every pixel in the box, since grid geometry doesn't depend on
land. `Depth` and `hFacC_2d` use the surface-wet rule above. `wet_frac[k]` is
`box_sum(rA * hFacC[k], b) / box_sum(rA, b)`.

**Stitching helper** (used by loaders, not by the pass):
`faces_to_latlon(a) -> (facets, cap)`.

- Columns run west→east: faces [0,1,2] | [3,4,5] | [9,8,7] | [12,11,10], each
  facet stacked south→north. The facets span roughly −38→52, 52→142,
  142→232 and 232→322° E.
- Faces 7–12 are turned with `np.rot90(k=1)`, as in notebook cell 8, and their
  `dxF`/`dyF` are swapped.
- Every facet's northern edge is at 67.5°N. The Arctic cap (face 6) is returned
  separately for now.
- This layout and rotation direction were checked against the raw XC/YC face
  corners on 2026-10-15.

## Script: `scripts/coarsen_llc.py`

Modes:

- **`init`**: run once, single process.
  - Refuse to run if any output store already exists, unless given `--force`.
    The array-creation helper overwrites, and a second `init` would erase days
    of chunks while the markers still say "done".
  - Create both coarse stores and the rechunk store. The rechunk arrays get
    `dimension_separator="/"`, which must be set at creation, so add that
    argument to `create_array`.
  - Compute and write all static fields: coarse statics, and the 1/48°
    `hFacC_2d`.
  - Assert CS² + SN² ≈ 1 wherever `mask_c[0]` is 1. CS and SN are 0 on some
    all-land tiles.
  - Scan the raw store for missing forcing chunks. A step is missing when its
    `<var>/<t>/` directory is absent: 60 steps each for oceQnet, oceTAUX and
    oceFWflx, 59 for oceTAUY, none for Eta. Check that none are back-to-back,
    then write `filled_time_steps`.
  - Copy `time` with its raw attrs (units "seconds since 2011-09-10",
    calendar).
- **`fill --part boundary`**: for each pending step t:
  1. Read the 5 whole-globe 2D chunks. Fill missing variables from t±1.
  2. Write the 468 tiles of each variable to the rechunk store.
  3. Centre and rotate oceTAUX/oceTAUY per face, then coarsen all 5 fields.
  4. Write `vars_2d[t]` to the 1° store, then to the 1/4° store, then create
     `_progress/boundary/t`.
- **`fill --part prognostic`**: for each pending step t, for each face:
  1. Read the whole face of U and V (36 tiles each) so in-face neighbours are
     exact. Set NaN to 0 and centre in place, one level at a time. Doing all
     levels at once with overlapping slices makes numpy allocate a hidden
     3.8 GB copy.
  2. For each of the 36 tiles:
     - rotate that tile;
     - read the Theta and Salt tiles;
     - coarsen all four fields into the step's buffers.

  Then write `vars_3d[t]` to the 1° store, then the 1/4° store, then create
  `_progress/prognostic/t`.
- **`stats`**: per-channel area-weighted mean/std over 96 evenly spaced train
  steps (0–8807), written into each coarse store.
- **`verify`**:
  - Count markers (10311 per part).
  - Check that every step's coarse chunk files exist.
  - Delete leftover `<chunk>.<uuid>.partial` files from killed writes in the
    coarse stores. Run it only when no fill is running, since in-flight writes
    use the same names. Leftovers in the rechunk store are harmless, because
    zarr never reads them, and walking ~24M files to remove them isn't worth it.
  - Check shapes and chunks.
  - Re-run the conservation check on a few random steps.
  - Consolidate metadata.

**Restart:**

- A step is done only when its marker exists. The marker is written after every
  write for that step.
- Zarr writes each chunk to a temp file and renames it into place, so a killed
  write never leaves a half-written chunk. A killed run just redoes its
  unfinished steps, which overwrite the same chunks.
- Fill workers reopen arrays with
  `zarr.open_array(..., mode="r+", write_empty_chunks=False)`. The setting is
  not saved in the metadata, and the default writes every all-land tile.
- `fill` never consolidates metadata, so array tasks can't race, the same rule
  as in `build_llc_face_cache.py`.

**Parallelism**

- One process per time step (`--workers W`), each with a thread pool for chunk
  reads (`--read-threads T`).
- Blosc's own threads are off and `OMP_NUM_THREADS=1`, as in the face-cache
  builder.
- Several nodes can split the time axis with `--time-splits N
  --time-split-index i`, the existing pattern.
- `--stride S --offset O` processes steps with t ≡ O (mod S) first.
  `S=24, O=6` gives a daily subset early and avoids the filled forcing steps,
  which are all multiples of 12. Later runs fill in the rest.
- Prognostic worker peak is ~15 GB: the U and V face arrays (7.6 GB), tile
  scratch, and the 1.37 GB output buffer plus its compressed copy. A 376 GB
  node fits about 16 workers, and the benchmark picks W.

**Job script:** `JOBS/coarsen_llc.sh`, copied from
`JOBS/other/old/build_llc_face_cache.sh`.

- Env vars: `MODE`, `PART`, `WORKERS`, `READ_THREADS`, `STRIDE`, `OFFSET`,
  `STEPS`, `LIMIT`, `COARSE_ROOT`, `RECHUNK_ROOT`, `FORCE`.
- Slurm: `-p pi_abodner`, `--cpus-per-task=120`, `--mem=320G`, 48 h,
  `-x node4005`. 320G rather than 340G so the job still fits next to a 40 GB
  interactive session on node4006 (377 GB).
- Run python with `.venv/bin/python`, never `uv run`.
- Before any work, run `timeout 20 ls` on the scratch path.
- To finish, resubmit until `verify` passes.

## Reuse

- `scripts/build_llc_face_cache.py`: `open_source_array`, `SourceArray`,
  `create_array` (its fill-value rules, plus a new `dimension_separator`
  argument), the time-split logic and the blosc/thread settings. Import them by
  module name; the script's directory is on `sys.path`.
- `scripts/add_boundary_channels.py`: the init → fill → finalize shape.
- `src/ocean_emulators/utils/data.py`: `_packed_spatial_features`, for the
  position/area channels at load time.
- `llc_cross_face.ipynb`: the formulas in `interp_to_center` /
  `rotate_to_geo`, ported to numpy. xgcm is not installed in `.venv`.
- New pure functions go in `src/ocean_emulators/llc_coarse.py`, so training
  code can import `faces_to_latlon` later: `box_sum`, `weighted_box_sum`,
  `ratio`, `center_inplace`, `rotate_to_east_north`, `bottom_hfac`,
  `unit_vector`, `lonlat_from_vector`, `fill_sources`, `faces_to_latlon`.

## Phases

0. **Preflight.** Run `hostname`, then `timeout 20 df -h` / `df -i` on both
   output disks and check the inode quota for ~24M files.
1. **Helpers and unit tests:** `llc_coarse.py` and `tests/test_llc_coarse.py`.
2. **Script and job.** Run `init` for real. It is fast and only touches grid
   fields.
3. **One-step check** on t=1 and t=288 (a filled step) into a test output root,
   using Verification items 2–3.
4. **Benchmark.** Run 20 steps of each part with W = 2, 4, 8, 12 and record
   s/step and GB/s, then fix W.
5. **Boundary pass**, all steps (hours).
6. **Prognostic pass.** Run `--stride 24 --offset 6` first, so coarse-model
   work can start, then all steps (days).
7. **`stats`, then `verify`.**

## Verification

1. **Unit tests** (`.venv/bin/python -m pytest tests/test_llc_coarse.py`):
   - Coarsening keeps Σ(x·w).
   - 1° built from 1/4° sums equals a direct 48×48 average.
   - An all-land box gives NaN and `wet_frac = 0`. A box with one full wet
     pixel (equal areas) returns that pixel's value, with `wet_frac = 1/144`.
   - With box size 1, `wet_frac` equals hFacC.
   - On a synthetic sloping shelf, `wet_frac` decreases with depth level.
   - Centring: interior values are the mean of two neighbours, the face edge
     is copied, and NaN becomes 0.
   - Rotation: CS=1, SN=0 is the identity. CS=0, SN=−1 (faces 7–12) gives
     east = v_local and north = −u_local.
   - Fill gives the mean of t±1, and t=0 copies t=1.
   - `bottom_hfac`: a deep column gives 1, a partial bottom cell gives its
     hFacC, and land gives 0.
   - `faces_to_latlon` on the real coarse XC/YC, where YC > −70:
     - every row's longitude increases eastward with one wrap;
     - every column's latitude increases northward;
     - `dxF`/`dyF` are swapped on faces 7–12.

     South of about −78° the grid stops being lat/lon over Antarctic land,
     so those rows are left out of the test.
2. **One real step, end to end:**
   - Shapes, chunks and dtypes are as in the tables, with one chunk per array
     per step.
   - Conservation against raw: for each level, Σ(Theta·rA) over raw wet pixels
     matches Σ(coarse Theta × coarse wet area) (`allclose`, float64 sums).
     Same for Eta.
   - Hand-check one coarse cell each on face 1, face 9 and the Arctic cap.
   - Raw Theta NaNs equal `mask_c == 0` on sampled tiles (the wet-mask
     assumption).
   - Stitched surface U/V are continuous across the 11|1, 2|5 and 5|7 seams,
     like the notebook's seam check.
   - Save a quick-look PNG of stitched Theta, U, V, Eta, and `wet_frac` at
     levels 0 and 40.
   - Coarse NaNs in `vars_3d` level k exactly match `wet_frac[k] == 0`.
3. **Rechunk:**
   - At t=1, every tile of every variable equals the slice of the raw globe
     chunk, bit for bit.
   - At t=288, values equal the mean of 287 and 289, and the step is listed
     in both stores' attrs.
4. **Restart:**
   - Kill a fill after a few steps and rerun. It skips marked steps and redoes
     the unfinished one.
   - Those steps are identical to an uninterrupted run.
5. **Final `verify`:**
   - 10311/10311 markers per part.
   - Random-step conservation checks pass.
   - Metadata is consolidated.

## Expected cost

| pass | read per step | total read | estimated time |
|---|---|---|---|
| boundary | ~2 GB (5 globe chunks × ~400 MB) | ~20 TB (+ ~20 TB written) | ~4–8 h |
| prognostic | ~72 GB (1872 tiles × ~38 MB average, measured at t=5000) | ~740 TB | ~2.8 days at 3 GB/s |

The 3 GB/s figure is the filesystem ceiling measured by the face-cache build
(~1 GB/s per task). The prognostic pass is limited by reads, so more workers
help only until that ceiling. If several nodes together go past it, split the
time axis across them.

**Measured on 2026-10-16, the ceiling was much lower.** The raw store is
served over NFS by a single server (`hstor002-n2`). It delivered:

- 0.6–0.75 GB/s to one node;
- about 1.4 GB/s in total to three nodes reading at once (0.47 GB/s each).

At those rates the prognostic pass takes about **11 days on one node** and
about **6 days on three nodes**. See Implementation status.

## Later, not in this script

- How the coarse model handles the Arctic cap (`faces_to_latlon` returns it
  separately for now).
- A loader for the coarse store, and injection at the 1/48° U-Net's 3rd
  dilation level.
- Training the coarse models a/b/c (full year, first half, second half).

## Definition of done

- Both coarse stores and the rechunk store pass `verify`, with all 10311 steps
  done for both parts.
- Unit tests pass.
- The one-step conservation, seam and rechunk checks are recorded in an
  "Implementation status" section at the end of this doc.

## Implementation status (2026-10-16)

Built:

- `src/ocean_emulators/llc_coarse.py`: pure helpers.
- `tests/test_llc_coarse.py`: 11 tests, all passing. One checks the stitch
  layout against the real XC/YC.
- `scripts/coarsen_llc.py`: modes `init`, `fill`, `stats`, `verify`.
- `scripts/check_coarsen_llc.py`: the one-step checks below.
- `JOBS/coarsen_llc.sh`: the production job.
- `JOBS/other/bench_coarsen_llc.sh`: the throughput sweep.
- `create_array` in `scripts/build_llc_face_cache.py` gained a
  `dimension_separator` argument.

The preflight (Phase 0) was skipped: write access for ~24M files is still
pending.

All test output went to `/orcd/scratch/seedfund/001/llc_emulator/coarse_test`,
with its rechunk store under `coarse_test/rechunk`. The production paths were
not touched.

**init:** 110 s on 2 CPUs, 2.8 GB peak. Missing 2D steps: oceTAUX 60,
oceTAUY 59, oceQnet 60, oceFWflx 60, Eta 0, all isolated. The fine
`hFacC_2d` is 1 (full column) on 49% of pixels, partial on 8.5% and land on
42%.

**One-step checks:** 0 failures for boundary steps 1, 288 (filled) and 6
(redone after a kill), and for prognostic steps 1 and 2000.

- **Rechunk:** every tile equals the raw globe chunk bit for bit. On t=288 it
  equals the mean of steps 287 and 289, and the step is listed in all three
  stores' attrs.
- **Conservation** (relative error):
  - Eta: ≤ 1.3e-9 on faces 1, 6 and 9.
  - Theta, all levels: ≤ 1.5e-7 on faces 1, 6 and 9. The Arctic face is the
    largest, from cancellation with near-zero temperatures.
- **Hand checks:** U/V (k = 0, 20) and wind-stress east/north at a tile-corner
  cell on faces 1, 6 and 9 match an independent numpy recomputation (rtol
  1e-4). That cell uses the next tile's U column and V row.
- **NaN patterns:** raw Theta NaNs equal `mask_c == 0` on 9 sampled tiles.
  Coarse NaNs equal `wet_frac == 0` at every level, for both resolutions.
- **Seams:** across the 11|1, 2|5, 5|7 and 8|7 seams, the stitched surface U/V
  jump is 0.6–1.3× the jump between neighbouring cells inside a face.
- **Quick-look:** the stitched Theta, U, V, Eta and `wet_frac` look physically
  right. The equatorial Pacific current flows west and the Antarctic
  Circumpolar Current east, with no visible facet edges.

**Guard change found by the first run:** the "all-NaN U tile on a wet tile"
check fired on face 1, tile (3600, 2880). That tile holds a single wet column
of Lake Victoria with no open U faces, so U is legitimately all NaN there. The
guard now checks that the chunk file exists, which is the actual failure mode.
Isolated lakes count as ocean under the "≥ 1 wet pixel" rule, with tiny
`wet_frac`.

**Restart:** I killed a 3-step boundary fill (main process and workers) during
its second step. Only the first step's marker existed afterwards. The rerun
skipped it, redid the other two, and the redone step passes every check.

**Throughput** (120-CPU job on node4006; `s/step wall` is the time per finished
step across all workers):

| part | workers × threads | s/step wall | read GB/s | worker peak |
|---|---|---|---|---|
| boundary | 2 × 36 | 6.0 | 0.33 | 12 GB |
| boundary | 4 × 30 | **2.6** | 0.83 | 12 GB |
| boundary | 8 × 15 | 3.1 | 0.64 | 12 GB |
| boundary | 12 × 10 | 5.5 | 0.36 | 11 GB |
| prognostic | 2 × 36 | **95** | 0.75 | 32 GB |
| prognostic | 4 × 30 | 118 | 0.61 | 29 GB |

- **Per-step time split (prognostic, 2 workers):** read U/V 98 s, centring
  11 s, tiles (Theta/Salt reads + maths) 63 s, write 13 s per worker step.
- **Centring fix:** the first version centred U/V level by level in one thread,
  which took ~150 s per step. It now runs one level per thread.
- **Plain `cat` agrees:** 64 parallel `cat`s give 0.72 GB/s from one node and
  0.47 GB/s per node from three nodes at once. So the limit is the raw store's
  server, not this code.
- **W=8 and W=12 prognostic weren't run.** W=4 was already slower than W=2.
- **Caveat on the boundary rows:** they were measured while a small restart
  test also ran. They were also writing to scratch (fstor012), not to
  `/orcd/data/abodner/002` (hstor009).

**Projected run time at these rates:**

- Boundary pass: ~7.4 h on one node.
- Prognostic daily subset (430 steps): ~11 h on one node.
- Full prognostic pass: ~11 days on one node, or ~6 days split across three
  nodes (`--array=0-2`).

The job script defaults to 4×30 for boundary and 2×32 for prognostic. For the
prognostic part, submit with `-c 64 --mem=128G`.
