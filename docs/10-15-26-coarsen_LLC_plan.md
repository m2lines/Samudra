# Coarsen LLC4320 onto the Samudra grid at 1/4° and 1° (plan, 2026-10-15, revised 2026-10-20)

> **Revised 2026-10-22:**
>
> - bad or missing raw data copies the previous clean hour, logged in
>   `repairs.log` and the `repaired` arrays;
> - a native 1/3° product is added;
> - the rechunk is back on `/orcd/data/abodner/002`;
> - array tasks take every N-th hour.
>
> **Revised 2026-10-20.** Coarse output is now regridded onto Samudra's OM4
> lat/lon grid. The old version used box averages on the native faces.
> Other changes:
>
> - one array per variable;
> - one merged pass per hour;
> - zstd-3 byte-shuffle compression everywhere;
> - a 3D `land_frac`.
>
> The 2026-10-16 native-face build and its measurements are summarised under
> History at the end.

## Context

The 1/48° tile emulator can't run on the whole globe on one GPU, so it sees no
information from outside its 9-tile window. The plan is a second, coarse U-Net
that runs on the whole globe at 1/4° on one GPU. Its predictions are injected
into the 1/48° U-Net at its 3rd dilation level. This gives each tile the
large-scale state and information from outside the tile.

This document covers only the data step. It builds:

1. **Coarse copies** of LLC4320 on Samudra's grid at 1/4° and 1°, for every
   hourly step. They hold all prognostic and boundary variables, with
   velocities turned to east/north. They also hold the static fields the
   coarse model needs. Every coarse cell has the same centre as the
   corresponding Samudra (OM4) cell, so the two datasets line up.
2. **A 720×720-tile rechunk of the 1/48° 2D fields.** The raw store keeps these
   as one whole-globe chunk per step, so a tile loader currently decompresses
   ~970 MB per variable to get one 2 MB tile.

Both are made by one script, `scripts/coarsen_llc.py`. Each unit of work is one
global hour, the script restarts at the level of single hours, and it runs
several hours in parallel.

## Decisions

Agreed 2026-10-15, revised 2026-10-20.

- **Target grid: Samudra's OM4 1° grid**, read from
  `/orcd/data/abodner/002/cody/OM4.zarr`.
  - 180 × 360 cells. Longitude edges are every 1° from 0 to 360.
  - Latitude bands are uneven: ~0.997° wide, with 1.258° bands at the poles.
  - The 1° output uses OM4's own `lat`/`lon` as cell centres. These are not
    band midpoints; they are off by up to 0.13°.
- **The 1/4° grid splits each Samudra cell 4×4.** Each cell is cut into 4
  equal-degree latitude strips and 4 longitude strips of 0.25°, giving
  720 × 1440. Its centres are the strip midpoints.
  - The 1° values are exact 4×4 sums of the 1/4° area sums, so only one set of
    weights is needed.
- **Conservative regridding with static weights**, built once in `init`.
  - Each 1/48° cell's area is shared out among the 1/4° cells it overlaps.
  - A coarse value is the overlap-area-weighted mean of the wet 1/48° cells.
  - A coarse cell is ocean at level k if any overlapping 1/48° cell is wet at
    k, so no ocean pixels are lost.
  - **Aligned tiles: exact.** Outside the polar regions, LLC cells are exact
    lon/lat rectangles. Their longitude edges sit on multiples of 1/48°, so
    every 0.25° line is a cell edge and each fine cell falls in one coarse
    column. A fine row that straddles a coarse latitude edge is split by
    spherical area (the sin-latitude fraction).
  - **Curvilinear tiles: 4×4 sub-cells.** These are the Arctic cap (face 6),
    the edges of faces 2, 5, 7 and 10 next to it, and the Antarctic-margin
    tiles of faces 0, 3, 9 and 12. The alignment test finds them; there is no
    hand-written list. Each fine cell is split into 16 sub-cells from its
    corner points. Its area is shared by where the sub-cell centres fall,
    which is good to about 1/8 of a fine cell (~250 m).
  - **16×16 sub-cells poleward of 85°.** There, 1/4° cells narrow to tens of
    metres, so these tiles use 16×16 sub-cells instead. It costs only `init`
    time.
  - **Polar fill.** A 1/4° cell that gets no area is filled from the 1/48° cell
    under its centre. This happens only within a few tenths of a degree of the
    poles, where cells become narrower than 1/48° cells. In that regime this
    is also what an exact regrid gives.
  - The coarse U-Net zero-pads beyond the poles, as Samudra does. That is a
    model setting, not part of this script. Samudra's own data has ocean
    values up to 89.24°N, and ours does too.
- **Velocities are centred and rotated at 1/48°, then regridded as scalars.**
  - U/V and oceTAUX/oceTAUY are moved to cell centres and rotated to east/north
    with CS/SN, as in `rotate_to_geo` / `interp_to_center`
    (`LLC_ocean_emulator/llc_cross_face.ipynb`).
  - At the last column/row of a face, the edge value is copied (the notebook's
    `boundary="extend"`). Inside a face, tile edges use the true neighbouring
    values.
  - Face rotations need no special handling, because each 1/48° cell carries
    its own lon/lat.
- **`land_frac` is 3D, (51, y, x).** It is 1 − the overlap-area-weighted mean
  of the 1/48° hFacC over the coarse cell. So 1 = all land, 0 = all water, and
  partial bottom cells and sloping shelves fall in between. The binary ocean
  mask is `land_frac < 1`.
- **`wet_area` is 3D, (51, y, x).** It is the area of wet 1/48° cells in each
  coarse cell: the averaging weight. Use it for area-weighted losses and
  conservation checks.
- **The 1/48° mask is the raw 3D `hFacC`.** It is already stored as one
  720×720 tile with all 51 levels per chunk, (51, 1, 720, 720), so it is not
  copied. The rechunk store still gets the 2D `hFacC_2d`.
- **`hFacC_2d` is the open-water fraction of each column's bottom cell.** It is
  hFacC at the deepest wet level among the 51: 1 for a column wet through all
  51 levels, the partial value for a partial bottom cell, and 0 for land.
- **One array per variable, one chunk per hour.**
  - U, V, Theta and Salt are (51, y, x) per chunk.
  - Eta, oceTAUX, oceTAUY, oceQnet and oceFWflx are (y, x) per chunk.
- **Compression: zstd level 3 with byte shuffle**, the 1-face cache's choice,
  for all three stores. Measured on our float32 output, it compresses 1.4–2.5×,
  as well as or better than bitshuffle, and writes about 2× faster than the
  raw store's zstd-5.
- **One merged pass.** For each global hour:
  - read the five 2D globes and write their rechunked tiles;
  - stream U, V, Theta and Salt face by face;
  - regrid everything;
  - write both resolutions, then the hour's marker.
  One hour of 3D fields is 198 GB in float32, so it is never held in memory
  at once. The rechunk finishes when the whole job does.
- **Values are float32 in physical units**, with land as NaN. Normalization
  stats come later from `stats` mode.
- **Bad or missing raw data is replaced by the previous clean hour** (revised
  2026-10-22).
  - A 720×720 tile of one variable is bad if:
    - its raw chunk is missing;
    - it holds values no ocean has (outside `LIMITS`, e.g. Theta −5..50 °C,
      Salt 0..100, |U|, |V| ≤ 20 m/s, |Qnet| ≤ 1e6 W/m²) or infinities;
    - Theta/Salt NaNs disagree with `mask_c`;
    - a 2D field has NaN on a wet cell.
  - That tile is copied from the same tile at the previous clean hour.
  - The 60/59 missing forcing steps copy the previous hour (hour 0 copies
    hour 1).
  - Found 2026-10-21: the raw store holds corrupted 1 KB blocks, e.g. hour
    467 Theta level 41 on face 6 (256 cells of garbage), and U/V at hours
    142, 397 and 433.
  - Every repair is recorded in `<coarse-root>/_repairs/<t>.json`.
    `--mode report` (also run at the end of each fill) merges them into the
    single log `<coarse-root>/repairs.log`.
  - Each store (rechunk, 1/4°, 1°, 1/3°) has `repaired[time, variable]`:
    −1 not written, 0 original, 1 repaired.
  - Hours finished before this rule that need a repair are rewritten
    automatically. Their marker is dropped during the rewrite.
- **Three resolutions** (2026-10-21):
  - 1/4° and 1° on the Samudra lat/lon grid (product `samudra`);
  - 1/3° on the native LLC faces (product `native`): 16×16 boxes, (13, 270,
    270) per level, `LLC4320_native_0p33deg.zarr`. It uses the same variables,
    east/north velocities, statics, `land_frac` and `wet_area`.
  - Each product has its own markers (`_progress`, `_progress_native`).
  - Every hour is read once for all products.
- **Static fields are stored once.** The position/area input channels are built
  at load time from the stored 2D `XC`, `YC` and `rA` with
  `_packed_spatial_features` (`src/ocean_emulators/utils/data.py:146`).
- **The tile model needs a lookup to inject coarse values.** Coarse cells no
  longer line up with 720 tiles. The regrid weights from `init` give, for each
  1/48° pixel, the coarse cell(s) it falls in. Turning that into injection is
  later work.

## Source and destination

```text
source (read-only, zarr v2, no consolidated metadata -> consolidated=False)
  /orcd/data/abodner/003/LLC4320/LLC4320
    3D  Theta, Salt (j,i), U (j,i_g), V (j_g,i)   [10311, 51, 13, 4320, 4320]  chunks (1,51,1,720,720)
    2D  Eta, oceQnet, oceFWflx, oceTAUX (i_g), oceTAUY (j_g)  [10311, 13, 4320, 4320]  chunks (1,13,4320,4320)
    grid XC YC XG YG rA Depth CS SN [13,4320,4320]; hFacC [51,13,4320,4320]
target grid
  /orcd/data/abodner/002/cody/OM4.zarr   lat, lon (180,360); lat_b, lon_b (181,361)

coarse (scratch; reach it only from node4006-type nodes, never node4005)
  /orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_samudra_0p25deg.zarr
  /orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_samudra_1deg.zarr
  /orcd/scratch/seedfund/001/llc_emulator/coarse/_progress/<t:05d>

rechunk (data disk; quota raised 2026-10-21, finished hours moved there from scratch)
  /orcd/data/abodner/002/cody/LLC_boundary_rechunk/LLC4320_2d_720.zarr
native 1/3 degree
  /orcd/scratch/seedfund/001/llc_emulator/coarse/LLC4320_native_0p33deg.zarr
```

Time is 10311 hourly steps, indices 0–10310, from 2011-09-13T00 to
2012-11-15T14. The train/val split matches the face-1 cache: steps 0–8807 are
train and 8808–10310 are val.

## Output layout

**Coarse store**, one per resolution: (ny, nx) = (720, 1440) at 1/4° and
(180, 360) at 1°. Arrays are float32 with NaN fill unless noted, compressed with
zstd-3 byte shuffle.

| array | shape | chunks | contents |
|---|---|---|---|
| `U`, `V`, `Theta`, `Salt` | (10311, 51, ny, nx) | (1, 51, ny, nx) | U east, V north |
| `Eta`, `oceTAUX`, `oceTAUY`, `oceQnet`, `oceFWflx` | (10311, ny, nx) | (1, ny, nx) | oceTAUX east, oceTAUY north |
| `lat`, `lon`, `lat_b`, `lon_b` | (ny,), (nx,), (ny+1,), (nx+1,) f8 | whole | centres and edges (1°: Samudra's exactly) |
| `XC`, `YC` | (ny, nx) | whole | 2D centre lon/lat, for `_packed_spatial_features` |
| `rA` | (ny, nx) | whole | spherical cell area |
| `dxF`, `dyF` | (ny, nx) | whole | east-west / north-south cell size at the centre latitude |
| `Depth`, `hFacC_2d` | (ny, nx) | whole | overlap-weighted mean over surface-wet 1/48° cells |
| `land_frac` | (51, ny, nx) | (51, ny, nx) | 1 − overlap-weighted mean of 1/48° hFacC |
| `wet_area` | (51, ny, nx) | (51, ny, nx) | wet 1/48° area in the cell (m²), the averaging weight |
| `time` | (10311,) int64 | (1024,) | copied from the raw store, with its attrs |
| `<var>_mean`, `<var>_std` | (51,) or (1,) | whole | written later by `stats` |
| `regrid/` (1/4° store only) | | | the weights (below) |

The regrid weights live in the 1/4° store as entries grouped by 720 tile:

- `regrid/pixel` (int32): the pixel's index within its tile, `j*720 + i`;
- `regrid/cell` (int32): the flat 1/4° cell index, `y*1440 + x`;
- `regrid/area` (float32, m²): the overlap area.

`regrid/tile_start` (469,) holds each tile's first entry, and
`regrid/tile_method` (468,) says whether the tile was regridded exactly (0) or
with sub-cells (1).

Attrs on each coarse store: the variable list, the source and target-grid
paths, `velocity_frame: east/north`, the face-edge rule, the regrid method,
`filled_time_steps_json`, the train/val split, `land_fill: NaN` and
`built_by`.

**Rechunk store.** Each of Eta, oceTAUX, oceTAUY, oceQnet and oceFWflx is its
own array with the raw dims (time, face, j, i / i_g / j_g).

- Shape (10311, 13, 4320, 4320), chunks (1, 1, 720, 720), float32, zstd-3 byte
  shuffle.
- `dimension_separator="/"` keeps directories small.
- `write_empty_chunks=False` skips all-land tiles.
- Also holds the static `hFacC_2d` (13, 4320, 4320) with chunks (1, 720, 720),
  `time`, and `filled_time_steps_json`.
- Values are raw and unrotated, the same as the source.

**Sizes**

| output | per hour (uncompressed) | total uncompressed | total compressed (est.) |
|---|---|---|---|
| 1/4° | 865 MB (4 × 211 MB 3D + 5 × 4 MB 2D) | ~8.9 TB | ~4 TB |
| 1° | 54 MB | ~0.56 TB | ~0.25 TB |
| rechunk | 4.85 GB | ~50 TB | ~20–25 TB |
| regrid weights | once | ~3.5 GB | |

Free space on scratch was 84 TB on 2026-10-20, enough for all three stores
(~30 TB). The rechunk is ~24M chunk files, minus all-land tiles. It can be
copied to `/orcd/data/abodner/002/cody/LLC_boundary_rechunk` once that quota
allows.

## How each value is computed

**Weights**, built once in `init`, face by face and tile by tile.

1. Build the face's corner points from `XG`/`YG`, which give the south-west
   corner of each cell in index space. The missing last row/column is
   extrapolated.
2. A tile is **aligned** if every cell's corners share two longitudes and two
   latitudes. Each aligned cell:
   - goes to the 0.25° column containing its centre;
   - goes to the latitude strip(s) its south/north edges span, split by
     sin-latitude when it straddles a strip edge.
   A check confirms each cell's longitude edges fall inside one column; if
   any don't, the tile falls back to sub-cells. A latitude edge within 1e-6°
   (~0.1 m) of a target edge counts as on it, so float noise never splits off
   a near-zero sliver into the next row.
3. Other tiles use **sub-cells**: 4×4 points per cell (16×16 for tiles reaching
   85°), bilinear between the four corner unit vectors. Each point carries an
   equal share of rA to the 1/4° cell it falls in.
4. **Polar fill.** Any 1/4° cell with zero total area gets one entry from the
   1/48° cell nearest its centre, found with a KD-tree over fine-cell centres
   poleward of 80°. The entry's area is the cell's own area.

**Values**, for one tile at one level k, with weights `W` and `wet` from the
data's NaN pattern (Theta's for 3D, `hFacC_2d > 0` for 2D):

```text
num[cell] += sum over entries of area * wet[pixel] * x[pixel]
den[cell] += sum over entries of area * wet[pixel]
x_quarter  = num / den                       # NaN where den == 0
x_one      = box_sum(num, 4) / box_sum(den, 4)
```

`num` and `den` are summed in float64 over every tile of every face, since a
coarse cell can take area from several faces. One per-tile sparse product
handles all 51 levels at once.

**Velocities**, for a whole face at a time:

1. Set NaN to 0, since there is no flow through a land face.
2. Centre: `uc[i] = (U[i] + U[i+1]) / 2`, one level per thread. The face edge
   copies its own value.
3. Rotate per pixel: `east = uc*CS − vc*SN` and `north = uc*SN + vc*CS`.
4. Regrid with Theta's wet mask.

**Statics** at each resolution:

| field | how it's computed |
|---|---|
| `XC`, `YC` | broadcast from `lat`/`lon` |
| `rA` | `R² Δλ (sin φn − sin φs)`, with R chosen so the total equals the LLC area |
| `dxF` | `R cos φc Δλ` |
| `dyF` | `R Δφ` |
| `Depth`, `hFacC_2d` | regridded with the surface-wet mask |
| `land_frac[k]` | `1 − Σ(area · hFacC[k]) / Σ(area)` |
| `wet_area[k]` | `Σ(area · [hFacC[k] > 0])` |

`init` also checks that CS² + SN² ≈ 1 on wet cells.

**Guards against silent errors:**

- A missing raw 3D chunk reads as NaN, and centring would turn it into 0. So a
  tile with any wet cell must have its U, V, Theta and Salt chunk files on
  disk.
- Salt's NaN pattern must equal Theta's.
- At the end of each hour, the coarse ocean pattern (`den > 0`) must equal
  `land_frac < 1` at every level and both resolutions.
- An all-NaN U/V tile is not itself an error. Isolated lakes such as Lake
  Victoria have wet cells but no open faces.

## Script: `scripts/coarsen_llc.py`

Modes:

- **`init`**: run once.
  - Refuses if an output store exists, unless given `--force`, which also
    clears the markers.
  - Scans for missing 2D steps.
  - Creates the three stores.
  - Builds the regrid weights, the coarse statics and the 1/48° `hFacC_2d`.
- **`fill`**: for each pending hour:
  1. Read the five 2D globes, filling missing steps from t±1.
  2. Write the 2340 rechunk tiles.
  3. Centre and rotate the wind stress, then regrid the 2D fields.
  4. For each face, read U/V, centre them, then for each tile read
     Theta/Salt, rotate, and regrid.
  5. Run the end-of-hour guard.
  6. Write every variable at 1°, then at 1/4°, then the marker.
- **`stats`**: per-variable, per-level area-weighted mean/std over 96 evenly
  spaced finished train hours.
- **`verify`**:
  - Count markers.
  - Check that each marked hour has every chunk file.
  - Delete leftover `*.partial` files in the coarse stores.
  - Spot-check conservation (Eta and Theta, global).
  - Consolidate metadata.
  - Run it only when no fill is running.

**Restart.**

- An hour is done only when its marker exists. The marker is written after
  every chunk of that hour.
- Zarr writes chunks by rename, so a killed run just redoes its unfinished
  hours.
- Rechunk arrays are reopened with `write_empty_chunks=False`, because that
  setting is not saved in the metadata.
- `fill` never consolidates metadata.

**Parallelism.**

- `--workers W` processes, one hour each, with `--read-threads T` threads.
  Blosc threads are off and `OMP_NUM_THREADS=1`.
- `--time-splits N --time-split-index i` splits hours across nodes.
- `--stride S --offset O` does hours with t ≡ O (mod S) first.
  `S=24, O=6` gives a daily subset that avoids the filled steps.
- Expected worker peak is ~40 GB:
  - U/V faces 7.6 GB;
  - 2D globes 4.9 GB, freed before the 3D part;
  - weights ~3.5 GB;
  - float64 sums 2.1 GB;
  - tile scratch.

**Job script:** `JOBS/coarsen_llc.sh`.

- Env vars: `MODE`, `WORKERS`, `READ_THREADS`, `STRIDE`, `OFFSET`, `STEPS`,
  `LIMIT`, `COARSE_ROOT`, `RECHUNK_ROOT`, `FORCE`.
- Defaults: 3 workers × 21 threads (benchmarked); Slurm `mit_normal` with the
  `mit_amf_advanced_cpu` account/QOS (4-day limit), `-c 64 --mem=250G`,
  96 h.
- Fill runs as `sbatch --array=0-2`: three one-node tasks, each taking a third
  of the hours. `--array` stays on the command line so `init`, `stats` and
  `verify` remain single jobs.
- Python runs from `.venv/bin/python`, never `uv run`.
- Run `timeout 20 ls` on the scratch path before any work.

## Reuse

- `scripts/build_llc_face_cache.py`: `open_source_array`, `SourceArray`,
  `create_array` (with its `dimension_separator` argument), and the time-split
  pattern.
- `src/ocean_emulators/llc_coarse.py`: the pure helpers (grid, weights,
  centring, rotation, `bottom_hfac`, missing-step rule).
  `tests/test_llc_coarse.py` covers them.
- `src/ocean_emulators/utils/data.py`: `_packed_spatial_features`.
- scipy sparse matrices for the per-tile regrid products, and `cKDTree` for
  the polar fill. Both are already in `.venv`; no new dependencies.

## Phases

0. **Preflight** (skipped for now: write access for ~24M files is pending).
1. **Helpers and unit tests.**
2. **Script, job, `init`** into a test root.
3. **One-hour checks** on t=1 and t=288 (a filled step).
4. **Benchmark** W = 1, 2, 3 on one node.
5. **Daily subset** (`--stride 24 --offset 6`), then all hours.
6. **`stats`, then `verify`.**

## Verification

1. **Unit tests:**
   - Aligned-cell weights split by sin-latitude and sum to rA.
   - Sub-cell weights sum to rA.
   - 1° from 1/4° sums equals direct 1° weights on a synthetic grid.
   - Polar fill gives each empty cell the value under its centre.
   - Centring and rotation are unchanged (the identity, and faces 7–12 giving
     east = v_local and north = −u_local).
   - `bottom_hfac` and `fill_sources` behave as specified.
2. **Weights, after `init`:**
   - The weights of every 1/48° cell sum to its rA (relative error < 1e-5).
   - The weights in every 1/4° cell sum to its spherical area: within 1e-3 on
     aligned rows; the curvilinear and polar error is reported.
   - Every 1/4° cell has at least one entry.
3. **One hour, end to end:**
   - Shapes, chunks and dtypes are as specified.
   - The rechunk equals the raw data bit for bit (filled steps: the mean of
     t±1).
   - Global conservation: Σ(coarse x · `wet_area`) equals Σ(fine x · rA) over
     wet cells, for Eta and every Theta level.
   - Hand-check one aligned cell with an independent overlap computation, and
     one Arctic cell against an 8×8 sub-cell estimate.
   - Coarse NaNs equal `land_frac == 1`.
   - No jumps along the LLC face-boundary longitudes (−38°, 52°, 142°, −128°)
     or at 67.5°N.
   - Save a quick-look PNG.
4. **Restart:** kill mid-hour and rerun. The finished hour is skipped and the
   rest are redone and pass the checks.
5. **`verify`:** all 10311 markers present, conservation spot checks pass.

## Expected cost

Reads are unchanged: ~74 GB per hour (72 GB 3D + 2 GB 2D), ~760 TB in total.
The raw store's NFS server (`hstor002-n2`) delivered 0.6–0.75 GB/s to one node
on 2026-10-16, and ~1.4 GB/s in total across three nodes. At those rates:

- the full run takes about **12 days on one node**, or **~6 days on three**;
- the daily subset (430 hours) takes about 12 h on one node.

Regridding adds CPU work (~60 ms per tile, variable and 51 levels), still far
below the read time.

## Later, not in this script

- The coarse U-Net's zero padding at the poles and periodic padding in
  longitude.
- A loader for the coarse stores, and injecting coarse values into the 1/48°
  U-Net's 3rd dilation level using the regrid weights.
- Training the coarse models a/b/c (full year, first half, second half).

## Definition of done

- All three stores pass `verify` with 10311/10311 hours.
- Unit tests pass.
- The weight and one-hour checks are recorded in an Implementation status
  section below.

## Implementation status (2026-10-20)

Built:

- `src/ocean_emulators/llc_coarse.py`: grid, weights, centring and rotation
  helpers.
- `tests/test_llc_coarse.py`: 16 tests, all passing. One checks real tiles:
  face 1 is aligned, the Arctic cap is not.
- `scripts/coarsen_llc.py`: modes `init`, `fill`, `stats`, `verify`.
- `scripts/check_coarsen_llc.py`: the weight and one-hour checks.
- `JOBS/coarsen_llc.sh`: production, an array of 3 one-node tasks on
  `mit_normal`.
- `JOBS/other/validate_coarsen_llc.sh`: init, hour checks, restart test and
  benchmark in one job.
- `JOBS/other/bench_coarsen_llc.sh`: the throughput sweep.

All checks ran in `/orcd/scratch/seedfund/001/llc_emulator/coarse_test_samudra`
(validation job 26455724 on node4006, 40 min, 0 failures).

**`init`:** 173 s on 64 CPUs, ~11 GB peak, 265.1M regrid entries.

- 180 of 468 tiles use sub-cells: Arctic cap, the cap-side edges of faces 2,
  5, 7 and 10, and the Antarctic margin. 24 of them use 16×16.
- No polar fill was needed: every 1/4° cell receives area.

**Weights:**

- Every 1/48° cell shares out exactly its rA (worst 6e-8).
- 1/4° cells fed only by aligned tiles match their spherical area within
  2.2e-5.
- Cells fed by sub-cells, worst / 99th percentile in ocean:

| latitude | worst | 99th percentile |
|---|---|---|
| south of 85°N | 2.2% | 0.8% |
| 85–89°N | 3.5% | 0.3% |
| 89–90°N | 18% | 1.6% |

  At 89–90°N it was 73% before the 16×16 change.

**Hours 1 and 288 (a filled step), and hour 3 after the restart:**

- **Rechunk:** bit-for-bit equal to the raw globes. At hour 288 it is the mean
  of 287 and 289, and the step is listed in all three stores.
- **Conservation:** global Eta and Theta (all 51 levels) conserved to ≤ 3e-9
  at both resolutions.
- **Aligned hand check:** a face-1 cell (−9.85°, 10.125°E), recomputed
  independently from raw cell edges, matches Theta, U east, V north (k = 0, 20)
  and Eta to all printed digits.
- **Arctic check:** a cap cell (85.1°N, 30.1°E) matches an 8×8 sub-cell
  estimate.
- **NaN patterns:** coarse NaNs equal `wet_area == 0` at every level and both
  resolutions. `land_frac == 1` exactly there too.
- **Face boundaries:** jumps across the LLC face-boundary longitudes and
  67.5°N are 0.75–1.26× the jumps a few cells inside.
- **Fix found by these checks:** float noise where a fine row edge sits on a
  target edge had split off ~1e-15 slivers, wrongly marking 23 (level, cell)
  pairs as ocean. Edges within 1e-6° now count as coincident.

**Restart:** a fill of hours 2–4 was killed (main process and workers) during
hour 3. Afterwards only hours 1, 2 and 288 were marked. The rerun skipped
hour 2, redid hours 3 and 4, and hour 3 passes every check.

**Throughput** (one node, 64 CPUs, fresh hours):

| workers × threads | s/hour wall | read GB/s | worker peak |
|---|---|---|---|
| 1 × 48 | 116 | 0.64 | 37 GB |
| 2 × 32 | 86 | 0.86 | 35 GB |
| 3 × 21 | **66** | 1.12 | 28 GB |

Production uses 3 × 21. Reading U/V and the tiles' Theta/Salt is ~85% of each
worker's time; centring ~10%; regridding and writing ~2 s per hour.

- **Full run estimate on 3 nodes:** about 2.6 days if each node keeps
  1.1 GB/s. It could take up to ~6 days if the raw store's server caps the
  total near 1.4 GB/s, as it did on 2026-10-16.
- **Daily subset:** 430 hours, about 2.6 h.

**Quota:** `/orcd/data/abodner/002` refused new files ("Disk quota exceeded"),
so the rechunk store is on scratch for now.

## History: native-face build (2026-10-16)

The first version box-averaged on the native faces and stored them as
(13, 360, 360). It passed all its one-step checks:

- bit-exact rechunk;
- conservation ≤ 1.5e-7;
- hand-checked U/V;
- seams continuous;
- restart after a kill.

Lessons carried into this version:

- **Lakes:** isolated lakes such as Lake Victoria (face 1, tile (3600, 2880))
  have wet cells but no open U/V faces, so U/V there is legitimately all NaN.
- **Centring:** U/V centring must run one level per thread. Serial centring
  cost ~150 s per hour.
- **Throughput** on node4006, limited by reads from the raw store's NFS
  server:

| part | workers × threads | s/step wall | read GB/s | worker peak |
|---|---|---|---|---|
| boundary | 4 × 30 | 2.6 | 0.83 | 12 GB |
| prognostic | 2 × 36 | 95 | 0.75 | 32 GB |
| prognostic | 4 × 30 | 118 | 0.61 | 29 GB |

  Plain parallel `cat`s gave 0.72 GB/s from one node, and 0.47 GB/s per node
  from three nodes at once.
