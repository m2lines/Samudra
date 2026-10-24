<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# OM4 and CM4 data viewer

Compare local 1° five-day means with native ¼° five-day means and instantaneous
state snapshots from the public OSN pod. Panels have independent dataset, field and depth selectors, with one
shared timeline and linked geographic zoom/pan. The initial layout puts local and native mean meridional velocity side by side,
then local mean and native snapshot zonal velocity.

## Start

```bash
./apps/om4_viewer/serve.sh --data ~/data/om4_onedeg
```

Open <http://127.0.0.1:8766>. The launcher installs a small standalone environment
with `uv` (Python 3.12 or 3.13); no training environment is needed. `--data`
accepts either `OM4.zarr` or its parent directory. Repeat it to add compatible
local stores. Use `--no-native --no-rollouts --no-cm4` to run entirely offline,
or `--no-snapshots` to omit the snapshot source.

## CM4 and calendars

Select **CM4 1° · piControl · OSN** in any panel. This reads the published
`s3://m2lines-pubs/Samudra/v2026-09/cm4_onedeg/CM4.zarr` through the server's
existing `nyu-osn-m2lines-pubs` rclone remote and bounded object cache. Use
`--no-cm4` to omit it. The product has 14,600 five-day means on a 180 × 360
Gaussian grid, with 19 depths, model dates **0151-01-06–0351-01-01**, and the
**noleap** calendar. It includes temperature, salinity, velocities, SSH, heat
flux, wind stresses, and freshwater flux. Depth labels come from its `lev`
coordinate; masks come from its depth-dependent `wetmask`. The reader exposes
depth slices without regridding or modifying the stored wet-cell samples.

The **Calendar** selector controls the shared timeline. Maps from another
calendar turn gray with a **Switch to … calendar** button. Switching remembers
each calendar's position during the session and preserves the linked map zoom,
fields and depths. If no selected source uses the active calendar, the timeline
switches to the first remaining calendar. The active calendar is saved in the
URL, independently of panel order; older links remain supported.

CM4 dates are model years, not historical dates matched to OM4. Its native
timestamps already label interval ends; the viewer uses the recorded
`time_bnds` right endpoints for interval matching, without applying OM4's
midpoint offset. Calendar arithmetic stays in the data's calendar, including
years that precede modern observations. Different calendars are never matched
by identical date strings or subtracted in a computed map. Within CM4,
**Current − previous frame** works as usual.

## Official Samudra 2 rollouts

The dataset selector includes the three evaluations selected in the
[Samudra 2 Official Runs workspace](https://forge.coreweave.com/wandb/ocean_emulators/default?nw=ip39lwu4tys):

| Resolution | Evaluation | Archived output directory |
| --- | --- | --- |
| 1° | [uw49dzuq](https://wandb.ai/ocean_emulators/default/runs/uw49dzuq) | `2026-01-30-samudra_om4_onedeg_20_cap_eval` |
| ½° | [20bfg825](https://wandb.ai/ocean_emulators/default/runs/20bfg825) | `2026-02-05-samudra_om4_halfdeg_20_cap_fix_eval` |
| ¼° | [9kl66qky](https://wandb.ai/ocean_emulators/default/runs/9kl66qky) | `2025-02-08-samudra_om4_quarterdeg_cap_20_eval_fix` |

Each directory contains `predictions.zarr` under
`nyu-osn-emulators:emulators/jr7309/outputs/`. These private archives are read
on demand by the local server using the installed `rclone` and its existing
`nyu-osn-emulators` configuration. Credentials never go to the browser. Use
`--no-rollouts` to omit them when that remote is unavailable. No model inference
is run and no complete rollout is downloaded. Objects share the bounded 2 GiB
disk cache with native OM4 reads.

The saved predictions are physical values on 1°, ½°, and ¼° Gaussian grids,
with 598 frames from October 20, 2014 through December 24, 2022 (Julian calendar)
and all 19 output depths. Field
units omitted by the old writer are supplied from each evaluation's recorded
OM4 source (`om4_onedeg_v3`, `om4_halfdeg_v4`, or `om4_quarterdeg_v2`). Its
depth-specific ocean masks hide finite fill values saved over land; source and
rollout coordinates are checked for equality. Predictions contain ocean state
fields, not the atmospheric forcing fields. Select **Samudra 2 … · official rollout** in any panel; its evaluation
link appears above the map. The shared timeline spans the dates from all selected
datasets in the same calendar. Outside a rollout's coverage, its map turns gray
and shows its available date range; click either displayed date to jump the
linked timeline to that first or last frame. The buttons respect the selected
time-alignment mode and retain the current zoom. The other maps keep advancing. Missing frames
are not interpolated or held at their last value. Color ranges use the same local
OM4 reference as the other maps.

For a server on another machine, forward the port:

```bash
ssh -N -L 8766:127.0.0.1:8766 your-host
```

## Public hosting

The supplied user services expose this viewer on port 61016.
The public port forwards to the same
loopback reader on port 8766, preserving the in-app browser connection.
All configured datasets, including official rollouts, are accessible without
login. OSN credentials remain in the server's existing rclone configuration;
the HTTP routes serve only the UI, metadata and selected data arrays.

The host-specific unit files are in `deploy/` and reference the persistent
`~/services/om4-viewer` checkout and its standalone environment. Do not deploy
from a disposable Codex worktree: deleting it removes static assets even while
the Python server remains alive. Both the viewer and public socket are enabled at
user-manager startup; user lingering is enabled on this host. The viewer restarts
automatically on failure. Keep this checkout and environment available while
these services are installed.

```bash
systemctl --user status om4-viewer.service om4-viewer-public.socket
systemctl --user restart om4-viewer.service
journalctl --user -u om4-viewer.service -f
```

## Comparison provenance

**The currently configured mean and snapshot archives are not a verified
mean/snapshot pair from one simulation trajectory.** Their source titles are
`OM4p25_5daily_cycle1` and `OM4p25_JRA55do1.5_r6_4Emulator`. These different
archive identifiers suggest different simulation runs/cycles, but the exact
meaning of `r6` has not been established from a run manifest. Matching calendar
intervals does not resolve that provenance gap. Do not interpret a difference
between these archives as solely an averaging effect. The viewer shows archive
titles above each map.

## Compare and navigate

- Start with four maps. **Add map** in the top bar copies the last panel;
  the **×** button removes a panel. Keep between one and sixteen maps, and choose
  one to four **Columns**. Narrow screens stack maps into a single column.
- The URL automatically saves the maps' datasets, fields, depths, difference
  calculations and color ranges, along with the current time, active calendar, time alignment,
  linked zoom/pan, column count and playback settings. Copy or bookmark it to
  restore that view. Updates replace the current browser history entry;
  Back/Forward still restores views when navigating between saved links, without
  creating an entry for every time step or zoom. Playback opens paused.
  View state lives in the URL fragment, so even a large layout does not enlarge
  the server request. Links require the same datasets to be available; an invalid
  or unavailable view opens the default layout with an explanatory message.
- Choose a dataset in each panel, then temperature, salinity, U/V velocity, SSH,
  heat flux, freshwater flux or wind stress, where available. Freshwater flux
  (`wfo`) is available in CM4 and the native OM4 snapshot archive, as a
  five-day mean in kg m⁻² s⁻¹ (positive into the ocean). Native depth labels come from the source's `lev`
  coordinate. Surface fields disable the depth selector.
- Scroll or double-click a map to zoom; drag to pan. Every panel follows the
  same geographic extent. The toolbar also offers zoom buttons and a world reset.
  Native detail is fetched again at the new extent; zooming does not enlarge a
  previously downloaded low-resolution image.
- Scrub, jump to a date, or press **Space** to play/pause. Arrow keys step.
  **Same five-day interval** links a mean to the snapshot at that interval's end.
  Means are timestamped at the midpoint (12:00), while snapshots are 2½ days
  later (00:00). The shared slider is explicitly labeled **Interval ending**;
  every map shows its actual source timestamp and whether the selected field
  is instantaneous or a five-day mean. Surface forcing fields in the snapshot
  store remain five-day means, even though ocean state fields are snapshots.
  **Exact timestamp** uses the union of recorded timestamps in the selected
  calendar, with no nearest-date substitution. A panel without an exact match
  shows a gray placeholder. Each available date maps to its own source index.
  A date jump picks the first timeline date on or after the
  requested date. All three sources share 4,745 intervals: midpoint dates
  January 3, 1958–December 29, 2022; end dates January 6, 1958–January 1, 2023
  (Julian calendar).
- Set frame stride and target playback speed independently. Playback waits for
  all panels, so OSN/network throughput may limit the actual frame rate.
- Hover to see a physical value at the cursor. Colors clip at the displayed
  range, but hover preserves the original value.
- Initial color ranges are the local reference's pooled 2nd/98th percentiles
  from its first, middle and last frames, with symmetric velocity/flux/stress
  limits. Fields absent from the local reference, such as `wfo`, use the selected
  source's first, middle and last frames instead. The range label identifies
  the reference. Matching fields/depths share these ranges across resolutions. Editing
  limits or selecting **Fit this frame** also updates matching panels. Ranges
  remain fixed through playback; **Reset range** restores the reference range.

## Difference maps

Choose **Computed…** from a panel's dataset selector to open a small
builder. It supports two calculations:

- **A(current) − B(current)**: select the two datasets and then the common field
  and depth in the panel. Exact source timestamps are required by default.
  **Same five-day interval** explicitly permits midpoint means to be compared
  with end-of-interval snapshots; both source timestamps appear above the map.
- **Current − previous frame**: select one dataset. Previous means the preceding
  recorded frame in that series, independent of the global timeline or playback
  step size. The first frame has no predecessor and shows a placeholder.

Only matching field/depth/unit combinations are offered. Differences keep the
source units and show A minus B, so positive error means A exceeds B. Use the
panel's **Edit** button to change a calculation, or choose a normal dataset to
return to a raw map. Missing operands produce a gray placeholder and never
reuse a previous result or substitute a nearby time.
The dataset selector names the calculation using its source dataset names
(or the source name followed by `current − previous`), without a duplicate
calculation label below the selectors.
Each calculation is added to every panel's dataset selector for the current
session. Switching to another source leaves it available for later selection;
creating another calculation retains the earlier entries. Identical calculations
reuse an entry. An edited calculation is added as a new entry if its formula
changes. Where the same dataset names occur with both matching modes, the
entries include their time-matching mode. Reloading restores computations in
the saved view; inactive session entries are not saved.

The calculation subtracts the geographic display rasters at identical bounds
and pixel locations, using each source's existing nearest-cell lookup. Across
different grids this is a visual comparison, not a conservative remapping.
Cells must be wet and finite in both sources; land in either is gray, and missing
wet-cell values remain white. No arbitrary expressions or executable code are
accepted. Input requests reuse the existing bounded cache and concurrent-request
deduplication; subtraction runs on the displayed float arrays in the browser.

Difference maps use a zero-centered diverging palette, initialized from the
98th percentile of absolute finite differences in the first available frame.
The range stays fixed during playback. Matching calculations share range edits;
raw-field ranges remain separate. **Fit this frame**, custom limits, and
**Reset range** are available as for ordinary maps.

## Native data and display

The Python reader first tries these published HTTPS stores anonymously:

- [Native instantaneous state snapshots](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/om4_5daily_snapshots.zarr/.zmetadata)
- [Native five-day means](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/om4_5daily.zarr/.zmetadata)
- [Native grid geometry](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/ocean_static_no_mask_table.zarr/.zmetadata)
- [Supergrid and vector angles](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/grids/ocean_hgrid.zarr/.zmetadata)

If OSN denies anonymous reads with HTTP 403, the server reads the **same object**
using the existing `nyu-osn-m2lines-pubs` rclone configuration. This fallback is
restricted to the published `m2lines-pubs/Samudra/raw/` prefix. It does not change
datasets or grid geometry, and credentials and rclone authentication diagnostics
never go to the browser. Authenticated and anonymous reads share the same bounded
object cache. This fallback was added October 15, 2026 after the pod began denying
anonymous reads, including metadata, while the bucket-specific credentials still
worked. The separate `nyu-osn-emulators` credentials continue to serve rollouts.

This is the native **1080 × 1440 tripolar grid**, not the separately published
720 × 1440 Gaussian regridding. Map lookup uses actual two-dimensional
`geolon`/`geolat`, including across the northern fold and longitude seam.
Staggered U/V (and stress) components are averaged to tracer centers and rotated
using `angle_dx` to geographic east/north, matching the preparation pipeline.
Missing face values are zero-filled for that centering step; ocean/depth masks
are applied afterward. Interior wet masks use the matching native temperature
slice; surface views use the grid's wet mask. Gray means land/below seafloor;
white means missing wet-cell data. Valid zero values remain visible.

The browser receives a geographic display raster (up to 1440 pixels wide).
Each native raster pixel takes its value from the nearest native cell center
on the sphere: there is no smoothing, area averaging or interpolation of field
values in this display step. The source cell lookup is recomputed for zoomed
views. Local rectilinear maps preserve the actual Gaussian latitude cell edges.
This display raster is for inspection, not a conservatively regridded dataset.

## Caching and throughput

Native means store **one time and all 19 depths per chunk**, commonly 50–60 MB
compressed for a velocity component. Native snapshots store **ten times and
one depth per chunk**. The reader respects both layouts: mean depth changes and
snapshot time steps within a chunk reuse the decoded data. It does not read
all 19 snapshot depths when only one is selected. A cold initial
map also reads static grid geometry. There is no speculative native prefetch
while paused and no full-dataset download.

Fetched OSN objects are cached atomically under `apps/om4_viewer/.cache`,
bounded to 2 GiB with eviction. Override with `--cache-dir PATH`. The cache
contains downloaded data from the configured sources, no credentials. It assumes the published
source objects remain unchanged; stop the server and remove the cache to force
fresh downloads. Decoded native chunks, displayed frames, geographic lookup
maps and the browser cache are separately bounded. A missing remote data chunk
raises an error rather than silently displaying a fill-value frame.
Source-read failures return a descriptive JSON error with HTTP 502, rather than
a generic internal-server-error page.

## Checks

```bash
uv run --project apps/om4_viewer python -m unittest discover -s apps/om4_viewer/tests -v
node --check apps/om4_viewer/viewer.js
# With Playwright and its Chromium browser installed:
node apps/om4_viewer/tests/timeline.browser.cjs
node apps/om4_viewer/tests/differences.browser.cjs
node apps/om4_viewer/tests/view-state.browser.cjs
node apps/om4_viewer/tests/calendars.browser.cjs
```

The tests cover local and native orientation, vector centering/rotation, depth
masks, valid zeros, geographic wrapping and zoom, multiple datasets, bounded
caching, missing-chunk errors and API validation. Browser checks against live
OSN additionally cover side-by-side maps, exact-date playback, depth/dataset
changes, linked wheel zoom and drag pan, mobile layout, and sources whose
matching dates occur at different indices.
The standalone timeline browser check uses small fixtures to test date unions,
short-record boundaries, missing frames, playback, linked zoom, and stale responses.
The view-state browser check covers adding/removing maps, columns, full URL
restoration including calculations, catalog reordering, replacement history,
Back/Forward navigation, layout limits, mobile stacking and invalid-link fallback.
The calendar check covers isolated calendars, ancient noleap date jumps,
remembered positions, linked zoom, URL restoration and computed-map restrictions.

Snapshot checks additionally cover `z_l` depth normalization, ten-time/one-depth
chunk reuse, Julian leap-day alignment, instantaneous versus forcing-mean labels,
and live mean/snapshot comparisons with the 2½-day timestamp offset.

## Local native gated rollout

`--local-rollout ~/data/samudra-native-gate-ema/viewer.zarr` adds the completed
native persistence-gated model's final epoch-70 EMA forecast. It is a continuous
598-frame rollout from October 2014 through December 2022, not the separate
October 2015-origin example used in the experiment's PNG maps. All 77 state
channels are available; this is the original feature-conditioned gate, not the
new direct-input/nonlinear training arms.

Prepare the downloaded `predictions.zarr` with:

```bash
uv run --project apps/om4_viewer python apps/om4_viewer/prepare_gated_rollout.py \
  --root ~/data/samudra-native-gate-ema --truth ~/data/om4_onedeg/OM4.zarr
```

The prepared store contains physical values, per-depth ocean masks, units,
original Julian timestamps, and a W&B provenance link. Preparing it flattens
`lev` into the viewer's field names and rechunks losslessly to individual frames;
no regridding or normalization is applied. Grid coordinates and times must match
the source OM4 store. All prepared values are compared with the saved predictions
on full read-back before enabling the dataset.

Choose **Samudra 1° · native gated · final EMA** in the dataset selector.
Use **Computed difference** against **OM4 1° · local** for forecast errors.
The official 1° rollout remains selectable but is not the matched ungated
training control. Local rollouts require explicit masks and do not alter the
existing official-rollout catalog or shared URL source identifiers.
