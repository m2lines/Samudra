<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Native surface observations and forcing from 1993

Extend the existing daily `duacs.zarr` with `adt` and `sla`, using the same
Copernicus DUACS release202411, dates and 0.125-degree grid as its four velocity
fields. Add a separate `era5-surface.zarr`: eight hourly 0.25-degree fields from
1993-01-01 through finalized ERA5 coverage (currently 2026-06-30 23:00).
Both final stores persist under `/mnt/home/jrusak/data/obs_full_range/prepared/`
and publish to `nyu-osn:emulators/jr7309/data/full_range`.

ERA5 fields: 10 m U/V wind, 2 m temperature/dewpoint, surface pressure, downward
solar/thermal radiation and total precipitation. The public ARCO ERA5 mirror
supplies the regular latitude/longitude grid. Its allocated array dates exceed
valid coverage: discovery uses `valid_time_stop`, excludes ERA5T, then verifies
actual selected coordinates. Radiation and precipitation retain provider units
and timestamps; no accumulation-to-flux conversion is applied.

`python -m ocean_preprocessing.obs_preprocessing.surface discover` freezes source
metadata, decoded coordinate hashes, native dates, variables and dtypes.
`run` streams 24 timestamps at a time, preserves decoded values/masks/units,
standardizes coordinate names/order and longitude to [0,360), and writes Zarr v2
with lossless Zstd compression. It reads back and exactly compares every value
and mask before saving each block receipt/checkpoint. Missing source chunks are
fatal, never implicitly replaced by fill values. There is no averaging,
regridding or raw source archive. Memory holds the current source block.

ERA5 uses four processes owning disjoint, complete time chunks in one shared
store; only the coordinator creates metadata and finalizes the store. Each
partition checkpoints independently, and changing partition counts is rejected.

Checkpoint resumes require matching code/inventory hashes. A clean time-budget
pause returns successfully for an `afterok` continuation; a source or validation
failure exits nonzero. Publication requires the final store, so exhausted
continuation segments cannot publish incomplete data. Jobs use native Grace
CPUs via `alpha`, the existing environment, and an isolated pinned checkout.

DUACS SSH is first prepared under `surface-staging/duacs-ssh.zarr`. After exact
coordinate matching, its arrays are hardlinked into the existing DUACS store;
velocity arrays and coordinate files are not rewritten. Consolidated metadata
is replaced atomically. The immutable original manifest remains available, and
`manifests/duacs-with-ssh.json` describes all six fields. The publisher only
accepts replacement of the specific previous remote inventory, retains that
inventory remotely, removes the obsolete success marker, verifies every uploaded
file with `rclone check --download`, then publishes `.zmetadata` and a new success
record. Staging hardlinks consume no second copy of the array bytes.

Prepared stores remain local after publication. The user has explicitly waived
raw retention for both this acquisition and the original three products. Delete
old raw downloads only after checking the corresponding prepared stores and
remote completion evidence; retain manifests, receipts, provenance and logs.
The original three-product results document records their historical retention
state at completion, not the later cleanup state.

## Initial execution evidence (2026-09-20)

Grace preflight 93531 verified public source and authenticated OSN access.
Pilot 93534 passed 21 focused tests, compared every decoded value/mask for
48 ERA5 hours and 48 DUACS days, and verified that full DUACS source coordinates
exactly match the existing velocity store. The four-process writer subsequently
adds a focused disjoint-chunk ownership test (22 tests total).

Raw cleanup job 93535 completed successfully: it removed 355,482,812,609 bytes
from `raw/{duacs,oisst,argo-iap}` after validating retained local stores and
matching remote inventories, consolidated metadata and historical success
records. Prepared stores, manifests, reports and logs remain. The cleanup
receipt is `reports/source-cleanup-20260920.json` under the work root.
