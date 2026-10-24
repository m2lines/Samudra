<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# CM4 quarter-degree preprocessing

Produce the unfiltered Gaussian 720×1440 CM4 release using the same processing
code and recorded package versions as the validated 1° release. The only intended
numerical changes are the target grid and its independently computed means/stds.
The full record contains 14,600 five-day means, 19 depths, all 81 physical channels
including wfo, native no-leap interval-end dates, and preserved grid metadata.
Physical fields remain float32 with one-time/all-depth/per-variable chunks,
Blosc Zstd level 5, and byte shuffle.

The pilot processes ten records to
`s3://m2lines-pubs/Samudra/experiments/cm4-quarterdeg-pilot-20261023/`.
It computes normalization over those ten records and independently reads every
physical chunk, checks expected chunk inventory, dates, masks, metadata and
codecs, and compares all 81 means/stds against float64 reductions.
A separate Rust-reader parity check must pass before promotion; pilot success
alone does not authorize bypassing that check. Pilot statistics do not substitute
for full-record normalization.

The authorized full destination is
`s3://m2lines-pubs/Samudra/v2026-09/cm4_quarterdeg/`.
The full stage refuses to start without `records/pilot-qualified.json` recording
both the successful pilot and exact Rust/Xarray parity for this source commit.
It refuses to overwrite an existing destination. It writes full-record
normalization and independently validates every physical chunk and statistic
before publishing `COMPLETE.json`. A partial store without completion is not
ready for training.

Torch run root: `/scratch/jr7309/cm4-quarterdeg-20261023`.
The pilot requests 16 CPUs/192 GiB/4h; production requests
32 CPUs/448 GiB/72h. Both are CPU-only; wall limits are ceilings, not estimates. The
processing environment reuses the verified package overlay at
`/scratch/jr7309/cm4-pr928-smoke-20261015/python_overlay`, rather than the drifted
shared base environment. Runtime checks enforce the original nine recorded
package versions. Grid files are checked against their original SHA256 manifest.
The checkout, source archive/checksum, environment versions, command, allocation,
read-back statistics, logs and completion records are retained.

User authorization covers the pilot followed by full processing after the
qualification gates pass. Durable callbacks/monitoring carry the task forward;
no further confirmation is needed for promotion. Model training is out of scope.

Torch login and DTN callback connectivity timed out during setup. The installed
callback remains best-effort, with a bounded startup check and exit delivery;
durable local reminders and scheduler/artifact checks provide follow-up even
when that endpoint is unreachable. Notification connectivity does not determine
scientific-job success or bypass any pilot qualification check.

## Submission record

On October 23, 2026, Torch accepted pilot **20750392** and production **20750393**.
Production has an `afterok:20750392` dependency and also checks the qualification
record before touching its destination. Both execute processing source
`ab0e5e4a4`; subsequent documentation changes do not alter that pinned checkout.
Earlier held submissions were canceled before execution because Slurm refused
to release them. The active jobs were submitted without holds.

The pilot has a two-hour fallback check; production has six-hour fallback checks.
The monitor verifies Slurm exit status and output artifacts rather than treating
a callback as proof of successful processing. The experiment record is
[PR #947](https://github.com/m2lines/Samudra/pull/947).

## Pilot results

Pilot **20750392** completed successfully on October 23, 2026, in **6m54s**.
Independent read-back passed for all ten records, including chunk inventory,
dates and bounds, masks, metadata, compression, and every normalization value.
All 81 physical channels matched Xarray exactly through the Rust reader, using
nonmonotonic and repeated time positions `[9, 0, 1, 1]`.

The three pilot stores occupy **1,351,947,113 bytes**. Linear scaling by the
record count suggests approximately **1.97 TB** for the full release; this is
a pilot-derived estimate, not a measured production size. Fixed grid overhead
and compression differences across the full record can change that estimate.
The largest recorded Slurm step MaxRSS was 20,673,576 KiB (19.7 GiB), which is
not a measurement of the sum of all worker peaks.

Production **20750393** cleared its dependency automatically after pilot success.
The pilot's normalization is not used for the final release.

## Full production results

Production **20750393** completed with exit status **0:0** on Torch `cs782`,
October 23 at 21:23:38 through October 24 at 03:36:51, 2026 (America/New_York),
for **6h13m13s** total. Processing source remained
`ab0e5e4a45dd852cfd0102a865b05d72abc35df0`.

Published artifacts:

- `s3://m2lines-pubs/Samudra/v2026-09/cm4_quarterdeg/CM4.zarr`
- `s3://m2lines-pubs/Samudra/v2026-09/cm4_quarterdeg/CM4_means.zarr`
- `s3://m2lines-pubs/Samudra/v2026-09/cm4_quarterdeg/CM4_stds.zarr`
- `s3://m2lines-pubs/Samudra/v2026-09/cm4_quarterdeg/COMPLETE.json`

The dataset contains **14,600 five-day means**, native no-leap timestamps from
**0151-01-06 through 0351-01-01**, Gaussian **720×1440**, and **19 depths**.
The nine physical variables represent 81 channels, including `wfo`.
The main store occupies **1,934,981,093,312 bytes (1.935 TB)** across **131,466
objects**, versus **4,904,697,887,536 logical bytes** (2.535× compression).

Independent full read-back passed for every expected physical chunk, native
dates and bounds, wet/dry-mask finiteness, metadata, float32/time-one/full-depth
layout, and Blosc Zstd level 5 with byte shuffle. All **81 means and 81 stds**
use the full **14,600-record** period and passed comparison against independent
float64 reductions. A fresh OSN read of `COMPLETE.json` confirms
`validated=true`, `steps=14600`, and `job_id=20750393`.

Measured stages were **4h03m14s** preprocessing, **39m09s** normalization,
and **1h29m05s** independent read-back. The largest recorded step MaxRSS was
108,169,480 KiB (103.2 GiB; not the sum of all worker peaks).
Callback delivery remained unavailable; durable local monitoring verified
scheduler status and the published artifacts. No Beta copy or training job has
been started as part of this production run.
