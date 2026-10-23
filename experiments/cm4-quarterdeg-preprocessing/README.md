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
The pilot requests 16 CPUs/192 GiB/4h; proposed production resources are
32 CPUs/448 GiB/72h, subject to the measured pilot. Both are CPU-only. The
processing environment reuses the verified package overlay at
`/scratch/jr7309/cm4-pr928-smoke-20261015/python_overlay`, rather than the drifted
shared base environment. Runtime checks enforce the original nine recorded
package versions. Grid files are checked against their original SHA256 manifest.
The checkout, source archive/checksum, environment versions, command, allocation,
read-back statistics, logs and completion records are retained.

User authorization covers the pilot followed by full processing after the
qualification gates pass. Durable callbacks/monitoring carry the task forward;
no further confirmation is needed for promotion. Model training is out of scope.
