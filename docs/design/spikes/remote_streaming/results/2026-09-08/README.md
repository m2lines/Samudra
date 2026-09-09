<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# LLC remote-layout tournament results: 2026-09-08

These files record the MIT ORCD A--E layout tournament over the immutable
eight-time, all-51-level, 2880-square real `Theta` fixture.

- `candidate_summary.csv` contains layout, storage, object-count, compression,
  and encoding measurements.
- `reader_validation.csv` records full-array correctness and validation time
  for Zarr Python, zarrs-python, and TensorStore.
- `reader_workload_summary.csv` contains median workload timings.
- `reader_workload_trials.csv` contains every individual timed trial.
- `raw/` preserves source JSON results and the exact Zarr metadata for every
  candidate. The multi-GiB physical data objects remain on ORCD scratch.

The `observed_data_*` columns from the Zarr Python local-store tracker are
retained for diagnostic provenance but have `byte_metrics_valid=false`.
Zarr's nested sharding path can bypass the wrapper used by this spike, so those
columns undercount partial payload reads for candidates D and E. Timings,
stored sizes, object counts, metadata, and equality checks are valid. Exact
range amplification will be measured at an HTTP/S3 server boundary.

ORCD run directory:

```text
/orcd/scratch/orcd/008/merose/llc_remote_layout_v3_20260908
```

Primary jobs:

```text
Python encode/read A--E: 22338978
Corrected local trace:   22339684
Native environment:      22339748
Native A--E readers:     22339749
```
