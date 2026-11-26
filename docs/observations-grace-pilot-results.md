<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Grace observation pilot results — 2026-09-18

All three bounded real-data conversions passed on Grace node `betagg41`, using
4 CPU cores and a 32 GiB allocation. The ARM environment, 14 pipeline tests,
Copernicus authentication, and a small OSN object write/read-back are verified.
No full-range Grace job or bulk OSN publication was launched. Torch was not
cancelled.

| Product | Timestamps | Download seconds | Prepare seconds | Compare seconds | Decoded bytes | Zarr bytes | Raw bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| duacs | 48 | 39.94 | 17.41 | 11.47 | 6,370,099,200 | 3,553,803,161 | 1,061,401,385 |
| oisst | 48 | 8.99 | 4.55 | 2.92 | 199,065,600 | 65,885,546 | 74,034,301 |
| argo-iap | 2 | 38.47 | 0.68 | 0.29 | 170,035,200 | 63,516,566 | 170,060,417 |

DUACS/OISST cover 2023-01-01 through 2023-02-17 (48 consecutive days); IAP
covers January and February 2023. DUACS midnight, OISST noon, and IAP calendar
month labels are retained. Every pilot value, NaN mask and coordinate was
compared with decoded, standardized source inputs; variable units were checked.
The comparison is against prepared source conventions, including the existing
float32 conversion for IAP. No averaging or spatial interpolation was applied.

## Execution and fixes

- Job `91343` stopped because system Python lacked `ensurepip`. Native ARM64
  `uv 0.9.7` now creates the environment without system package changes.
- Job `91344` passed 13 tests and Copernicus authentication, then failed an
  unnecessary rclone bucket-creation preflight. `--s3-no-check-bucket` fixed
  the upload to the existing OSN bucket; object-write permission was tested.
- Job `91348` at `76c3d9ae` successfully downloaded, converted and compared
  DUACS and OISST. Its overall Slurm state is FAILED because IAP exposed a
  longitude-normalization bug. Its two successful product reports are retained.
- IAP native longitude is 0.5..360.0. Commit `79f35e69` wraps the 360-degree
  column to 0 and sorts its data with it, without changing values. A regression
  test verifies the moved column and preserves the original input.
- Job `91349` at `79f35e69` reran IAP in a separate directory and COMPLETED
  with exit 0 in 59 seconds. All 14 tests and the OSN probe passed in this job.
  The earlier failed IAP staging store and raw files were preserved.

The peak pilot-process RSS values reported by `/usr/bin/time` were approximately
16.3 GiB for DUACS, 1.1 GiB for OISST, and 1.0 GiB for IAP. These include the
exhaustive pilot comparisons; they are not simultaneous job-wide peaks and do
not establish the memory requirement for a full annual Copernicus request.

## Retained files

All paths below are under `/mnt/home/jrusak/data/obs_full_range/`, visible from
both Alpha/Grace and beta home.

- `/mnt/home/jrusak/data/obs_full_range/pilot/20260918/prepared/duacs.zarr`
  Report: `/mnt/home/jrusak/data/obs_full_range/pilot/20260918/reports/duacs.json`
- `/mnt/home/jrusak/data/obs_full_range/pilot/20260918/prepared/oisst.zarr`
  Report: `/mnt/home/jrusak/data/obs_full_range/pilot/20260918/reports/oisst.json`
- `/mnt/home/jrusak/data/obs_full_range/pilot/20260918-iap-lonfix/prepared/argo-iap.zarr`
  Report: `/mnt/home/jrusak/data/obs_full_range/pilot/20260918-iap-lonfix/reports/argo-iap.json`

The native environment is `envs/grace-arm64`, and logs are
`logs/grace-pilot-91348.out` and `logs/grace-pilot-91349.out`. Tiny retained OSN
connectivity probes are under `emulators/jr7309/data/full_range_pilot_checks/`;
these are not uploaded observational datasets.

## Planning implications

Simple proportional scaling of preparation time and compressed bytes from this
small sample gives the following, **not full-run measurements or promises**:

| Product | Frozen full count | Scaled prepare time | Scaled prepared bytes |
| --- | ---: | ---: | ---: |
| duacs | 12,069 | 73.0 minutes | 893.6 GB |
| oisst | 16,438 | 26.0 minutes | 22.6 GB |
| argo-iap | 765 | 4.4 minutes | 24.3 GB |

The three prepared stores scale to about 0.94 TB in total. Retained raw data
adds about 0.36 TB; 2 TB of local headroom remains a reasonable initial budget.
The prepared DUACS representation is substantially larger than its packed raw
store because it preserves the decoded precision. Allow extra OSN headroom
beyond 1 TB. Compression, provider rates, metadata overhead, annual download
memory and full-range behavior can differ from this small sample.

Before a full migration, implement the Grace production driver and portable
publisher from the plan, use the IAP fix, preserve any frozen Torch inventory,
and get the user's switch decision. Keep the native environment and prepared
outputs; do not copy an x86 environment or delete existing data.
