<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Full-range observations: verified publication

**Retention update, 2026-09-20:** At the user's request, verified raw downloads
were removed by Grace job 93535 (355,482,812,609 bytes). All three prepared stores
remain local and published. Raw-retention statements below describe the original
completion audit. See [the surface acquisition record](observations-surface-1993.md).

Completed on 2026-09-18. All three native-cadence stores were prepared on
Empire AI Grace, validated, copied to OSN and verified by full file read-back.
No temporal averaging, regridding or filtering was applied. Standardization
matches the validated pilot, including IAP longitude 360° → 0° relabeling.

| Product | Frozen upstream coverage | Timestamps | Native grid | Prepared bytes | Retained raw bytes |
| --- | --- | --- | --- | ---: | ---: |
| duacs | 1993-01-01–2026-01-16 | 12,069 daily | 0.125°; 1440 × 2880 | 888,616,870,458 | 263,079,598,674 |
| oisst | 1981-09-01–2026-09-03 | 16,439 daily | 0.25°; 720 × 1440 | 22,400,359,221 | 27,355,258,613 |
| argo-iap | 1960-01-01–2023-09-01 | 765 monthly | 0.5°; 360 × 720; 41 depths | 24,393,095,476 | 65,047,955,322 |

Prepared stores total **935,410,325,155 bytes** (935.4 GB, decimal).
Retained raw archives total **355,482,812,609 bytes** (355.5 GB).
These totals exclude pilot outputs, environment, logs and auxiliary reports.
Raw archive totals include their inventory metadata.

## Locations

OSN stores:

- `s3://emulators/jr7309/data/full_range/duacs.zarr`
- `s3://emulators/jr7309/data/full_range/oisst.zarr`
- `s3://emulators/jr7309/data/full_range/argo-iap.zarr`

The configured rclone prefix is `nyu-osn:emulators/jr7309/data/full_range`.
Each store has a sibling `PRODUCT.inventory.json` and `PRODUCT.SUCCESS.json`.

All three final stores also remain under
`/mnt/home/jrusak/data/obs_full_range/prepared/`. Their raw sources remain under
`/mnt/home/jrusak/data/obs_full_range/raw/`: 34 DUACS yearly Zarr stores,
16,439 OISST daily NetCDF files, and 1,530 IAP temperature/salinity files for
765 months. No local data was removed after upload.

## Execution and validation

- Grace preparation job `91356`: completed, exit `0:0`, elapsed **1:49:08**.
- Grace publication job `91369`: completed, exit `0:0`, elapsed **0:30:00**.
- Both ran on `betagg41`, account `ny_lz1955_multiscale`, partition `grace`.
- Processing checkout: `438713002c43d1d340b8dc96ec78ddbf286d3c84`.
- Publication scripts: `7f8f82773c803f2315aa6e7f9a7560ad25158e41`, deployed
  separately so the running processing checkout was unchanged.

Local validation checked exact native timestamps, variable names, provenance,
coordinate order/convention, every expected data chunk, and sampled decoded
values. The real-data pilot compared all pilot values, masks, coordinates and
units against decoded sources. Full-range source values were not exhaustively
compared again.

Publication used `rclone check --download`: **zero differences** across
128,787 DUACS files, 10,973 OISST files and 786 IAP files. Consolidated metadata
was then published last and byte-compared separately. Final store object counts,
including that metadata, are 128,788 / 10,974 / 787.

An independent completion audit fetched all three OSN success records, compared
remote inventories and consolidated metadata byte-for-byte with the local
files, matched remote object counts and sizes to the local audit, and rechecked
that all expected raw archives remain present. Audit time:
`2026-09-18T21:37:23.266174+00:00`.

Evidence under `/mnt/home/jrusak/data/obs_full_range/`:

- `jobs-grace-20260918.tsv`
- `logs/grace-91356/` and `logs/publish-91369/`
- `reports/local-retained-sizes-20260918.json`
- `reports/publication-completion-20260918.json`

The Torch and Grace recurring check-in timers are disabled. The earlier Torch
files are retained, but the full final local copy described here is on Empire AI.
