<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Full-range observation preparation on Grace

The user authorized the Torch-to-Grace switch on 2026-09-18 after the
[successful native ARM pilot](observations-grace-pilot-results.md). Use the
Alpha/Grace scheduler (`ssh alpha`), account `ny_lz1955_multiscale`, partition
`grace`, excluding `alphagh01`. No GPUs or container are used.

The production harness `scripts/slurm_obs_grace_full_range.sbatch` requests
32 CPUs, 256 GiB RAM and a 12-hour ceiling. It runs three independent product
pipelines concurrently, with download → prepare → validate in each pipeline.
Preparation uses 16/8/8 threads for DUACS/OISST/IAP; each downloader uses four
workers where supported. Twelve hours is a conservative scheduling limit,
not a measured full-run duration. The validated ARM environment is reused
without installing or changing dependencies.

All paths are under `/mnt/home/jrusak/data/obs_full_range`:

- `manifests/{duacs,oisst,argo-iap}.json`: unchanged frozen inventories copied
  from the final Torch discovery jobs before migration.
- `raw/`: retained source downloads.
- `prepared/{duacs,oisst,argo-iap}.zarr`: retained final native-cadence stores.
- `logs/grace-JOBID/`: per-product logs, stage/exit status, dependency versions
  and inventory hashes. Successful siblings are retained if another fails.

| Product | Frozen coverage | Timestamps |
| --- | --- | ---: |
| DUACS release 202411 | 1993-01-01–2026-01-16 | 12,069 daily |
| OISST final | 1981-09-01–2026-09-03 | 16,439 daily |
| IAP | 1960-01–2023-09 | 765 monthly |

The later Torch discovery includes one more final OISST day than the pilot's
parent inventory. Migration preserves this new frozen inventory byte for byte.
Scientific processing is unchanged from the pilot, including the IAP 360° → 0°
longitude correction. Values, masks, native grids and cadence are preserved;
there is no averaging or regridding.

Submit with `WORK_ROOT`, `REPO_DIR`, and `EXPECTED_COMMIT` exported. Use an
absolute log path and record the returned ID. The harness refuses a changed
revision, tracked changes, or a concurrent production job. Never modify the
checkout or environment while it runs. Resumes must keep the same inventories
and processing hashes; do not remove partial stores to bypass those guards.

Validation checks exact time coverage, schema/provenance, every expected chunk
object, and sampled decoded values. The pilot compared every pilot value and
mask against sources; full-store validation does not reread every source value.

The Torch chain was canceled after its setup/check/discovery jobs completed
and two downloads briefly started. Existing Torch files are retained. The
recurring Torch check-in was explicitly stopped by the user and is not
re-enabled by this harness. No periodic Grace monitor is installed here.

OSN publication is a separate outstanding step after local preparation. The
harness neither uploads to `emulators/jr7309/data/full_range` nor removes local
data. The approximately 2 TB working budget is inferred from the pilot; the
user reports 100 TB home capacity, but filesystem-wide free space is not an
independent measurement of the user's quota.

## Authorized monitoring and publication

The user subsequently authorized an active goal to monitor processing, recover
straightforward failures, raise complex blockers, and publish on completion.
The Grace check-in runs every 15 minutes in the existing task and stays quiet
unless progress is meaningful, recovery occurs, work completes, or user input
is needed. The old Torch check-in remains disabled.

`scripts/slurm_obs_grace_publish.sbatch` can be submitted with
`--dependency=afterok:PREPARATION_JOB --kill-on-invalid-dep=yes`. It revalidates
all three stores before publishing, then runs the portable publisher with the
native ARM rclone. Publication uses full `rclone check --download`, consolidated
metadata last, and a byte-verified success record. A processing failure cancels
the dependent upload job; recovery must recreate the dependency chain.

Keep the running processing checkout pinned. Deploy the publication shell
scripts into a separate versioned operations directory, setting `PUBLISH_SCRIPT`
to that fixed copy and `REPO_DIR` to the original processing checkout. The
publisher's `RCLONE_MODULE=''` selects the already available Grace binary;
`RCLONE_S3_NO_CHECK_BUCKET=true` avoids an unauthorized bucket-creation check.
Neither successful publication nor monitoring deletes local data.
