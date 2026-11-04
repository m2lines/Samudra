<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Full-range observations at native cadence

The `full_range` CLI preserves each provider's native time cadence and grid,
standardizes `lat`/`lon` and longitude to `[0, 360)`, decodes packed values and
missing-data masks, and writes one consolidated **Zarr v2 store per product**.
A Zarr store is a directory/object prefix, not one large binary file.
There is no OM4 dependency, temporal averaging, interpolation, subsampling,
spatial smoothing, or regridding. The existing five-day preparation commands
remain available separately.

The destination is `nyu-osn:emulators/jr7309/data/full_range/`, with:

- `duacs.zarr`: `ugos`, `vgos`, `ugosa`, `vgosa`, preserving decoded precision.
- `oisst.zarr`: `sst` in Celsius; final OISST only.
- `argo-iap.zarr`: `temp`, `salt`, float32, with the existing 41 depth levels.
- `<product>.inventory.json`: frozen upstream file/catalogue inventory.
- `<product>.SUCCESS.json`: inventory digest and successful read-back verification.

DUACS `adt`/`sla` can optionally be included by setting `INCLUDE_SSH=1` **before
DUACS discovery**. They are excluded by default to match the existing metric
archive. This increases DUACS storage by approximately 50%.

## Coverage discovered on 2026-09-18

| Product | Full upstream range | Samples | Grid |
| --- | --- | --- | --- |
| DUACS, release 202411 | 1993-01-01 to 2026-01-16 | 12,069 daily | 0.125 degrees |
| OISST v2.1 final | 1981-09-01 to 2026-09-02 | 16,438 daily | 0.25 degrees |
| IAP/CZ16 temperature and salinity | 1960-01 to 2023-09 | 765 monthly | 0.5 degrees, 41 depths |

Discovery enumerates all final NOAA files and both IAP field directories,
rejects missing/duplicate dates or mismatched IAP coverage, and queries the
current Copernicus catalogue. It pins the DUACS dataset version. Download and
prepare consume the saved inventory; neither discovers a moving end date.
A new upstream snapshot needs a new manifest/raw/output directory. Final-product
providers can revise historical data in place: the inventory pins URLs and the
Copernicus version, not provider content checksums.

DUACS retains midnight labels, OISST noon labels. IAP months are reconstructed
from filenames as in the existing scripts. These are provider Level-4 analyses,
not raw sensor measurements. Current model-comparison metrics still require
identical model/observation timestamps; native daily stores do not automatically
make five-day model output a daily prediction.

## Environment and credentials

Both the checkout (`/scratch/jr7309/Ocean_Emulator`) and the Python venv
(`/scratch/jr7309/data/obs_full_range/venv`) live on scratch. No container
or dependency image is used. `~/Ocean_Emulator` may be a symlink to the scratch
checkout. Torch has Python 3.12. Use account `torch_pr_347_lzanna` and let
Slurm select the partition from the resource request and time limit, as
[NYU recommends](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/#partitions).
The short setup/check/discovery requests route to `cpu_short`; the longer
download/preparation requests route to `cs`. No GPUs, GHCR credentials, W&B key, or full training
container are needed. `STAGE=setup` builds a scratch-backed venv from
`scripts/requirements-observations.txt` and records `environment-freeze.txt`.
`STAGE=check` runs the small pipeline tests and
validates Copernicus authentication without printing credentials.
Commit these edits into a fixed checkout for the entire run so the stored git
revision identifies the processing code. A separate SHA-256 of the actual
processing modules also protects checkpoint resumes. Do not modify
that checkout or its environment while jobs are queued/running.

Required access:

1. **Torch SSH/Slurm:** reuse the existing authenticated `ssh torch` master.
2. **Copernicus Marine:** a free account, configured on Torch with
   `COPERNICUSMARINE_SERVICE_USERNAME` and
   `COPERNICUSMARINE_SERVICE_PASSWORD`, or a toolbox credentials file selected
   with `COPERNICUS_CREDENTIALS_FILE`. `copernicusmarine login` can create the
   default file once the environment exists. Keep credentials out of scripts,
   commits and command-line arguments; use a private file or inherited environment.
3. **OSN on dtn011:** working `nyu-osn` rclone remote for
   `https://nyu1.osn.mghpcc.org`, with list/read/write access to
   `emulators/jr7309/data/full_range`. Torch login and DTN share `~/.config/rclone/rclone.conf`. On 2026-09-18,
   their shared OSN entry was updated securely from the working local-machine
   configuration after the old key returned `InvalidAccessKeyId`. Keep secrets
   out of chat and job logs. Read access is needed for post-copy verification;
   write authorization still needs a destination write check.
4. NOAA and IAP downloads need no account. Public listing discovery needs no
   Copernicus login either; DUACS data download does.

Budget approximately **2 TB free scratch and 1 TB OSN headroom** initially,
including raw plus prepared copies and retries. These are conservative planning
budgets, not measured output sizes. The decoded retained fields total about
1.74 TB (DUACS dominates at 1.60 TB); compression will reduce actual disk usage.
The existing daily DUACS raw archive suggests about 268 GB for the full raw
velocity range; OISST raw is about 28 GB and IAP raw about 65 GB. New prepared
compression has not been benchmarked at full scale. OSN publication reads back
one additional compressed copy over the network for verification.
Torch's shared scratch filesystem has space, but the per-user quota query was
not permitted; verify quota/headroom before launching bulk downloads.

## Proposed CPU jobs

All requests below are CPU only. Time limits are allocation ceilings, not
promised completion times. Downloads resume at file/year boundaries; preparation
resumes at completed 24-timestamp block boundaries.

| Stage | CPUs | Memory | Time limit | Dependency |
| --- | --- | --- | --- | --- |
| Setup | 2 | 8 GB | 1 hour | none |
| Tests and authentication check | 4 | 16 GB | 15 minutes | setup |
| Discover each product | 2 | 8 GB | 1 hour | check |
| Download DUACS | 8 | 64 GB | 48 hours | DUACS discovery + credentials |
| Download OISST | 4 | 16 GB | 24 hours | OISST discovery |
| Download IAP | 4 | 16 GB | 24 hours | IAP discovery |
| Prepare DUACS | 16 | 128 GB | 48 hours | DUACS download |
| Prepare OISST | 8 | 32 GB | 12 hours | OISST download |
| Prepare IAP | 8 | 64 GB | 24 hours | IAP download |

Run these commands on Torch **after copying/committing the implementation into
an isolated fixed checkout**, choosing its actual path as `REPO_DIR`. This is
a fresh-run example: do not resubmit it over the active chain recorded in
`$WORK_ROOT/jobs-20260918.tsv`. Prepared stores and raw inputs remain on scratch;
publication copies them to OSN without removing the local data.

```bash
export REPO_DIR=/scratch/jr7309/Ocean_Emulator  # codex/observations-full-range
export WORK_ROOT=/scratch/jr7309/data/obs_full_range
export RAW_ROOT="$WORK_ROOT/raw"
export OUTPUT_ROOT="$WORK_ROOT/prepared"
mkdir -p "$WORK_ROOT/logs" "$WORK_ROOT/manifests"
HARNESS="$REPO_DIR/scripts/slurm_obs_full_range.sbatch"
common=(--parsable --account=torch_pr_347_lzanna
        --chdir="$WORK_ROOT" --output="$WORK_ROOT/logs/%x-%j.out")

setup=$(STAGE=setup sbatch "${common[@]}" --job-name=obs-env \
  --cpus-per-task=2 --mem=8G --time=01:00:00 "$HARNESS")
check=$(STAGE=check sbatch "${common[@]}" --job-name=obs-check \
  --dependency="afterok:$setup" --cpus-per-task=4 --mem=16G \
  --time=00:15:00 "$HARNESS")

for product in duacs oisst argo-iap; do
  # sbatch exports these values by default. Do not unset the Copernicus
  # credentials before submitting the DUACS download job.
  export PRODUCT="$product"
  discover=$(STAGE=discover sbatch "${common[@]}" \
    --dependency="afterok:$check" --job-name="obs-discover-$product" \
    --cpus-per-task=2 --mem=8G --time=01:00:00 "$HARNESS")
  case "$product" in
    duacs) download_cpu=8; download_mem=64G; download_time=2-00:00:00
           prepare_cpu=16; prepare_mem=128G; prepare_time=2-00:00:00 ;;
    oisst) download_cpu=4; download_mem=16G; download_time=1-00:00:00
           prepare_cpu=8; prepare_mem=32G; prepare_time=12:00:00 ;;
    argo-iap) download_cpu=4; download_mem=16G; download_time=1-00:00:00
              prepare_cpu=8; prepare_mem=64G; prepare_time=1-00:00:00 ;;
  esac
  fetched=$(STAGE=download sbatch "${common[@]}" \
    --dependency="afterok:$discover" --job-name="obs-fetch-$product" \
    --cpus-per-task="$download_cpu" --mem="$download_mem" \
    --time="$download_time" "$HARNESS")
  prepared=$(STAGE=prepare sbatch "${common[@]}" \
    --dependency="afterok:$fetched" --job-name="obs-prepare-$product" \
    --cpus-per-task="$prepare_cpu" --mem="$prepare_mem" \
    --time="$prepare_time" "$HARNESS")
  printf '%s discovery=%s download=%s prepare=%s\n' "$product" "$discover" "$fetched" "$prepared"
done
```

If a dependency fails, retry that stage with the same manifest and resubmit its
downstream dependent job against the new job ID. Do not rerun discovery over an
existing manifest. A staging store interrupted before its first checkpoint has
no completed data blocks; inspect and remove only that product's
`.<product>.partial.zarr` before retrying. Other interrupted writes are safely
overwritten from the last checkpoint. Do not run multiple copies for the same
product/output; the supplied harness uses `flock` to prevent this.

Large initial copies of existing raw archives, if desired, must also go through
`dtn011`, with `rclone copy` (never `sync`). OISST NetCDF files may seed the new
raw tree; they will be validated again. DUACS reuse needs matching provider
release, variables, complete year filenames and coordinates. The proposed jobs
work without any seed copy, avoiding dependence on the limited original archive.

## Publish from dtn011

Wait for the corresponding CPU prepare job to finish successfully. The
publication script revalidates the local store, reserves the remote inventory,
copies all chunks, compares every file with `rclone check --download`, then
publishes `.zmetadata` and the success record. It refuses a destination with a
different inventory. A retry of the same publication resumes copying without
removing remote objects. Consumers should require `<product>.SUCCESS.json`.

```bash
ssh torch
ssh dtn011
module load rclone/1.72.1
export REPO_DIR=/scratch/jr7309/Ocean_Emulator
export WORK_ROOT=/scratch/jr7309/data/obs_full_range
export PRODUCT=duacs  # repeat for oisst and argo-iap after their prepare jobs
nohup bash "$REPO_DIR/scripts/publish_obs_full_range.sh" \
  > "$WORK_ROOT/logs/publish-$PRODUCT.log" 2>&1 < /dev/null &
echo "Publication PID: $!"
```

The three CPU preparations may run independently. Publish products one at a
time initially to avoid competing transfers. Monitor logs from either the DTN
or login node; scratch and home are shared. Completion means successful
read-back checks and a success record, not just the launching shell returning.
