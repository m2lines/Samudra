<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Proposed migration of full-range observations to Empire AI Grace

Planning snapshot: 2026-09-18. No Empire AI jobs have been submitted, no
credentials copied, and no Torch jobs cancelled for this plan. The existing
Torch monitor remains active. A switch is a separate user decision.

## Important routing distinction

The CPU-only Grace-Grace nodes `betagg01` through `betagg60` use the **Alpha/Grace
Slurm scheduler**, partition `grace`. They are not in the scheduler reached by
`ssh beta`. Empire AI's [systems guide](https://empireai.freshdesk.com/support/solutions/articles/157000374441)
and [Grace submission guide](https://empireai.freshdesk.com/support/solutions/articles/157000374494)
document this separation. Grace nodes have 144 ARM CPU cores and 478 GiB RAM.

Live inspection of beta showed only the GPU partition `beta`, with a four-GPU
minimum for the user's available production/test QoS tiers. Those are not the
resources to request for this workflow. The existing `alpha` SSH alias resolves
to `alpha.empire-ai.org`, but a noninteractive connection failed host-key
verification. Resolve the correct Alpha login and verify its host key before
claiming live Grace availability or a working job submission.

The account `ny_lz1955_multiscale` is associated with `jrusak` on beta. Confirm
the account separately on Alpha/Grace; it is a candidate, not yet a verified
Grace account. Use `sbatch --test-only` for the final requests after checking
`sinfo -p grace`, `scontrol show partition grace`, and user account/QoS limits.

## Storage and credentials

Use the user's Empire AI home allocation, with this proposed layout:

```text
/mnt/home/jrusak/data/obs_full_range/
  code/Ocean_Emulator/       fixed branch checkout
  envs/grace-arm64/          native Python 3.12 environment
  cache/grace-arm64/         package/build cache
  manifests/                frozen inventories
  raw/                      retained upstream files
  prepared/                 retained duacs.zarr, oisst.zarr, argo-iap.zarr
  logs/                     stage logs and job records
  pilot/                    small, separate validation outputs
```

The user reports approximately 100 TB of home capacity. On beta, `df` on the
home path reports the entire shared `/mnt/home` filesystem (about 19 PB total,
14 PB free), not the user's quota; it does not independently verify 100 TB.
The project path `/projects/ny/lz1955/multiscale` separately reports about
101 TB free. Use home as requested, confirm the individual quota/headroom and
that Grace sees the same home files before launch. Initially budget 2 TB for
raw plus prepared data and staging; refine this from the pilot.

Retain both raw and prepared data at completion. OSN publication is a copy,
never a move or cleanup. Keeping these files on Empire AI does not automatically
create another completed copy on Torch scratch.

Credential findings from beta:

- Copernicus username/password environment variables are absent. At migration,
  securely provision the already-provided credential source from Torch and check
  authentication inside a Grace allocation. Do not log secrets or copy a whole
  shell profile. The names remain `COPERNICUSMARINE_SERVICE_USERNAME` and
  `COPERNICUSMARINE_SERVICE_PASSWORD`.
- `~/bin/rclone` and a `nyu-osn` remote exist, but a bounded listing of
  `emulators/jr7309/data/` returned HTTP 403 AccessDenied. Compare endpoint and
  scope, then securely provision the known-working OSN configuration if needed.
  Verify read/list and a small isolated destination write/read-back before any
  bulk publication. Do not infer that the mere presence of a config means access.
- NOAA and IAP require no accounts. Test provider and OSN connectivity from the
  actual Grace compute node, not only the login host.

## Software and implementation changes

Keep the scientific processing unchanged: the same retained fields, provider
versions, native cadence, timestamps, coordinates, units and masks; no temporal
averaging, regridding or filtering. Keep the existing frozen-inventory and
checkpoint rules and Zarr v2 output format.

1. Use a fixed checkout of `codex/observations-full-range` (current processing
   revision includes routing fix `0bd0cd8e`). Create a fresh ARM64 Python 3.12
   environment; do not transfer Torch's x86 venv and do not use a container.
2. Check the pinned requirements on ARM. PyPI currently provides candidate
   Python 3.12 ARM64 or universal wheels for all direct pins except
   `numcodecs==0.15.1`; build that pin natively with a compiler and Python
   headers. This wheel check is not a completed dependency resolution/import
   test. Beta's login host has Python 3.12, GCC, and Python development headers;
   verify those prerequisites on the Grace compute allocation. Preserve versions
   first; only change a pin if necessary and verified.
3. Reuse `scripts/slurm_obs_full_range.sbatch` with explicit `REPO_DIR`,
   `WORK_ROOT`, `OBS_ENV`, `RAW_ROOT`, and `OUTPUT_ROOT`. Add a Grace wrapper
   that supplies the verified account, `--partition=grace`, and appropriate
   QoS, overriding the Torch account default. Make the pip-cache override
   configurable so it uses `cache/grace-arm64`.
4. Add a small production driver that runs the three product pipelines in one
   allocation, each in order: discovery -> download -> prepare -> validate.
   Give each subprocess its own log. Check every exit status; a failed product
   must not be reported complete because a sibling succeeded. Retain successful
   products and checkpoints for targeted retries.
5. Make the publisher's hard-coded `module load rclone/1.72.1` optional or
   configurable and use the verified ARM rclone binary on the approved transfer
   host. Keep full read-back comparison, metadata-last publication, inventory
   ownership checks and success records. The existing Torch DTN name is not an
   Empire AI transfer endpoint.

## Proposed jobs after a switch decision

| Job | Resources | QoS / wall-time ceiling | Purpose |
| --- | --- | --- | --- |
| ARM setup and pilot | 1 node, 4 CPUs, 16 GiB | `test`, 1 hour | Native environment, 13 pipeline tests, credentials, small real-data samples |
| Full product pipelines | 1 node, 32 CPUs, 256 GiB | `standard`, at most 48 hours; reduce after pilot | Three concurrent product pipelines, persisted outputs |
| OSN publication | Approved transfer host, or permitted Grace allocation | Size after measuring transfer throughput | Copy, read-back verification, retain local files |

One Grace node has enough documented memory for this initial split:
DUACS preparation 16 workers / about 128 GiB budget; OISST 8 workers / 32 GiB;
IAP 8 workers / 64 GiB. The 256 GiB allocation leaves additional headroom. These
are starting resource budgets, not measured peak usage. Downloads start with
four workers per product. A single allocation avoids relying on packing three
separate node-billed jobs; no GPUs or MPI are needed.

The pilot should convert about 48 daily timestamps for DUACS and OISST, plus two
IAP months, into separate pilot stores. Exercise more than one preparation block
for the daily products. Compare source/output timestamps, coordinates, units,
NaN masks and values; record decoded throughput, compressed bytes and peak RSS.
Choose the production wall time and concurrency from these measurements rather
than treating 12–48-hour ceilings as runtime estimates. The full-store time
range remains the complete frozen inventory, not the pilot range.

Illustrative submission shape, **not yet a verified command**:

```bash
sbatch --account=<verified-grace-account> --partition=grace --qos=standard \
  --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=256G \
  --time=<pilot-derived-ceiling> \
  --chdir=/mnt/home/jrusak/data/obs_full_range \
  --output=/mnt/home/jrusak/data/obs_full_range/logs/production-%j.out \
  <grace-production-wrapper>
```

## Switch sequence and acceptance

1. Resolve Alpha/Grace access, actual account/queue availability, shared-home
   visibility, home headroom, and transfer policy. Prepare code changes and
   credential provisioning for review. Leave the Torch chain in place.
2. After the user selects migration, run the small Grace pilot. Keep Torch
   available as a fallback until the pilot passes; stop here on access, ARM
   build, provider connectivity, authentication or data-validation failures.
3. Before cancelling Torch, inspect the latest job record and actual progress.
   Preserve any frozen manifests and reusable completed downloads. If a frozen
   inventory already exists, carry it over unchanged so the migration preserves
   the upstream snapshot. Otherwise discover and freeze once on Grace.
4. After a successful pilot, cancel only the current observation chain on
   Torch, then submit the production allocation. Redirect the existing monitor
   to the new host and job records. Avoid two active writers or two bulk download
   chains. Do not remove existing Torch files.
5. Validate all three full local stores against their inventories and report
   observed ranges, counts, bytes and paths. Publish to
   `nyu-osn:emulators/jr7309/data/full_range` through the permitted transfer route
   with full read-back verification. Report local completion separately from
   verified OSN publication; retain the local stores in either case.

The blocking planning detail is currently **working Alpha/Grace access**, not
the scientific pipeline. Home quota, runtime compatibility and compute-node
network access also need their stated checks before treating the plan as ready
to execute.
