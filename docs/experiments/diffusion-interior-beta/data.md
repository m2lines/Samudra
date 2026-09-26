<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Campaign data staging

## Engaging migration

The user approved moving the campaign to Engaging and reusing its existing OM4
releases, including version differences from Torch. The destination is
`/orcd/pool/008/jrusak/diffusion-interior-engaging`. Its pool had approximately
520 GiB free when checked on September 26. Observation bundles and selected
checkpoints have been copied and verified: all 350 monthly and 416 annual payloads
passed checksum checks, and both the selected reference and pre-observation
source checkpoint hashes match. Each OM4 store passed completeness checks for
379,625 chunks, sampled decoding, and exact alignment of all 4,745 timestamps.
The coarse normalization copy matches its beta source across all 900 files.

Reuse `/orcd/data/abodner/002/jrusak/om4_onedeg_v3` and
`/orcd/data/abodner/002/jrusak/om4_halfdeg_v4`. The local staging audit records
each release and checks chunks, sampled decoding, and timestamp alignment; it
does not claim a full OSN read-back for these existing stores. Invoke
`scripts/stage_diffusion_beta.py --reuse-one-degree PATH` with the Engaging
root, half-degree, and observation paths. The coarse registration uses a campaign
wrapper: its `OM4.zarr` links to the existing Engaging fields, while its small
normalization stores are copied from the already verified beta staging. This
avoids the depth-based variable names in Engaging's older normalization stores
and preserves the prior forcing normalization. The audit verifies the required
80 scalar means/stds and records their payload hashes. Common observation-based
state scales remain unchanged.

All campaign arms use the same local coarse release. For H0/H1, surface inputs,
forcings, latent grid, observation targets, and common state normalization stay
fixed; only diffusion targets change. Comparisons with older Torch results must
identify the changed pretraining data. The selected observation checkpoint and
observation evaluation bundles remain unchanged.

The Engaging qualification launcher uses four independent tasks on one host,
initially requesting preemptible L40S GPUs. Production suitability depends on
measured memory and throughput. All allocated GPU time counts toward the same
576 GPU-hour campaign cap across both clusters.

## Original beta staging (historical)

Campaign root: `/mnt/home/jrusak/data/diffusion-interior-beta`.

| Role | Source / registered beta path | Verification |
| --- | --- | --- |
| One-degree OM4 inputs, original forcings and targets | OSN `emulators/am16581/data/2025-11/om4_onedeg_v3` → campaign `data/om4_onedeg_v3` | Verified: full source read-back comparison, chunk audit and exact timestamp alignment |
| Half-degree OM4 targets only | `/mnt/home/jrusak/data/om4_halfdeg` → campaign `data/om4_halfdeg_targets` | Verified: 379,625 declared chunks; six fields decoded at three times |
| Monthly observation samples, grid and scales | `/mnt/home/jrusak/data/obs_full_range/d-observation-pilot/samples` → campaign `data/observations` | Verified: all 350 manifest payloads; 243 train / 9 validation / 96 test samples |
| Continuous annual observations | `/mnt/home/jrusak/data/obs_full_range/d-observation-pilot/annual-instance-v1` → campaign `data/annual_observations` | Verified: all 416 recorded payload checksums across four origins |
| Baseline weights, architecture/config and selection evidence | Upstream final observation campaign → campaign `checkpoints` | Verified: final transfer baseline and its matched pre-observation OM4 source |

The beta `om4/v2026-09/om4_onedeg` release is deliberately not substituted for the
upstream `om4_onedeg_v3` release. Both old one-/half-degree products have 4,745
five-day frames spanning 1958–2022. The audit compares every numeric CF timestamp,
units and Julian calendar, and checks grids 180x360 versus 360x720. Training uses
only the upstream authorized dates; staging full temporal coverage does not
permit training on validation/test dates.

Existing datasets are registered by symlink, not duplicated or modified. The
half-degree store's forcings are never experiment inputs. Its means/stds are not
used to change the common state scales. The audit's half-degree check establishes
chunk completeness plus sampled decoding, not a full remote-source checksum
comparison; observation payloads and newly copied one-degree data have stronger
read-back/checksum verification.

Reproduce on beta with the existing preparation environment:

```bash
/mnt/home/jrusak/data/obs_full_range/envs/grace-arm64/bin/python \
  scripts/stage_diffusion_beta.py --existing-only
# After the one-degree rclone source read-back completes:
/mnt/home/jrusak/data/obs_full_range/envs/grace-arm64/bin/python \
  scripts/stage_diffusion_beta.py
```

`EXISTING_DATA_VERIFIED.json` and `DATA_READY.json` are written atomically only
after their checks pass. They record exact paths, byte counts, hashes, sampled
values and source provenance. Dataset readiness does not imply baseline readiness
or authorize using incomplete checkpoints.
