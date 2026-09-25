<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Beta data staging

Campaign root: `/mnt/home/jrusak/data/diffusion-interior-beta`.

| Role | Source / registered beta path | Verification |
| --- | --- | --- |
| One-degree OM4 inputs, original forcings and targets | OSN `emulators/am16581/data/2025-11/om4_onedeg_v3` → campaign `data/om4_onedeg_v3` | Copy followed by full source read-back comparison; in progress |
| Half-degree OM4 targets only | `/mnt/home/jrusak/data/om4_halfdeg` → campaign `data/om4_halfdeg_targets` | Verified: 379,625 declared chunks; six fields decoded at three times |
| Monthly observation samples, grid and scales | `/mnt/home/jrusak/data/obs_full_range/d-observation-pilot/samples` → campaign `data/observations` | Verified: all 350 manifest payloads; 243 train / 9 validation / 96 test samples |
| Continuous annual observations | `/mnt/home/jrusak/data/obs_full_range/d-observation-pilot/annual-instance-v1` → campaign `data/annual_observations` | Verified: all 416 recorded payload checksums across four origins |
| Baseline weights, architecture/config and selection evidence | Upstream final observation campaign → campaign `checkpoints` | Pending final upstream selection; no provisional weights substituted |

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
