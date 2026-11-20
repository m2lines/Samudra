<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Initial OBS-only baseline

OBS denotes observation supervision from OISST sea-surface temperature (SST)
and DUACS sea-surface height (SSH). The mixed arm combines these observations
with coarsened LLC4320 ocean-simulation snapshots.

The OBS-only six-hour model completed 2,000 optimizer updates on EAI Beta in
job `234497` (exit 0, allocation time 1 h 26 min 49 s, including qualification).
It used 16,000 observation-window draws and zero LLC training windows.
[W&B run 0oci13x4](https://wandb.ai/ocean_emulators/observational-transfer/runs/0oci13x4)
contains the training and validation history.

At the fixed 2,000-update checkpoint, exponential-moving-average (EMA) weights
beat persistence for SST at 48 and 72 hours, but not at 24 hours. SSH RMSE is
higher than persistence at every lead, with nearly equal scores at 72 hours.
The aggregate normalized RMSE is 0.026248 versus persistence 0.025361, about
3.50% worse. Lower RMSE is better.

| Lead | SST model / persistence (°C) | SSH model / persistence (m) |
| --- | --- | --- |
| 24 h | 0.2603 / 0.2197 | 0.009781 / 0.005885 |
| 48 h | 0.3325 / 0.3458 | 0.012933 / 0.011623 |
| 72 h | 0.3707 / 0.4265 | 0.017144 / 0.017129 |

These scores use eight fixed validation origins from the OBS validation split
(2013-11 through 2014-07). For each channel and lead, squared errors are
wet-area averaged on identical joint origin/target valid support, pooled equally
across origins, then square-rooted. Persistence repeats the last daily surface
analysis. Normalized RMSE divides physical RMSE by a fixed channel standard deviation
fitted on observation-training support: approximately 10.0999°C for SST and
0.641986 m for SSH. The aggregate averages these normalized RMSEs equally over
the two channels and three leads. This is one seed (1729) and a small validation cohort, with
prescribed future ERA5 weather. No test-period or long-rollout result is claimed.

The origins are evenly spaced across eligible complete windows: 2013-11-09,
2013-12-16, 2014-01-22, 2014-02-28, 2014-04-07, 2014-05-14, 2014-06-20 and
2014-07-28, all at the midnight analysis slot. Their forecast windows do not
overlap. No uncertainty intervals are available; differences are descriptive.

The saved 1,000-update EMA checkpoint has aggregate normalized RMSE 0.035059.
It used 8,000 OBS-window draws, matching the mixed run's planned OBS exposure at
2,000 updates. The primary mixed comparison uses both fixed 2,000-update EMA
checkpoints. The [completed mixed comparison](mixed-versus-obs-only.md) now reports both
controls; this record preserves the OBS-only measurements. The
validation-selected best EMA checkpoint in this run also occurs at update 2,000.
The [machine-readable curves](obs-only-validation.csv) contain every validation
checkpoint from update 100 through 2,000.

The model retains the mixed arm's 205-channel architecture, 29 six-hour history
frames, six-hour dynamics, daily observation targets, batch eight, optimizer,
normalization, geometry and holdouts. Its ocean inputs are surface temperature
and SSH; missing subdaily observation slots are masked. It predicts U/V
velocity, temperature and salinity at 51 LLC depths plus SSH, totaling 205
channels. LLC masks and interior normalization are
retained, but interior state variables receive no direct training targets.
LLC diagnostic validation uses an untrained LLC task adapter and is excluded
from the comparison above.

The production allocation used all four GPUs. Its inline qualification completed
eight OBS updates (64 draws), with a checkpoint after four and a resumed four;
LLC exposure remained zero. Qualification used separate outputs; production
started from scratch, so those eight updates do not influence the reported
model. Redundant pending test job `234498` was cancelled
without allocating GPUs. The production receipt verifies 2,000 OBS updates,
16,000 fresh window draws and no replay. Here fresh means sampled outside the
replay pool; window draws may repeat dates. All 20 planned validations are present.

Runtime code is `83df2678f91c29aba4187e79fdc9e8e46bfe0d54`, matching the mixed
arm. The archive SHA-256 is
`a9402e3edf174b876358db7736a984e46672088e76417f0f7b9114d246aaed71`.
Prepared-data proof SHA-256 is
`3fb27c8fe8d926e82b2aa2758aac9bd1100a20d86ac4baa44733de037895cb7e`.
Full metrics SHA-256 is
`55ebe5ea1f59f2940ef7eca01e6de615db78773b9d87ba3d7812409f9f0893cc`.

Receipts, logs and checkpoints remain under
`/projects/ny/lz1955/multiscale/jrusak/runs/obs-only-six-hour-20261120` on EAI.
The final and exposure-matched checkpoints are
`outputs/obs-only-six-hour-v1/saved_nets/main-2000.pt` and `main-1000.pt`;
use their EMA state for the comparisons above. No checkpoints are stored in git.
