<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Matched global observation-only comparator

Authorized September 30: train one fresh observation-only model and capture
intermediate checkpoints. Add fixed 8k and 16k observation checkpoints to the
[focused global physical-only report](global-physical-day30-2026-09-30.md).

**Small conditioned scratch global** has exactly the mixed global model's
62,966,546 parameters, 77 physical state channels, small InstanceNorm initializer
and D processor, and source-specific input adapters. It starts from random weights
and performs **zero OM4 / 16,000 observation optimizer updates**. The OM4 adapter
remains inactive; no OM4 weights or optimizer history enter this run. The existing
implementation computes OM4 validation diagnostics, which affect neither the
observation selection objective nor the fixed learning rate.

Inputs, observation losses, validation and test scores use all available latitudes.
Land/missing labels remain masked. Retain observed-only copying, zero missing
placeholders, learned completion, observation-derived normalization, geographic/
seasonal context, ERA5 forcing adapter and the same reconstruction/completion losses.
Seed 1729; AdamW 1e-4, weight decay 0.01, clipping 1, effective batch eight,
no warmup and no separate reconstruction phase. Per-task deterministic sampling
matches the mixed arm's observation stream at equal observation counts.

| Literal report name | OM4 updates | Observation updates | Comparison with mixed endpoint |
|---|---:|---:|---|
| Obs-only: 8,000 obs | 0 | 8,000 | Equal observation exposure; half the optimizer updates |
| Obs-only: 16,000 obs | 0 | 16,000 | Equal total optimizer updates; twice the observation exposure |

Retain immutable checkpoints at 0, 10, 25, 50, 100, 250, 500, 1k, 2k, 4k, 6k,
8k, 10k, 12k, 14k and 16k observation updates. Validation remains the global
integrated-plus-spectral observation objective every 100 updates and at milestones.
The validation-selected checkpoint is retained separately; fixed-budget comparisons
do not substitute it for 8k or 16k.

Monthly and annual evaluations for each endpoint depend on successful full 16k
training. Runtime additionally requires a completion marker and exact checkpoint
lineage. Monthly tests cover all 96 origins in 2015–2022; annual tests use January
2015/2018/2021, 73 autoregressive five-day steps, and prescribed ERA5. Report
initializer/day30/annual maps, day-30-only broader spectra against the identical
OM4/coarse/native references, integrated metric components, and undetrended annual
SST/SSH/OHC means with persistence/climatology. Preserve the OHC representation
check and existing 1:1 map pixels. Intermediate snapshots permit later diagnostics.

Training/evaluation producer remains
`79025a163817a577ab81af95b401f5cb0563cd12`, exactly matching the mixed physical-only
arm. Its existing global fitting and resume qualifications (jobs 18811858/18811859)
are reused after checking producer, data/reference hashes and model contracts.
Qualification weights are not loaded into production. The submission records
successful accounting for these dependencies and runtime checks them again.
No training implementation or scientific protocol change is required.

Root: `/scratch/jr7309/runs/2026-09-30-observation-global-scratch`.
Launcher: `scripts/submit_observation_global_scratch.py`.
One RTX6000 GPU on LZanna; online W&B and 15-second utilization telemetry.
Requested caps: 24 training GPU-hours plus six evaluation GPU-hours; charge actual
allocated time including retries. Internal training cap 23.5 hours; latest deadline
October 3 at 02:00 UTC, allowing routine recovery without changing the contract.
The completed mixed arm's observation throughput suggests roughly 15–17 hours
plus queueing, to be refined after bring-up.

Torch access and data paths were checked. Personal scratch usage is 4.36/5 TB,
with about 640 GB headroom. The previous mixed arm's checkpoints occupy 12 GB;
reserve approximately 30 GB for new checkpoints/evaluations and processing.
Reuse the verified existing code overlay and cached SIF. No dataset transfer,
cleanup, extra seed or model variant is included. Monitor with interruptible
waits; keep the old timer disabled.

## Verified bring-up

September 30 at 19:28 UTC: job **18890321** is running on gr102, an RTX PRO 6000
Blackwell GPU. Structured training reached 54 observation updates with zero OM4
updates, finite losses/gradients, online W&B and confirmed producer/global flags.
Checkpoints 0/10/25/50 are already written. The 50-update global observation
validation composite is 1.9116; this is an early validation result, not test skill.

| Evaluation | Job | Dependency |
|---|---|---|
| Obs-only 8k monthly | 18890323 | Successful 18890321 |
| Obs-only 8k annual | 18890324 | Successful 18890321 |
| Obs-only 16k monthly | 18890360 | Successful 18890321 |
| Obs-only 16k annual | 18890437 | Successful 18890321 |

All four evaluations are pending on that dependency. The runtime lineage check
will separately require full training completion and 0/8k or 0/16k task counts.
[Startup provenance](artifacts/2026-09-30-global-scratch/startup.json.gz) includes
actual manifests, qualification/reference hashes, source overlay, job records,
first checkpoints, telemetry and initial accounting. Reused qualification GPU
cost belongs to the preceding wave, not this allocation. Initial cold-cache
updates were slower; refine the duration estimate after cache warming.

[W&B run](https://wandb.ai/ocean_emulators/observational-transfer/runs/2f252fb3ca72).
