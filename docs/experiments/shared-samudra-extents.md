<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Local DUACS velocity forecasting with a shared original Samudra backbone

Updated 2026-09-11. This campaign implements the user's latest scope: **velocities only, local DUACS, OM4 auxiliary data, original convolutional Samudra**. LLC is unavailable and deferred. The eventual objective remains DUACS SSH evolution, but a positive velocity result does not establish SSH forecast skill or recover the missing SSH datum.

The Rust loader is merged **locally** with main on the experiment branch. Do not merge its GitHub PR as part of this campaign. A separate `codex/duacs-velocity-runtime` branch at `9cd36b1bdcf4921fb027494afa4157e217818ccd` supplies the container environment. Experiment source changes use immutable code overlays after exact lockfile checks.

## Data and forecast contract

Use the existing local five-day DUACS store at `/scratch/am16581/data/obs/duacs.zarr`, rather than downloading observations. It contains absolute (`ugos`, `vgos`) and anomaly velocities, but no SSH. Its 598 timestamps span 2014-10-20 to 2022-12-24. Native grid: 0.125°, 1440 × 2880. The raw daily velocity archive is also available at `/scratch/am16581/data/obs_raw/duacs/`, covering 2014-10-18 through 2022-12-26.

Predict **absolute geostrophic velocity pairs**. Coarsen DUACS to quarter degree using cosine-area-weighted 2×2 averaging and require all four native cells to be valid. Derive matching geostrophic velocities from OM4 `zos` using the existing spherical-gradient metrics kernel. Actual OM4 currents include ageostrophic components; they are not silently substituted for the DUACS target. No temperature, salinity, SSH prediction head, or forcing input is included.

OM4 roots on Torch:

| Resolution | Local root | Use |
| --- | --- | --- |
| 1° | `/scratch/jr7309/data/om4_onedeg_v3` | Global auxiliary forecasting |
| ¼° | `/scratch/jr7309/data/om4_quarterdeg_v2` | Regional auxiliary forecasting |
| ½° | `/scratch/jr7309/data/om4_halfdeg_v4` | Optional resolution-interpolation diagnostic |

These are regriddings of the same underlying simulation, not independent model trajectories. Prepare derived fields under `/scratch/jr7309/data/velocity-transfer-v1/{duacs,om4_1deg,om4_quarterdeg}`; preserve the shared input archives.

Four input velocity maps span approximately 15 days. Predict the next map and shift the history using predictions during recursive forecasting. Supply actual history and next-step time differences because the OM4 calendar includes occasional six-day increments. Score steps 1, 2, 4, and 6 (nominal 5/10/20/30 days); write actual elapsed days for every forecast. Train two-step global rollouts and one-step regional examples initially.

The existing DUACS store uses centered five-day means on OM4 timestamps. This is a **retrospective mapped-field experiment**, not operational issuance-vintage forecasting. No claim of real-time causality should be made from this archive. Future target values and masks never enter forecast inputs.

Exclude the ±5° equatorial band for both sources, since the geostrophic calculation is ill-conditioned there. This campaign cannot establish equatorial forecast skill. Fit spatial mean fields, channel scales, validity masks, and monthly climatology using training dates only. Retain the source mean circulation as two static channels; normalize by each source's training temporal variability, not each crop's instantaneous statistics.

## Temporal separation

| Split | Allowed dates, including every input and target map |
| --- | --- |
| Training | Source start through 2018-09-30 |
| Validation | 2019-04-01 through 2020-09-30 |
| Test | 2021-04-01 through 2022-12-31, limited by source availability |

Require the entire four-input/six-target window to fit within its split, and reject missing timestamp intervals. The large gaps also reduce coupling from temporal averaging and reprocessed mapping. They do not turn retrospective maps into historical operational vintages. Discard OM4 dates after the training cutoff before preparation, so no simulation state from observation validation/test years enters training. Freeze the test set until model/task selection is complete.

## Shared model and boundaries

Use the repository's `UNetBackbone` and `ConvNeXtBlock` implementation. A DUACS 1×1 input/output adapter and an OM4 1×1 input/output adapter surround **one shared set of all U-Net weights**. The same OM4 adapters serve both resolutions. There are no Perceiver modules or learned fixed-grid positional embeddings.

Pilot widths are `[64, 96, 128, 192]`, with four levels, twofold channel expansion, and dilation `[1,1,1,1]`. These are configurable original Samudra blocks. Using unit dilation keeps the regional context manageable; the standard `[1,2,4,8]` dilation stack has a much larger receptive field. Keep widths and dilation fixed across every screened arm. Checkpoint activations and use bfloat16 on A100.

Inputs are four velocity pairs, four history-validity maps, spherical position (three channels), log physical dx/dy (two), a training wet mask, training mean velocity (two), four actual history offsets, seasonal sine/cosine, and next-step duration. The DUACS-only control has exactly the same target adapter/backbone capacity and conditioning.

Global examples use zonally periodic convolutions and upsampling. Regional examples use nonperiodic outer padding throughout the U-Net, including upsampling. Pad selection is an explicit per-forward argument, preserved during activation recomputation; shared module state is not mutated between tasks. Regional crops include 128-pixel halos around a 128×128 scored interior (384×384 input). Longitude can wrap only while extracting context across the real global date line. Test sensitivity to larger halos before interpreting fine-scale transfer.

BatchNorm remains matched across arms. Synchronize evaluation running statistics to the exact rank-zero values saved in checkpoints. If normalization interference dominates transfer, separate source statistics is a subsequent controlled ablation, not a silent change to one arm.

## Six-arm screen

| Arm | Training | Question |
| --- | --- | --- |
| D0 | DUACS only | Learned target-only control |
| D1 | DUACS + global 1° OM4 | Transfer from broad model context |
| D2 | DUACS + regional ¼° OM4 | Transfer from finer regional dynamics |
| D3 | DUACS + both OM4 tasks | Simultaneous sharing across resolution and extent |
| D4 | D3 with position and spacing channels zeroed | Does geometry conditioning help? |
| D5 | OM4 pretraining for 40% of compute, then DUACS fine-tuning | Sequential versus simultaneous transfer |

D4 replaces the earlier proposed physical-variable auxiliary heads to respect the velocities-only scope. D1/D2 choose DUACS/OM4 batches with probabilities 0.6/0.4; D3/D4 use 0.6/0.2/0.2. All ranks receive the same task each update, with independent dates/crops. Normalize losses by valid ocean area and channel scale; do not weight tasks merely by pixel count. Log actual source exposures and compute.

Compare equal GPU budgets first, and compare learning curves at equal DUACS exposure as a diagnostic. D1 versus D2 changes both extent and resolution, so it alone cannot isolate a resolution effect. Use a matched-geographic-box or regional 1° control later if the joint model improves.

## Evaluation and decisions

Select using 10-day DUACS vector RMSE in m/s on fixed validation dates. Vector RMSE is `sqrt(mean(du_error² + dv_error²))`; it is not the per-component RMSE. Log 5/10/20/30-day RMSE and persistence-relative skill, then evaluate all validation dates for finalists. Keep per-date numerator/denominator data so comparisons use exactly matched cohorts and weights.

Mandatory baselines: persistence, training-only monthly climatology, and damped linear extrapolation using the same history. The initial damping is fixed at 0.25; any tuning must use validation, then freeze the coefficient for test. Report Gulf Stream, Kuroshio, Southern Ocean, North Pacific gyre, and all valid ocean outside ±5°. Record bias, coverage, actual lead, checkpoint, preprocessing, and compute provenance. No extra future forcing is available to any method.

Confirm D0 and the best transfer arm with three seeds each. A useful provisional gate is at least 3% lower 10-day RMSE, supported across seeds and temporally blocked paired comparisons, without meaningful longer-lead or regional regression. Do not treat overlapping maps/crops as independent samples. With only two held-out years, confidence intervals are limited; report paired date-block differences and seed variation rather than a strong interannual generalization claim.

Follow-up diagnostics within contingency: larger regional halos, unseen crop dimensions, the unused ½° grid, variance loss/spectra, and source-specific normalization statistics. Improvements in OM4 losses alone do not count as observational transfer. A negative short-budget result also does not establish that multitask sharing cannot work.

## Compute and execution

Total ceiling: **1,344 A100 GPU-hours**, approximately eight A100s for seven days. Torch A100 nodes have four GPUs, so run matched jobs as pairs of four-GPU allocations rather than assuming an eight-GPU node.

| Stage | GPU-hours |
| --- | ---: |
| Loader, memory, boundary and throughput pilots | 96 |
| Six screening arms, 48 each | 288 |
| D0 and selected transfer arm, three seeds, 128 each | 768 |
| Final full validation/test evaluation | 96 |
| Contingency | 96 |
| **Total** | **1,344** |

Count failed/restarted allocations and evaluation toward the ceiling. Screening jobs request four A100s, 32 CPUs, and **64 GiB host memory initially**, with four Rust read threads per rank and zero PyTorch data workers. CPU preparation starts with eight CPUs and 32 GiB. Verify actual Slurm RSS and GPU peaks in the pilots before changing requests; these are initial requests, not measured requirements. Raw fields stay on CPU until the current bounded batch is transferred. A regional read currently loads a full two-channel frame through Rust, then crops; it does not preload the archive or all physical channels.

The general allocation currently accepts A100 requests through scheduler-managed preemption routing: omit a partition, request `--constraint=a100 --gres=gpu:4 --comment="preemption=yes;requeue=true" --requeue --signal=B:USR1@300`, and set `REQUEUE_ON_USR1=1` in the existing harness. This request passed `sbatch --test-only`; a normal A100 request without the preemption comment was rejected. Save checkpoints every 64 updates and resume automatically on `SLURM_RESTART_COUNT`. Follow the [Torch submission policy](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/) if routing changes.

Use the existing `slurm_apptainer_train.sbatch` custom-module support, a pinned SIF built from the runtime branch, and immutable experiment code layers. No source checkout is bind-mounted into running jobs. W&B project: `ocean_emulators/samudra-velocity-transfer`. Each run writes config, source/runtime provenance, JSONL learning curves, atomic resumable checkpoints, and matched evaluation CSVs.

Implementation entry points:

```bash
python -m samudra.experiments.velocity_transfer.prepare --input <local-zarr> --output <new-source-dir> --kind duacs
python -m samudra.experiments.velocity_transfer.prepare --input <local-OM4.zarr> --output <new-source-dir> --kind om4
torchrun --standalone --nproc-per-node=4 -m samudra.experiments.velocity_transfer.train --data-root <prepared-root> --output <new-run-dir> --variant D3 --gpu-hours 48
python -m samudra.experiments.velocity_transfer.evaluate --checkpoint <best.pt> --data <prepared-root>/duacs --output <new-score-dir> --split validation
```

Execution order: local Rust integration and tests → branch container → streaming CPU preparation → D0/D3 memory and throughput pilots → six-arm screen → validation selection → three-seed confirmation → held-out evaluation. Escalate neither model size nor training allocation until pilot evidence supports it.

## Submitted execution

Experiment branch: `codex/duacs-velocity-transfer`. Training, preparation, evaluation, and comparison code are pinned to `744e0d92864721bbfe4f35de5f552c381c87be2a`. The stage controller is separately pinned to commit `908ff7b1704d8353603f1eed78208b7f0ebd19cc` and verifies its deployed file checksum before every stage. Container build: [Actions run 34651347097](https://github.com/m2lines/Samudra/actions/runs/34651347097).

Initial Torch jobs submitted on 2026-09-11:

| Job | Purpose |
| --- | --- |
| 17410031 | Pull the branch container into a scratch-backed SIF |
| 17410101 | Build and verify the immutable experiment code layer |
| 17410102 | Prepare local DUACS velocities |
| 17410103 | Prepare global 1° OM4 geostrophic velocities |
| 17410104 | Prepare ¼° OM4 geostrophic velocities |
| 17410105 | D0 pilot: two A100s, 16 CPUs, 32 GiB host RAM |
| 17410106 | D3 pilot with the same resource request |
| 17412750 | Check pilots, then advance the campaign |

The pilots use 128 updates at most and target one GPU-hour each, including their small validation runs; each allocation has a 45-minute wall limit. The controller requires completed pilots, finite validation results, selected checkpoints, and conservative memory margins before submitting the screen. Subsequent jobs use two sequential lanes of four GPUs, limiting simultaneous training to eight A100s. Stage transitions check Slurm allocation accounting against the remaining budget. Failures or incomplete artifacts prevent advancement.

The controller selects the best transfer arm from D1–D5 on the fixed 10-day validation score, trains that arm and D0 with seeds 15/16/17, then evaluates both validation and test with matched baselines. It produces paired seed/date-block comparisons for each region and lead. No improvement is assumed in advance, and the test set is not used for selection.

Live execution record: `/scratch/jr7309/runs/velocity-transfer/campaign.json`. This contains job IDs for each later stage, source/container identities, resource accounting, selected arm, and any stopped-stage error. Per-run directories contain `config.json`, `run-provenance.json`, `history.jsonl`, `best.pt`, and resumable `checkpoint.pt`. Final comparisons are `validation-comparison.json` and `test-comparison.json` in the campaign directory. Slurm logs are `/scratch/jr7309/velocity-*.out` and `.err`; W&B is `ocean_emulators/samudra-velocity-transfer`.

Verification before cluster launch: full CPU suite 570 passed, two skipped, ten expected failures; subsequent focused comparison/controller tests passed; two-process training, validation, checkpoint resume, and evaluation smoke tests passed. A full 720×1440 two-step CUDA backward pass used about 4.6 GiB on the local GB10. This is a capacity check, not an A100 throughput measurement or a scientific result. The regional halo test verifies that doubling context leaves the scored interior unchanged in evaluation mode.
