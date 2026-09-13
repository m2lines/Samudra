<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Local DUACS velocity forecasting with a shared original Samudra backbone

Updated 2026-09-12. This campaign implements the user's latest scope: **velocities only, local DUACS, OM4 auxiliary data, original convolutional Samudra**. LLC is unavailable and deferred. The eventual objective remains DUACS SSH evolution, but a positive velocity result does not establish SSH forecast skill or recover the missing SSH datum.

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

The [prepared-data audit](velocity-transfer-artifacts/data-audit.json), run on CPU job `17423457` with script commit `6f517589`, verified positive finite normalization scales, monotonic dates, finite conditioning and climatology, and the OM4 training cutoff. DUACS has 280 training, 101 validation, and 118 test windows; each OM4 grid has 4,426 training windows. DUACS component scales are 0.128/0.119 m/s, versus 0.046/0.042 m/s at 1° and 0.074/0.069 m/s at ¼°. Sampled raw velocity magnitudes were checked for data quality across splits; no held-out forecast metrics or model selection were involved. The predefined tasks, normalization, and selection rule were retained.

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

Scheduling target: **results within roughly one week of wall-clock time**. The original 1,344 GPU-hours is an approximate sizing guide, not a hard cutoff; the user explicitly clarified this on 2026-09-12 UTC. The user clarified on 2026-09-12 UTC that the intended reference was RTX6000, and authorized other available preemptible Torch GPUs. Record actual allocation by GPU family; do not call mixed GPU-hours A100-equivalent compute. Use at most two simultaneous four-GPU training allocations.

| Stage | GPU-hours |
| --- | ---: |
| Loader, memory, boundary and throughput pilots | 96 |
| Six screening arms, 48 each | 288 |
| D0 and selected transfer arm, three seeds, 128 each | 768 |
| Final full validation/test evaluation | 96 |
| Contingency | 96 |
| **Total** | **1,344** |

Retain available allocation estimates for context without letting accounting delay experiments. New screening jobs request four RTX6000s, 32 CPUs, and **64 GiB host memory initially**, with four Rust read threads per rank and zero PyTorch data workers. CPU preparation starts with eight CPUs and 32 GiB. Verify actual Slurm RSS and GPU peaks in the pilots before changing requests; these are initial requests, not measured requirements. Raw fields stay on CPU until the current bounded batch is transferred. A regional read currently loads a full two-channel frame through Rust, then crops; it does not preload the archive or all physical channels.

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

Experiment branch: `codex/duacs-velocity-transfer`. Training, DUACS preparation, evaluation, and comparison code are pinned to `c4f36fa6194f9af9334d7f9924aace7c2c9c33f2`. OM4 preparation uses `488c572d09dfc20cd75030a33efcb226ca0d614e`; the subsequent change only fixes DUACS coordinate names and velocity unit attributes. The stage controller is separately pinned to commit `908ff7b1704d8353603f1eed78208b7f0ebd19cc` and verifies its deployed file checksum before every stage. Container build: [Actions run 34651347097](https://github.com/m2lines/Samudra/actions/runs/34651347097).

Current Torch jobs submitted on 2026-09-11 (superseding the initial staging attempts):

| Job | Purpose | State at this update |
| --- | --- | --- |
| 17415893 | Build and verify the immutable experiment code layer | Completed |
| 17415895 | Prepare local DUACS velocities | Completed in 2m25s; peak host RSS 26.7 GiB |
| 17415219 | Prepare global 1° OM4 geostrophic velocities | Completed in 16m36s |
| 17415220 | Prepare ¼° OM4 geostrophic velocities | Completed in 18m04s |
| 17415896 | D0 pilot: two A100s, 16 CPUs, 32 GiB host RAM | Completed 128 updates; 1m54s allocation |
| 17415898 | D3 pilot with the same resource request | Completed 128 updates; 2m11s allocation |
| 17415899 | Check pilots, then advance the campaign | Completed; all six screen arms submitted |

The runtime SIF is ready and its embedded source revision was verified. Initial publication failed at the scratch quota after image creation; the completed cached image was recovered by hard link. Only disposable OCI temporary image cache entries older than 30 days were removed, freeing approximately 56 GiB. Existing datasets, checkpoints, and published SIFs were retained. The pull helper now bounds image compression threads/memory and prefers node-local temporary storage. The first DUACS preparation attempt exposed the local store's `lat`/`lon` naming; preparation now accepts both these names and the raw archive's `latitude`/`longitude`, and the replacement job completed successfully.

The [D0 pilot](https://wandb.ai/ocean_emulators/samudra-velocity-transfer/runs/pilot-D0-s15) completed with finite training losses, validation metrics, and both checkpoint files. Peak CUDA allocation was 5.10 GiB and peak host RSS was 2.70 GiB on rank zero; Slurm reported 7.86 GiB MaxRSS for the training step. Total allocated compute was approximately 0.063 A100 GPU-hours. Its four-date pilot validation gave 10-day vector RMSE 0.110 m/s versus persistence 0.153 m/s. This small bring-up check does not establish transfer skill.

The [D3 pilot](https://wandb.ai/ocean_emulators/samudra-velocity-transfer/runs/pilot-D3-s15) also completed all 128 updates, including DUACS, global 1° OM4, and regional ¼° OM4 batches, with finite validation metrics and saved checkpoints. Peak CUDA allocation was 5.11 GiB and peak host RSS was 2.84 GiB on rank zero; Slurm reported 9.45 GiB MaxRSS for its training step. Both pilots together used 0.1361 allocated A100 GPU-hours. D3's four-date 10-day validation RMSE was 0.120 m/s versus D0's 0.110 m/s. These pilots have different DUACS exposure and only 128 updates, so this is not a matched transfer result.

The controller passed the pilot gates and submitted the screen at 23:22:59 UTC on 2026-09-11:

| Job | Arm | Dependency |
| --- | --- | --- |
| 17423438 | D0 | Ready; waiting for A100 resources |
| 17423439 | D1 | Ready; waiting for priority |
| 17423440 | D2 | After D0 allocation |
| 17423441 | D3 | After D1 allocation |
| 17423442 | D4 | After D2 allocation |
| 17423443 | D5 | After D3 allocation |
| 17423444 | Validate screen and submit seed confirmation | After all six screen allocations |

Screen jobs use four A100s, 32 CPUs, 64 GiB host memory, and a 13-hour allocation limit for each 48 GPU-hour target. The live requests were verified to retain preemption routing, automatic requeue, and immutable code commit `c4f36fa6`. The [screen-submission snapshot](velocity-transfer-artifacts/campaign-screen-submitted.json) records identities and accounting. Status checked at 23:32 UTC; scientific screen results remain pending.

The pilots use 128 updates at most and target one GPU-hour each, including their small validation runs; Each pilot retains a 45-minute wall limit. An attempted pending-job reduction to 15 minutes was rejected by the scheduler; verification confirmed D3 started at 23:20:45 UTC with its original request unchanged. The controller requires completed pilots, finite validation results, selected checkpoints, and conservative memory margins before submitting the screen. Subsequent jobs use two sequential lanes of four GPUs, limiting simultaneous training to eight GPUs. The later wall-clock clarification removed strict allocation-budget gates; planned per-run sizes remain unchanged. Failures or incomplete artifacts prevent advancement.

The controller selects the best transfer arm from D1–D5 on the fixed 10-day validation score, trains that arm and D0 with seeds 15/16/17, then evaluates both validation and test with matched baselines. It produces paired seed/date-block comparisons for each region and lead. No improvement is assumed in advance, and the test set is not used for selection.

Live execution record: `/scratch/jr7309/runs/velocity-transfer/campaign.json`. This contains job IDs for each later stage, source/container identities, resource accounting, selected arm, and any stopped-stage error. Per-run directories contain `config.json`, `run-provenance.json`, `history.jsonl`, `best.pt`, and resumable `checkpoint.pt`. Final comparisons are `validation-comparison.json` and `test-comparison.json` in the campaign directory. Slurm logs are `/scratch/jr7309/velocity-*.out` and `.err`; W&B is `ocean_emulators/samudra-velocity-transfer`.

Verification before cluster launch: full CPU suite 570 passed, two skipped, ten expected failures; subsequent focused comparison/controller tests passed; two-process training, validation, checkpoint resume, and evaluation smoke tests passed. A full 720×1440 two-step CUDA backward pass used about 4.6 GiB on the local GB10. This is a capacity check, not an A100 throughput measurement or a scientific result. The regional halo test verifies that doubling context leaves the scored interior unchanged in evaluation mode.

## Scientific report

Once the controller completes the campaign, generate the brief report from the complete evaluation directory (or a local copy of its JSON/CSV artifacts):

```bash
python -m samudra.experiments.velocity_transfer.report --campaign-root <campaign-directory> --output <new-report-directory>
```

This produces `report.md`, `metrics-by-seed.csv`, and `summary.json`. The report includes pooled physical vector RMSE against every baseline, regional 10/30-day transfer differences, individual seed results, paired seed/date-block intervals, coverage, Slurm allocation accounting, and scientific limitations. It recomputes comparisons from the raw score CSVs and rejects unfinished campaigns, missing forecast dates/regions/leads, and baseline mismatches across arms or seeds. Report generation does not change the pinned training source or consume GPU time. The final interpretation requires inspection of the completed artifacts; a passing pilot is not evidence of multitask transfer.

## RTX6000 scheduling update

The user corrected the intended hardware from A100 to RTX6000 and authorized whichever preemptible Torch capacity is available. The scheduler accepted immediate RTX6000 scheduling in `rtx6000_lzanna` with account `torch_pr_347_lzanna`, using the same preemption/requeue comment and no manually selected partition.

A100 D0 screen job `17423438` began during the availability checks and is retained. Only the five still-pending A100 transfer-screen jobs (`17423439`–`17423443`) and their pending confirmation controller were cancelled. They consumed no GPU time. The first four-GPU RTX6000 pilot `17425088` stalled during NCCL initialization and was cancelled, together with its dependent controller `17425089`. The symptom matched the documented Torch P2P hang (100% GPU utilization at about 1 GiB, no initial checkpoint). The controller now exports `NCCL_P2P_DISABLE=1` and `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` for RTX jobs. Its pinned commit is `001b783486d35a8513a4bfc3c1ee6ddfc75a4ee0`; training code remains unchanged at `c4f36fa6`.

Replacement RTX6000 pilot `17425402` passed all 128 updates in a 1m47s allocation. Peak CUDA memory was 5.11 GiB and peak rank-zero host RSS was 3.31 GiB. GPUs were identified as NVIDIA RTX PRO 6000 Blackwell Server Edition. The controller retained the running D0 job ID and submitted D1–D5 on RTX6000. Selection compares D1–D5 only, all on the same hardware. Both arms of the final three-seed confirmation use RTX6000, preserving the primary matched comparison. The A100 D0 screen is a reference run; the mixed-hardware screen is not interpreted as an equal-compute estimate of transfer versus D0. Allocation estimates include retired attempts and hardware pilots where available, grouped by GPU type. The later wall-clock clarification supersedes the original strict ceiling.

Current RTX screen jobs: D1 `17425534`, D2 `17425535`, D3 `17425536`, D4 `17425537`, D5 `17425539`; confirmation controller `17425540`. D0 remains `17423438`. At the post-pilot accounting gate, allocation use was 2.473 GPU-hours: 1.549 A100 and 0.923 RTX6000, including the failed NCCL attempt. Slurm duplicate/requeue records are included, so the failed attempt remains counted even when its most recent job record shows zero elapsed time. The two training lanes preserve the eight-GPU concurrency limit.

### Wall-clock target and checkpoint migration (2026-09-12 UTC)

The controller now treats allocation accounting as descriptive, with no hard GPU-hour gate. Planned 48-GPU-hour screens and 128-GPU-hour confirmations remain unchanged; at two continuously occupied four-GPU lanes they require about 5.5 days of training, before queue delays and evaluation. Controller source is committed in `be77378d` and deployed as `/scratch/jr7309/velocity-campaign-be77378d.py`.

A100 D0 job `17423438` was preempted after about 31 minutes and remained pending. An RTX6000 preflight reported immediate capacity, so its existing checkpoint was resumed as `17428065` on gr105. Original attempt config/provenance were preserved under `screen-D0-s15/attempt-17423438/`. W&B confirmed resuming the same run, and update 3960 verified continued training beyond the earlier update 2640. D1 `17425534` remains running on gr103. The scheduler rejected holding a pending dependency; pending D2/D4 jobs were therefore cancelled before retiring D0 and recreated safely as `17428066`/`17428067`. D3/D5 retain `17425536`/`17425539`. Confirmation controller `17428069` waits for all six current jobs.

D0 screening now contains both A100 and RTX6000 compute. It remains a reference only: transfer-arm selection compares D1-D5 on RTX6000, and the final D0-versus-winner three-seed confirmation uses RTX6000 for both arms. No transfer claim is based on this mixed-hardware D0 screen.

### Four screens complete; D5 checkpoint recovery (2026-09-13 UTC)

D0, D1, D2 and D3 completed their planned screening runs. Their best fixed-cohort 10-day validation vector RMSEs were respectively 0.10212144, 0.10874011, 0.10365186 and 0.10395321 m/s. These are preliminary selection scores, not held-out test results; the mixed-hardware D0 screen does not support a matched-compute transfer claim. D4 and D5 remain incomplete, so no transfer variant has been selected.

Slurm cancelled D5 job `17425539` after 2h25m (`CANCELLED by 0`, reported reason `QOSMaxGRESPerUser`). No training exception preceded the termination. Its checkpoint and original attempt metadata were preserved. The same RTX6000 request passed scheduler preflight, and replacement `17574711` was submitted with `--resume`. It briefly waited at the QoS limit, then started on gr102; update 83670 verified continuation beyond the prior update 76870. D4 `17428067` continues on gr105. Replacement confirmation controller `17574712` waits for these two remaining runs. The interruption added approximately 35 minutes of wall-clock delay without restarting training from scratch.
