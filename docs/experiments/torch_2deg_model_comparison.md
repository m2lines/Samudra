<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Torch full-data 2-degree model comparison

## Question

Can the native-SDPA Perceiver encoder plus direct output-query decoder become a
competitive single-scale predictor before multi-scale training? Which remaining
gap comes from the attention implementation, the Perceiver representation, or
the surrounding SamudraMulti architecture?

## Controlled budget

All runs use the anonymous public stores under
`s3://m2lines-pubs/Samudra/v2026-07/om4_twodeg/`, the same train/validation
dates, `thermo_dynamic_all` prognostics, `tau_hfds` forcing, seed 15, plain
normalized MSE, one forecast target (`steps: [4]`), learning rate 0.0006
without a scheduler, and 12 epochs. The historical Perceiver uses microbatch 1
with 32-step accumulation. The two native-SDPA Perceivers use microbatch 2 with
16-step accumulation after preflight showed a roughly 77 GiB peak. The Direct
and U-Net controls retain microbatch 8 with four-step accumulation. The
effective batch is 32 on one GPU throughout.

The 90x180 grid cannot pass through the presets' full-depth U-Nets without an
odd-size mismatch. Every experimental model therefore uses one downsampling
stage. The Perceiver models use 3x5-cell patches (6x10 degrees), producing a
30x36 processor grid, and six-patch decoder windows. Samudra Direct retains one
latent per native 2-degree cell.

## Matrix

| Role | Experiment branch | Config | Primary contrast |
| --- | --- | --- | --- |
| Main Multi | `experiment/main-baselines-2deg` | `train_multi_main.yaml` | Historical full PerceiverIO control |
| Native SDPA PIO | `experiment/perceiver-2deg-hpc` | `train_pio.yaml` | Attention implementation/runtime only |
| Perceiver candidate | `experiment/perceiver-2deg-hpc` | `train_direct_query.yaml` | Remove the second latent decoder bank and widen output transport |
| Samudra Direct | `experiment/direct-2deg` | `train_direct.yaml` | Native-grid learned representation and deterministic decoder transport |
| Samudra U-Net | `experiment/main-baselines-2deg` | `train_samudra.yaml` | Established non-multi architecture reference |

All W&B runs use entity `ocean_emulators`, project `default`, and group
`torch-2deg-perceiver-20260812`.

## Interpretation

- Native SDPA PIO versus Main Multi tests numerical/optimization parity and
  runtime after removing the external Perceiver implementations.
- The candidate versus Native SDPA PIO isolates the decoder intervention.
- The candidate versus Samudra Direct tests whether a Perceiver bottleneck is
  already competitive with a native-grid representation at this scale.
- Samudra U-Net is a capability reference, not a parameter-matched causal
  ablation.

Primary quality is validation unweighted normalized MSE overall and by variable
group. Also record wall time, samples/second, peak GPU memory, predicted/target
spectral power where available, Slurm job ID, W&B run ID, resolved config, Git
commit, and checkpoint hashes.

## Preflight ledger

The first tracked smoke pass ran five train and five validation batches. It
established that 16 CPUs and 175 GiB of host memory were unnecessary: the
successful Direct run peaked below 8 GiB host memory. Subsequent jobs request
4 CPUs and 32 GiB. These smoke runs are diagnostic and must not be compared as
trained models.

| Role | Slurm | W&B ID | Result |
| --- | ---: | --- | --- |
| Main Multi | 15655375 | none | Failed before W&B: optional `flash_perceiver` unavailable; retry with built-in naive backend |
| Native SDPA PIO | 15655376 | `63ywsas9` | OOM above 95 GiB at microbatch 8; retry at microbatch 1 |
| Perceiver candidate | 15655377 | `i5fjyehv` | OOM above 95 GiB at microbatch 8; retry at microbatch 1 |
| Samudra Direct | 15655378 | `7uuxmov0` | Completed; peak GPU memory about 12 GiB |
| Samudra U-Net | 15655379 | `736ti5gx` | Pending completion at time of entry |

The reduced requests allowed the three retry jobs to start together on node
`gr102`. That exposed a launcher issue: each job inherited port 40000 for its
single-process distributed group. Jobs 15655627 and 15655629 collided; the
harness now sets `MASTER_ADDR` and derives a distinct `MASTER_PORT` from the
Slurm job ID. (Samudra requires both variables before using `env://`.) Job
15655628 retained the port and continued, so the collided jobs are retried
without treating this as a model result.

The corrected microbatch-1 smoke runs all completed:

| Role | Slurm | W&B ID | Peak GPU memory | Diagnostic validation MSE |
| --- | ---: | --- | ---: | ---: |
| Main Multi (naive backend) | 15655686 | `25xojbcg` | 95.1 GiB | 0.600 |
| Native SDPA PIO | 15655628 | `yh3jrgei` | 39.4 GiB | 0.600 |
| Perceiver candidate | 15655687 | `kiu1eg22` | 38.8 GiB | 0.645 |
| Samudra Direct | 15655378 | `7uuxmov0` | 12.1 GiB | 0.621 |
| Samudra U-Net | 15655379 | `736ti5gx` | 83.0 GiB first batch | 0.650 |

These losses are not model-quality results. Debug mode stops after five train
batches: accumulation-32 runs have made no optimizer update, while the
accumulation-4 controls have made one. They exist only to verify execution and
bound memory.

Native-SDPA microbatch-2 checks 15655828 (`hbs5xhtl`) and 15655829
(`vvhl7gts`) also completed, peaking at 77.5 and 77.0 GiB respectively. They
verify the selected batch shape but remain non-training diagnostics.

## Evaluation protocol

Use three fidelity levels for architecture iteration:

1. A five-batch smoke checks construction, forward/backward execution, memory,
   and throughput. It produces no model-quality ranking.
2. A one-epoch probe gives every configuration about 88 optimizer updates at
   effective batch 32. Rank these primarily by one-step normalized validation
   MSE and by 30/90-day autoregressive raw-field RMSE from the rollout-validation
   work in PR 770. Record per-variable/depth values, finite-rollout status,
   throughput, and peak memory; do not combine raw-unit variables into one
   arbitrary score. Extend only promising candidates to three epochs and a
   360-day rollout. Use one seed for screening and multiple seeds for finalists.
3. After the 12-epoch finalist runs finish, evaluate the same selected
   checkpoint from every architecture over the same 2015--2022 rollout and
   enable the durable observation metrics against the shared OM4 baseline.

The durable metrics test geostrophic velocity, instantaneous EKE, SST error and
residual variability, and 0--700 m / 700--2000 m heat content. They are
important because MSE alone can prefer an over-smoothed predictor with weak
variability. PR 770's horizon metric is the mean of per-step area-weighted RMSE
through each horizon, rather than endpoint-only RMSE; this makes it a useful
stability/trajectory score but should be labelled accordingly.

Do not run the durable metrics on the debug checkpoints. Besides their unequal
optimizer-update counts, the current one-year inference fixture does not cover
the complete 2015--2022 calendar-year scoring window required by the metrics.
At most, use one trained checkpoint for an early end-to-end eval-pipeline probe;
the final comparison should score every fully trained model with the identical
rollout configuration.

## Run ledger

| Role | Commit | Slurm | W&B | State | Best validation MSE |
| --- | --- | ---: | --- | --- | ---: |
| Main Multi | pending | pending | pending | Prepared | pending |
| Native SDPA PIO | pending | pending | pending | Prepared | pending |
| Perceiver candidate | pending | pending | pending | Prepared | pending |
| Samudra Direct | pending | pending | pending | Prepared | pending |
| Samudra U-Net | pending | pending | pending | Prepared | pending |
