<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Surface-initialized ocean prediction: wave 1

Approved September 18, 2026: approximately 576 GPU-hours, at most eight Torch
RTX6000 GPUs concurrently. Subsequent waves require another explicit go-ahead.
The branch `codex/surface-initialized-wave1` locally merges Rust loader branch
`u/jder/rust-loader` at `9cd36b1bdcf4921fb027494afa4157e217818ccd`.

## Scientific scope

This is an OM4-only feasibility wave while observational data are prepared.
It cannot establish observational forecasting skill, daily surface skill, or the
benefit of higher-resolution simulations. It trains at the existing five-day
cadence using OM4 tauuo/tauvo/hfds, and forecasts leads 5–30 days. ERA5 surface-state
forcing and daily SSH/SST outputs remain requirements for subsequent observation
experiments. Current model fluxes are an interim pretraining input.

Inputs use six surface frames (25 days between oldest and newest timestamps).
The initializer predicts two full states with T/S/u/v at all 19 available levels
plus SSH, copying the two known SST/SSH frames into its output. Its ConvNeXt U-Net
uses widths 128/192/256/384, surface masks, spherical coordinates, and season.
There is no hidden-interior input. Pretraining labels are OM4 interiors.

Both evolution models use the same initializer checkpoint and backbone widths:

- **AR:** two full states to one next five-day state, recursively. Pretraining
  increases from one to three to six steps; joint fine-tuning uses six steps.
- **Direct:** initial two full states, requested lead, and the ordered forcing
  frames through that lead predict its full state in one model call. A small
  per-cell temporal MLP encodes the ordered padded forcing sequence and validity
  mask. Training cycles through the six leads. It never sees later forcing.

Each evolution model first trains on true initial states for 70% of its training
cap, then jointly tunes its own initializer/evolution pair for 30%. Joint loss
includes 0.1 times interior reconstruction loss. All model T/S/u/v targets remain
supervised throughout this simulation-only wave. No surface observations are
injected after initialization. Forecast loss gives equal weight to variable
groups and equal weight to normalized levels within each group, with wet-cell
cos(latitude) area weighting on the regular one-degree grid.

## Data and splits

Torch root: `/scratch/jr7309/data/om4_onedeg_v3` containing `OM4.zarr`,
`OM4_means.zarr`, and `OM4_stds.zarr`. Existing mean/std files are used without
restricting their time range, as explicitly approved by the user. Training
examples remain temporally restricted:

- Train: 1975-01-03 through 2013-10-04.
- Validation/checkpoint selection: 2013-10-05 through 2014-10-05.
- Held-out report: origins/targets entirely within 2014-10-10–2022-12-24.

Every history and target window stays within its split. Held-out windows are
sampled every six stored frames; this is a collection of 30-day hindcasts, **not**
a continuous eight-year rollout. The latter belongs in a later stability wave.
Seasonal climatology is fitted only on training origins sampled every six frames
and grouped by calendar month. It is a cheap sampled climatology baseline.

## Execution and accounting

Use `samudra.experiments.surface_wave` through the existing Apptainer/torchrun
harness with an immutable code overlay and the matching Rust-enabled SIF.
Explicitly select account `torch_pr_347_lzanna`, partition `rtx6000_lzanna`.
Rust uses no PyTorch data-loader workers, two prefetched batches, CUDA prefetch,
and an explicitly capped native read pool per rank. Initial resource request:
two CPUs and 16 GiB host RAM per GPU. The qualified configuration uses eight
native I/O readers per rank (I/O concurrency, not eight allocated CPUs) and
batch size two. The two-GPU initializer smoke measured 3.1 samples/s and
about 4.5 GiB peak host memory per rank. These are short-run measurements,
not sustained four-GPU throughput claims.

Planned caps (scheduler wall time includes evaluation/setup):

| Job | GPUs | Wall-time cap | Maximum GPU-hours |
| --- | ---: | ---: | ---: |
| End-to-end smoke | 1 | 1 hour | 1 |
| Shared initializer and climatology | 2 | 8 hours | 16 |
| AR pretraining | 4 | 40 hours | 160 |
| AR joint tuning and metrics | 4 | 20 hours | 80 |
| Direct pretraining | 4 | 40 hours | 160 |
| Direct joint tuning and metrics | 4 | 20 hours | 80 |
| Reserved for checks/recovery | — | — | 79 |

Initializer training cap is 7 hours; each forecast's training cap is 54 hours
(37.8 pretraining, 16.2 joint). Validation time is charged to training caps;
final evaluation and initialization occupy the scheduler margin. AR/direct
start only after initializer success and run concurrently when capacity allows.
Pretraining and joint phases are separate dependent jobs, each below Torch's
48-hour short-job QOS boundary. The joint job resumes the completed pretraining
checkpoint and its own joint checkpoint if present.
The 576 GPU-hour authorization is a ceiling, not an instruction to consume it.
Record actual GPU-hours from Slurm, including failed attempts, before recovery.

Phase checkpoints include optimizer, epoch, within-epoch cursor, cumulative
phase time, and best validation score. Sampling resumes deterministically.
There is no dropout; batch-normalization buffers are synchronized before scoring.
Checkpoints are atomic, every 20 minutes and at phase completion. No automatic
requeue or unrestricted retry is configured. Restarting consumes the recovery
allowance and requires checking actual accounting first.

## Outputs and interpretation

Each run writes a provenance manifest, progress JSONL, W&B run, last/best phase
checkpoints, completion marker, and held-out CSV. The CSV distinguishes inferred
versus true initialization under the same final model, inferred-state
persistence, and seasonal climatology. It reports normalized and physical RMSE
by variable/depth, lead, and global/tropical/extratropical region. Physical second
moments provide a first velocity-amplitude diagnostic; they do not establish
balanced or realistic circulation. Pretraining validation also records the
true-versus-inferred initialization gap before joint tuning.

The initial report must distinguish executed work from pending jobs and must
not interpret missing/failed results as zero skill. Recommend the next wave
from initializer quality, its downstream forecast gap, AR/direct skill, velocity
amplitude, throughput, and memory; do not submit that wave without approval.


The bounded CPU monitor writes `monitor.json`, `monitor.jsonl`, and a
`needs_attention.md` file if a job fails, progress stalls, or observed concurrency
exceeds eight GPUs. It checks every ten minutes and stops after all GPU jobs are
terminal (or after five days). It does not silently retry failures or submit a
new wave. The dependent CPU report writes `report.md` and `report.json` after both
forecast branches terminate. Automated monitoring/reporting is distinct from
agent diagnosis and fixes; any failure recorded by the monitor still needs review.
