<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Initializer capacity and historical inputs: wave 3

Authorized September 21, 2026. Deliver results, reproducible code and linked assets
in draft PR #885 by **September 24, 2026, 17:00 America/New_York (21:00 UTC)**.
Training stops by 13:00 Eastern Thursday to protect evaluation/report time.

## Questions and arms

Does initializer capacity, historical input information, or attention improve
reconstruction and downstream forecasts? Longer surface and forcing histories
change together, as requested; this wave does not isolate their separate effects.

| Arm | Architecture | Inputs |
| --- | --- | --- |
| A | Current ConvNeXt U-Net, 30.2M parameters | Six SSH/SST frames, 25 days; no forcing history |
| B | Current ConvNeXt U-Net | 19 SSH/SST and OM4 forcing frames, 90 days |
| C | Wider ConvNeXt U-Net, 120.6M parameters | Same inputs as A |
| D | Wider ConvNeXt U-Net | Same inputs as B |
| E | Swin-style reconstruction network, 122.1M parameters | Same inputs as A |
| F | Swin-style reconstruction network | Same inputs as B |

Wider U-Net widths are 256/384/512/768, retaining the baseline's blocks and
four scales. The Swin-style variant uses a stride-two patch embedding, widths
192/384/768/1024, stage depths 2/2/12/2, six-cell shifted attention windows,
global attention at the coarsest scale, and a multiscale dense decoder with a
full-resolution input skip and nonlinear refinement. That skip gives the model
a path for reconstructing detail finer than its two-cell patch embedding. Longitude
is periodic in window attention; local attention cannot wrap across the poles.
It is a study-specific Swin variant, not a pretrained image classifier or a claim
to reproduce an ocean architecture from a paper. Exact parameter counts go in
run manifests; expanded-input models have slightly larger first layers.

All arms receive surface masks, geographic features and annual sine/cosine.
They predict two full 77-channel states (T/S/u/v at 19 depths and SSH), copying
known surface values exactly. Historical forcing is normalized OM4 tauuo,
tauvo and hfds through initialization; it contains no future surface observations.
Existing means/stds are reused. No climatological-residual reformulation in this
wave: that would change another factor.

## Data, dates and supervision

Use `/scratch/jr7309/data/om4_onedeg_v3`. Training data remain January 3, 1975
through October 4, 2013. Every arm uses the same eligible 19-frame training
windows; short arms consume only the final six surface frames. This reduces the
training-origin count slightly relative to wave 1, equally in all six arms.

Keep the original approximately monthly validation and [99 held-out forecast](surface-wave3-origins.json)
origins, and six five-day forecast leads. To supply earlier history without
moving these origins, validation/test context stores begin 65 days earlier:
August 1, 2013 and August 6, 2014 respectively. Those extra frames are read-only
historical context, not additional validation targets or training labels. Verify
actual origin timestamps against wave 2 before accepting production results.
The actual training source retains its original end date independently of the
read-only context bundle.

Reconstruction training retains the existing equal-variable normalized interior
T/S/U/V loss. Select checkpoints by equal subsurface-T/S normalized reconstruction
MSE averaged over the two reconstructed states and fixed validation origins.
Evaluate fixed training probes using the same evaluation mode and metric to
measure the train/validation gap. The full-variable loss remains a diagnostic.
All new initializers start from scratch; references retain their original weights.

## Qualification, training and follow-ups

1. Qualify all six arms on real data, including native-versus-cached equality,
   backpropagation, finite validation, checkpoint writing and a reduced evaluation.
2. Run equal short learning-rate pilots at 1e-4 and 3e-4 for each arm. Select one
   rate per architecture using the average relative validation improvement across
   its two input conditions. Use validation only. Primary runs restart from the
   same seed; pilot checkpoints are not warm starts.
3. Give each primary run up to eight training hours on two GPUs, global batch
   eight with gradient accumulation, AdamW, weight decay 0.01, BF16 and gradient
   clipping at one. Validate every 20 minutes and checkpoint at most every three
   minutes. Early-stop after six non-improving validations and at least two hours.
   Report actual updates, samples, elapsed time, GPU model and GPU-hours. Where
   hardware differs, do not call wall-time equality FLOP equality.
4. Replicate A and up to two validation-selected alternatives from independent
   initial weights/data order. Select at most one improved-input CNN and one
   attention model when both remain competitive; final choices and rationale are
   recorded before held-out evaluation. Before these choices, qualify joint
   adaptation from each completed primary checkpoint using the full validation
   set. Its baseline event scores the unmodified checkpoint through fixed
   pretrained dynamics, before any adaptation update. Use this forecast validation
   to choose promising follow-ups, while retaining reconstruction validation for
   primary checkpoint selection. The subsequent three qualification updates are
   plumbing only; they are not warm starts for adaptation. Extend promising
   unconverged training only when validation curves and the report deadline justify it.
5. Run matched short joint adaptation of A and selected alternatives, initialized
   with each chosen initializer and the same wave-1 pretrained evolution model.
   Use six-step full-variable forecast loss plus 0.1 reconstruction loss,
   learning rate 1e-4 and validation T/S forecast selection. Starting checkpoints
   are eligible; keep the initial fixed-dynamics results separately.

Primary reconstruction rankings use reconstruction validation. Joint-adaptation
rankings use forecast validation. Do not select separately for each held-out lead
or region. All held-out evaluation happens after the relevant selections are
locked, except reduced plumbing qualifications, whose metrics do not rank models.

## Evaluation and report

- Matched reconstruction training/validation errors; held-out lead-zero errors.
- Five- through 30-day forecasts through unchanged pretrained dynamics for all
  six primary initializers, followed by separately labeled adapted dynamics.
- True-interior evolution, inferred-state and true-state persistence, and existing
  initializer/E references, compared at exact matching origins and targets.
- T/S by depth and region; SSH/SST and U/V diagnostics; fixed-reference anomaly
  amplitude, pattern correlation, mean bias and representative spatial maps.
- Per-origin/channel dumps, paired calendar-year uncertainty summaries and
  independent-initializer-seed comparisons; no claim that temporal bootstrap
  measures training-seed uncertainty.
- Actual job/attempt accounting, checkpoint and producer provenance, learning
  curves, failures and remaining limitations. These remain OM4 five-day feasibility
  results, not observational, ERA5, daily-output or long-continuous-rollout skill.

## Execution and monitoring

Prefer qualified H200 preemption capacity when useful; RTX6000 remains a fallback.
The user authorized helpful compute within Slurm limits. The initial planning
envelope was 400 GPU-hours and eight concurrent GPUs; the updated goal removed
those fixed caps. Track consumption and use the deadline and scientific comparisons
to bound follow-ups, rather than filling available capacity without purpose.

Use immutable commit-named code layers and the existing Rust-loader SIF. Keep
scratch-backed logs, checkpoint/cache paths, W&B online and per-job utilization
telemetry. Native readers remain eight per rank with low CPU/host-memory requests.
H200 requests use the documented Slurm preemption comment, requeue and USR1
warning; resume atomic checkpoints with optimizer, epoch and sample cursor.
Checkpoint recovery must not silently change architecture, input definition,
world size, batch or data order.

Observe bring-up until real steps appear, then inspect approximately hourly.
Recover safe failures, reassess prolonged low utilization, and report meaningful
changes or blockers. No extra wave outside this approved protocol is submitted.
The saved goal remains active through report publication.

## Initial execution record

All six reconstruction configurations passed real-data qualification on two
RTX PRO 6000 Blackwell GPUs. Twelve 30-minute learning-rate pilots then completed
on that same hardware, with the same seed, global batch, data and validation
protocol. The predefined paired-input selection rule chose **3e-4 for all three
architectures**. The [frozen decision](surface-wave3-pilot-selection.json) contains
all pilot scores and job IDs; it was recorded before primary submission.

Six primary jobs were submitted September 21, 2026 (Eastern), starting from
fresh seed-1729 weights: A 18221169, B 18221171, C 18221172, D 18221173,
E 18221174 and F 18221175. They retain the eight-hour maximum and validation
stopping rule above. This is an execution record, not the final results report.

H200 execution also passed a reconstruction check and baseline/largest-model
joint-adaptation checks. Final comparisons will use matched RTX hardware:
across four qualification origins, H200 versus RTX fixed-dynamics T/S RMSE
changed by at most 0.015%, while some individual velocity errors changed more.
The unchanged persistence control matched exactly; cross-hardware bitwise or
near-bitwise evolution equality is not assumed. H200s remain available for
training and auxiliary work with hardware recorded explicitly.

### CNN follow-up decision (September 22, 07:26 UTC)

A–D completed their prescribed validation stopping rule. Scoring each selected
initializer through the unchanged pretrained dynamics on all 11 validation
origins, before any joint update, gave:

| Arm | Forecast validation T/S normalized RMSE |
| --- | ---: |
| A | 0.06666 |
| B | 0.06486 |
| C | 0.05998 |
| D | 0.05903 |

D is the selected CNN alternative: its validation forecast RMSE is 11.45% lower
than A. This decision uses no production held-out scores. Baseline and D fresh
seed-2718 replicas retain the primary eight-hour maximum (jobs 18238731 and
18241142). Their matched joint adaptations use a three-hour training maximum,
learning rate 1e-4 and the original primary best checkpoints (jobs 18240594 and
18241143). Qualification-updated checkpoints are not used as warm starts.

The short-history Swin arm E stopped after six non-improving checks. Its best
checkpoint remains intact, but both training-probe and validation error rose
sharply later in training. Thus optimization instability is a confounder for
architecture comparisons. The attention follow-up decision remains pending F's
completion and the full forecast-validation checks; any stability follow-up will
be labeled separately from the predefined six primary runs.
