<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global/patch transfer: accessory supervision and processor capacity

**Production started October 4 at 1:49 a.m. ET.** Job 209608 is running on
`b2-14-s1-dgx-02-c04`, after successful completion of ablations 209574.
At 1:52 a.m., all four first-stage processes (U-aux01, U-aux10, W-global,
W-multitask) were warming the 2,829-frame global cache. Axial arms follow their
paired accessory arms on the same GPUs. No production result is claimed yet.
By **1:56 a.m. ET**, all four had finite optimizer updates: U-aux01/U-aux10 at
four updates, W-global at seven and W-multitask at five. Accessory losses were
logged separately from physical objectives, as qualified.

Follow-up within the user's authorized iteration window ending Monday October 5
around 9 a.m. ET. This follows the [original screen](extent-wave-2026-10-02.md)
and complements the running [mechanism ablations](extent-ablations-2026-10-03.md).
The original native-patch task hurt observation scores; these experiments test
whether fine-scale targets help without a regional task switch, and whether
processor capacity or attention improves transfer. They do not use LLC.

## Predeclared runs and methods

All arms start fresh with seed 1729, use the original training splits, masks,
channel normalization, spatial InstanceNorm, geometry inputs and task adapters.
The common 31.28M-parameter U-Net initializer consumes 19 surface/forcing frames
and predicts two full physical states. The processor evolves 77 physical fields
through six nominal five-day steps. Global fields are 180×360; regional examples
are 128×128 native quarter-degree crops, with a 64×64 scored interior and no true
future boundary conditions. Native data remain the verified v2026-09 five-day
averages, and global data remain the original 1° OM4 v3 store.

| Literal name | Global OM4 updates | Native patch updates | Observation updates | Change from the corresponding original U arm | Total parameters |
|---|---:|---:|---:|---|---:|
| U-aux01 | 2,000 | 0 | 2,000 | Global processor, accessory weight 0.01 | 62,967,809 |
| U-aux10 | 2,000 | 0 | 2,000 | Global processor, accessory weight 0.1 | 62,967,809 |
| W-global | 2,000 | 0 | 2,000 | Wider processor, global-only control | 100,659,264 |
| W-multitask | 1,000 | 1,000 | 2,000 | Same wider processor, mixed global/patch tasks | 100,659,264 |
| A-global | 2,000 | 0 | 2,000 | Original widths plus bottleneck axial attention, global-only control | 64,151,938 |
| A-multitask | 1,000 | 1,000 | 2,000 | Same attention processor, mixed global/patch tasks | 64,151,938 |

Original **U-global** and **U-multitask** have 62,967,680 parameters. U-global
has the same update allocation as the accessory arms; U-multitask has the same
allocation as W-multitask and A-multitask. Those names and methods are also
documented in the original report. No fine-scale input is supplied to either
accessory arm. The extra head has only 129 parameters and is unused at inference.
Adding it preserves the original U-global initialization, including subsequent
ERA5 adapter initialization; a unit test compares all shared weights exactly.

**Accessory target.** At each future OM4 lead, a 1×1 linear head on the processor's
final full-resolution features predicts standardized `log1p(q / scale)`, where
`q = 0.5 * (mean(u² + v²) - mean(u)² - mean(v)²)` over native surface velocities
inside that coarse cell. Actual stored latitude/longitude cell bounds determine
spherical area overlaps. Quarter-degree latitude edges are not exactly nested
in groups of four within the 1° grid. Use only common wet native U/V support,
require at least 50% native wet coverage and the global surface wet mask, and
average target loss over valid area. The loss averages all six predicted leads
and is added to the unchanged physical forecast/reconstruction/completion loss.
The two weights bracket a modest versus stronger representation constraint;
there is no adaptive weight selection during a run.

This target is **spatial subcell velocity variance in five-day-mean fields**.
It is neither all unresolved kinetic energy nor the SSH-derived geostrophic EKE
used in observation evaluation. It supervises near-surface structure only; any
deeper or long-term benefit must be demonstrated. Global and native OM4 stores
also come from different preprocessing releases, as in the first screen.

**Wider processor.** Widths increase from `[128, 192, 256, 384]` to
`[192, 288, 384, 576]`; the initializer and other training choices remain fixed.
This is an update-matched capacity comparison, not a FLOP-matched comparison.

**Axial processor.** Keep the original widths and insert one row-attention then
column-attention residual block after the U-Net bottleneck convolution. Each
axis uses eight heads over 384 channels, pre-attention LayerNorm on features,
zero dropout, and a learned residual scale initialized to 0.1. There is no
fixed-size positional table; existing geographic/task inputs still apply.
LayerNorm here normalizes hidden features, not physical inputs. Attention mixes
only the supplied extent; global periodic convolution and regional constant
padding retain their existing semantics. This adds 1.184M parameters and changes
compute, so any gain cannot be attributed to mixing alone without further work.

## Fixed selection and execution gates

AdamW, LR 1e-4, weight decay 0.01, gradient clipping at 1, accumulation over eight
samples, no warmup, the mixed schedule and 4,000 total updates remain unchanged.
Select checkpoints using the frozen global observation integrated-plus-spectral
validation composite. RMSE and physical training losses are diagnostics only.
Report the same 96 monthly 30-day test forecasts and each model's initialized
persistence, with common test-climatology normalization as in the original
report. These are repeated exploratory uses of the held-out cohort, not a newly
untouched test set. Do not choose subsequent runs from test rankings.

Each architecture must pass fresh fitting and real joint-training/resume probes
under the new source pin. Both members of every pair must qualify separately.
Production requires successful probes and measured throughput consistent with
the review window. A failing probe is preserved and investigated before any
replacement. No checkpoint from the earlier producer is resumed into these arms.

Proposed four-GPU packing: GPU 0 runs U-aux01 then A-global; GPU 1 runs U-aux10
then A-multitask; GPUs 2 and 3 run W-global and W-multitask. Production starts
after the current ablation allocation succeeds. Queued work is allowed to wait;
do not cancel and resubmit to chase capacity. The Monday report will state any
unfinished budgets explicitly.

## Target preparation and verification

Grace job **116869** completed successfully in 32 seconds with 16 CPUs, 32 GiB
and no GPUs. Producer `6f678cbfccf7d630780c97111d95aff4aefa0d4a`.
All **2,829** training frames (1975-01-03 through the last included timestamp,
2013-09-30; training cutoff 2013-10-04) were read back
exactly, and all contributing wet native velocities were finite. There are
43,252 valid coarse cells. Training-only transform values are scale
0.0027922708904027874 m²/s², log mean 0.43588650490263964 and log std
0.5750147760256608. No validation/test frames enter this normalization.

Target root:
`/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg_surface_variance_1deg`.

| File | SHA-256 |
|---|---|
| manifest.json | `62161cf57f7c924b4e769960d6a5f0491bdf53a5cd07deaa0ef8b2ec45b5ea72` |
| grid.npz | `a2ca5850c268c4bd13628eee7e7766b379241af4e129170034b658864f14d7e6` |
| normalization.json | `9c12f699735797f9b2775a68b0efd592e32d98f734b83362841b8c61b8e86084` |

Tests cover non-nested conservative aggregation, mean-flow-invariant variance,
wet nonfinite failures, exact future-date target alignment, masked target loss,
accessory gradient routing, preserved core initialization and dynamic-extent
axial gradient propagation. Production qualification/results will be appended.

### How much of the accessory target is static?

A training-only decomposition over every valid target frame gives:

| Fixed-map predictor | Area-weighted standardized target MSE | Fraction explained relative to zero |
|---|---:|---:|
| Zero in standardized target space | 1.0000 | 0% |
| Each cell's training time mean | 0.3220 | 67.8% |
| Each cell's training calendar-month mean | 0.2799 | 72.0% |

Thus a substantial fraction can be learned from geography alone, and about 28%
remains after fitting geography plus monthly seasonality. These are in-sample
fixed-map fits to training targets, not held-out predictive skill. An accessory
loss can help through geographic/seasonal representation without teaching
fine-scale dynamics; improvement in its loss alone cannot distinguish those
mechanisms. If the accessory arms improve observation validation, a useful next
control is supervision from the fixed training climatology instead of the
time-varying target. That control is not folded into the current six arms.

The diagnostic independently reproduced unit standardized target variance.
Grace job **116904** completed in 16 seconds using four CPUs and 8 GiB, with
no GPU or model optimization. Diagnostic producer
`c1bc69796b1738b9fa36495240e743cfe843ea43`.
[Raw decomposition and hashes](artifacts/extent-representation-2026-10-03/target-decomposition.json).

## Submission snapshot, October 3 at 7:02 p.m. ET

Training producer: `b14ce8acd0ddf77b42b6341b25cdfe242dc8eec6`.
The source-only archive hashes to
`cf02c2a6a611388227bdb15934de3dafff5085c3fe633f67f9b63c636c84c113`.
Local validation passed 24 targeted tests, Ruff, mypy for ten touched Python
files, shell syntax, and actual wider/axial forward/backward checks.

| Family | Fitting job | First joint/resume probe | Paired joint/resume probe |
|---|---:|---:|---:|
| Accessory | 209580 | U-aux01: 209581 | U-aux10: 209582 |
| Wider | 209583 | W-global: 209584 | W-multitask: 209585 |
| Axial | 209586 | A-global: 209587 | A-multitask: 209588 |

Each second probe depends on its first, and each first probe depends on fitting,
limiting this qualification phase to three concurrent GPUs. At this snapshot,
all fitting jobs were **pending** for capacity/priority; all probes were pending
on dependencies. No production job has been submitted for this new wave yet.
The four-arm mechanism-ablation allocation **209574 remains running**.

New root: `/projects/ny/lz1955/multiscale/jrusak/runs/2026-10-03-extent-representation`.
[Exact submission receipts](artifacts/extent-representation-2026-10-03/qualification-jobs.json)
include resource requests and dependency IDs. All qualifications request one GPU,
16 CPUs, 128 GiB, and at most two hours on `beta_test` with QoS `test`.

## Qualifications passed; production queued at 8:32 p.m. ET

All three fitting jobs and all six joint probes completed successfully, with
exit 0. Initial-to-final fitting losses were 1.703→0.282 (accessory),
1.669→0.287 (wider), and 1.704→0.283 (axial). Required initializer, processor,
adapter and surface-completion gradients were present. Joint probes additionally
verified accessory-head gradients on OM4 and their absence on observation tasks.
The reported physical objective excludes the separately logged accessory term.

All six probes verified exact serialization/restoration and serialized replay
differences within the measured native nondeterminism tolerance. Measured peak
GPU allocation was 71.7–71.8 GiB, including the shared global-data cache.

| Arm | Probe seconds/update | Extrapolated training hours for 4,000 updates |
|---|---:|---:|
| U-aux01 | 8.23 | 9.15 |
| U-aux10 | 7.71 | 8.57 |
| W-global | 8.03 | 8.92 |
| W-multitask | 10.83 | 12.03 |
| A-global | 8.10 | 9.00 |
| A-multitask | 11.09 | 12.32 |

These short-probe estimates exclude validation/startup; the first screen's
probes were slower than its eventual production average. The longest proposed
serial GPU lane totals 20.89 estimated training hours, leaving about three hours
within a 24-hour allocation for overhead. This is a feasibility check, not a
guaranteed completion time. Qualifications consumed **1.271 allocated GPU-hours**.

Production **209608** is queued on `beta`, QoS `standard`, one node with four
GPUs, 144 CPUs, `--mem=0`, and a 24-hour cap. It depends on successful completion
of current ablation job **209574**, and uses the fixed packing documented above.
The scheduler's dry-run start estimate was **October 4 at 3:17 a.m. ET**; actual
start remains unconfirmed. No queue-chasing resource change was made.

The first submission dry-run rejected completed probe IDs because they were no
longer available to Slurm's live dependency mechanism. It launched no job and
used no GPU time. Admission instead explicitly checked all six `COMPLETED/0:0`
accounting records and their producer-matching qualification contracts before
submission, while retaining the live `afterok:209574` dependency. The training
runner also checks each exact qualification contract before production. The
failed submission script is preserved; no scientific or resume contract changed.

Evidence: [qualification results](artifacts/extent-representation-2026-10-03/qualification-results.json),
[production admission](artifacts/extent-representation-2026-10-03/production-admission.json),
[exact production receipt](artifacts/extent-representation-2026-10-03/production-jobs.json).
