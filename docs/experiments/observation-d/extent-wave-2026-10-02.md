<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global and native-quarter-degree patch transfer screen

Approved October 2, 2026. One seed (1729), physical state first, no LLC,
no additional latent channels and no diffusion. Branch `codex/patch-global-om4`
starts from observation-report commit `4f1aaa8281b9c49f3e388f8734ab9e2ec067ab32`.
This screen asks whether replacing some coarse global model-data updates with
fine regional updates improves **global observational** forecasting.

| Literal run name | Processor | Global OM4 updates | Native quarter-degree patch updates | Global observation updates |
|---|---|---:|---:|---:|
| U-global | Geometry-conditioned D-style U-Net | 2,000 | 0 | 2,000 |
| L-global | Geometry-conditioned bounded local processor | 2,000 | 0 | 2,000 |
| L-multitask | Same local processor, shared across all tasks | 1,000 | 1,000 | 2,000 |
| U-multitask | Same U-Net, shared across the whole supplied global or regional field | 1,000 | 1,000 | 2,000 |

The primary comparisons are U-multitask versus U-global and L-multitask versus
L-global. The two processor families are **not parameter matched**. All arms
start from random weights. Within each family, global and multitask arms have
identical initialization. The existing report is an external reference; this
2k+2k screen is not an endpoint-budget reproduction of its 8k+8k model.

## Fields and tasks

Global OM4 uses the existing, unfiltered five-day-mean `om4_onedeg_v3` store at
`/mnt/home/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3`: 180×360.
Regional OM4 uses the **native values**, without resizing or averaging, from
`s3://m2lines-pubs/Samudra/v2026-09/om4_quarterdeg/OM4.zarr`: 720×1440.
Its verified transfer destination is
`/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_quarterdeg`.
These are different preprocessing releases of OM4; the screen does not isolate
release effects from the resolution/extent intervention.

Each regional example supplies a 128×128 native-resolution crop (approximately
32°×32°), predicts six five-day steps, and scores the central 64×64 cells
(approximately 16°×16°). The 32-cell halo is supplied context. Regional edges
use zero padding, not periodic wrap. Crops may cross the geographic dateline,
but that does not connect opposite edges of the regional computational domain.
There is no future ocean-state boundary forcing or teacher forcing.

Origins are deterministic functions of the sample seed; spatial origins are
sampled across the entire native source. Reject cores with under 20% surface wet
support. Latitude crops stay in bounds; longitude extraction wraps. All 25
frames of each training trajectory lie on or before 2013-10-04, matching the
global OM4 training cutoff. No observation test data enter these tasks.

All models retain 77 physical channels: T/S/U/V at 19 depths and SSH. The
initializer consumes 19 historical five-day SST/SSH and forcing frames (95 days)
and their validity, plus spherical geography and season. It produces two full
states. The processor consumes those two states, current forcing and geometry,
and predicts the next full state. Observed surface values overwrite initializer
outputs where valid; artificially hidden values supply a completion loss.

The OM4 tasks use native `tauuo`, `tauvo`, `hfds`; observations use the existing
8→32→3 ERA5 adapter. State scaling uses the same observation-training physical
channel constants on both resolutions. Forcing uses the existing global OM4
constants on both resolutions. There is no per-cell physical normalization.
Regional completion uses synthetic hiding over native wet support; global OM4
also emulates the existing observation coverage. This distinction is recorded,
not treated as a controlled missingness experiment.

The existing observation samples are unchanged at
`/mnt/home/jrusak/data/obs_full_range/d-observation-pilot/samples`: 180×360,
243 training months, nine validation months, 96 test months. They retain the
existing 19-frame histories, six/seven forecast bins, monthly interior targets,
strict five-day surface support and calendar-month integration.

## Architecture

The common initializer is the current report's small, affine-InstanceNorm
ConvNeXt U-Net (widths 128/192/256/384), not original D's larger initializer.
Global calls use periodic longitude and zero latitude padding; patch calls use
zero padding in both directions. Its weights and OM4 input adapter are shared
between coarse global and native fine regional calls.

The U processor uses the corresponding D-style U-Net and affine InstanceNorm.
Its operations and spatial normalization can mix over the entire supplied
field. This is the intended full multitask hypothesis, not an assertion that
U-Net crop and global outputs are equal.

The L processor has a 512-channel pointwise input projection, four residual
blocks, and a pointwise output. Each block has one depthwise 3×3 convolution,
512→1024→512 pointwise mixing, GELU, and learned residual scale initialized to
0.1. There is no spatial pooling or input-dependent normalization. Single-step
radius is four cells; six-step radius is 24 cells. Crop equivalence therefore
holds in the scored core **conditional on equal supplied states and geometry**.
It does not hold end to end through the common nonlocal initializer.

Both processors receive the same geometry interface: supplied spherical xyz
coordinates and season, local zonal/meridional spacing, supplied field extent,
regional flag, wet surface and fraction of wet physical slots. Spacing is
computed from coordinates, never inferred as 360 divided by tensor width.
A zero-initialized learned pointwise projection injects these additional
geometry planes. Source-specific OM4/observation input adapters are retained;
native patches share the OM4 adapter.

| Family | Initializer parameters | Processor including adapters/geometry | ERA5 adapter | Total |
|---|---:|---:|---:|---:|
| U | 31,281,670 | 31,685,623 | 387 | 62,967,680 |
| L | 31,281,670 | 4,399,879 | 387 | 35,681,936 |

## Training and evaluation

Use the existing one-optimizer mixed schedule, increasing observation frequency
through training. Multitask arms alternate global/patch on successive OM4
updates. AdamW, learning rate 1e-4 for both sources, weight decay 0.01, gradient
clip 1, effective batch eight by accumulation, no warmup. OM4 optimizes the
existing variable-balanced full-state six-step forecast loss plus 0.1 initial
interior reconstruction and 0.1 surface completion, scoring only the regional
core on patch tasks. Observation losses remain unchanged.

Selection uses the exact full-global reference from the current report:
0.5 × mean normalized integrated errors + 0.5 × mean 27 spectral errors in dex.
Validation runs every 100 total updates and at observation milestones
500/1000/1500/2000. Report held-out test metrics and inferred-state persistence
using the existing evaluator only after training and validation selection finish.
Training MSE is diagnostic, not the selection criterion. Physical-size or
supervised-cell equivalence is not claimed from equal update counts.

## Qualification and execution

Reuse the pinned ARM64 PhysicsNeMo image
`sha256:d7cf627257df3e1fda74202435e00f55c15ebe2d2be9648f47daf31905f51bad`.
Fit each architecture for ten updates on one training observation example.
Each arm then runs the existing six-OM4/four-observation joint probe, including
gradient routing, strict state serialization/restoration and measured native
versus serialized next-update replay. Multitask probes exercise native patches.
Production requires matching producer/data/model qualification contracts.

Before training, rehash all 350 observation NPZ files. The native patch cache
packs float32 fields into 128×128 spatial chunks and keeps separate SST/SSH
history arrays; it changes storage layout only. Every physical frame is read
back and compared exactly (including land NaNs); wet values must be finite.
Cache creation waits for the full quarter-degree transfer/readback job 114235.

Beta_test qualifications use one GPU each. Production uses one full beta node
with four independent one-GPU arms, with Slurm afterok dependencies on all four
arm probes. Fit and patch probes stay held until their data are verified. A
bounded CPU-light readiness bridge on the beta login host releases only those
recorded held jobs; it uses no GPU allocation and creates no conversation timer.
Ordinary long beta queue waits are expected; do not cancel/resubmit to chase
capacity. Production has a 24-hour allocation and a 20-hour training cap per
arm. Probe timing must predict at most 18 hours for the 4,000 training updates;
otherwise production fails before training rather than silently changing budgets.

Entry points: `scripts/run_extent_wave.py`, `scripts/slurm_extent_beta.sbatch`,
`scripts/prepare_om4_patch_cache.py`, and
`src/samudra/experiments/{extent_models,extent_data,extent_training}.py`.
The launch root records frozen paths, producer hash, selection reference,
scheduler IDs, readiness releases and logs. Production completion requires
actual `TRAIN_COMPLETE.json` and held-out evaluation `COMPLETE.json` outputs.

## Submission snapshot, October 2 at approximately 10:45 a.m. ET

Pinned producer: `ec9ecf916ebbff21306f8c3aa94553c8165f3a23`.
Source archive SHA256:
`29b28bb5aa05e22c051a6ca60a9fcb8a7fd1f258c8673f405b69a8f9be5a21fc`.
Exact existing full-global selection-reference SHA256:
`c94c0601e168858e4ba41423aed74d8604eaa1efbc68dd76370b32b8eeacd26c`.
Launch root:
`/projects/ny/lz1955/multiscale/jrusak/runs/2026-10-02-extent-om4`.
The complete submission commands and dependency IDs are in `jobs.json` there.

| Scheduler | Job | Purpose | Observed status |
|---|---:|---|---|
| Alpha/Grace | 114309 | Rehash all observation inputs | Verification output confirms all 350 files match; finished |
| Alpha/Grace | 114235 | Quarter-degree transfer and full readback | Running |
| Alpha/Grace | 114311 | Lossless native patch cache and readback | Pending transfer success |
| Beta | 209206 | U architecture fitting | Released after observation verification; pending resources |
| Beta | 209208 | L architecture fitting | Released after observation verification; pending priority |
| Beta | 209210 | U-global resume/gradient/throughput probe | Pending successful U fitting |
| Beta | 209212 | L-global resume/gradient/throughput probe | Pending successful L fitting |
| Beta | 209214 | L-multitask probe | Held for verified cache; also requires successful L fitting |
| Beta | 209216 | U-multitask probe | Held for verified cache; also requires successful U fitting |
| Beta | 209218 | Four-arm production and held-out evaluation | Pending successful completion of all four probes |

The readiness bridge on beta is PID 4007051; its receipt is
`gate-process.json`, its releases are in `released.json`, and its log is
`logs/readiness-gate.log`. It has released only 209206 and 209208 so far.
No GPU training or scientific result is claimed in this snapshot.

Local validation passed: 23 tests covering extent geometry, six-step crop
equivalence conditional on supplied states, boundary restoration, checkpointed
geometry gradients, native sample timing/repeatability, exact task counts,
existing joint resume/gradient routing, global observation support, completion,
latent-state compatibility, and task scheduling. Mypy passed for all eight new
Python files/tests and the five modified Python modules. Ruff and shell syntax
checks passed.

## Monitoring recovery, October 2 late morning

The user requested an in-conversation sleep loop, with longer sleeps during
ordinary queue waits. No conversation timer was created.

The first two fitting jobs completed successfully. Global joint probes 209210
and 209212 failed before training because the container lacked the optional
Rust loader. Their logs and the successful fitting outputs remain in the original
root. The unstarted patch probes 209214/209216 and blocked production 209218 were
cancelled for this concrete runtime repair, not to chase cluster availability.

Training recovery root: `<launch-root>/runtime-v2`.
Training producer: `ba22420144ef7576c324c40f74bae614ab77c1d4`.
The loader was built from the pinned Rust source, passed 90 loader/extent/joint
CPU tests (four CUDA/manual tests deselected), and constructed successfully on
beta. The wheel SHA256 is
`13b05ce0e69b9e1e54fda416e0d34cb6886108f2713749018fea58ce4a592c6b`;
the extension binary SHA256 is
`b6c47dc2ea183b482b9d1faf3d704ad5068bfee975f133f930b276936a75969c`.
Launches check the binary hash and actually construct the loader before starting
a fitting or training process. Qualification and resume contracts bind the
runtime package hash as well as the code producer. Fresh fitting was rerun;
old certificates were not relabeled.

| Replacement beta job | Purpose | Observed status |
|---:|---|---|
| 209227 | U fitting | Passed; training-example loss 1.70294 → 0.28793 |
| 209229 | L fitting | Passed; training-example loss 1.55556 → 0.95169 |
| 209231 | U-global joint probe | Running; native OM4 cache warmup reached |
| 209233 | L-global joint probe | Running; native OM4 cache warmup reached |
| 209235 | L-multitask joint probe | Held until the native cache is verified |
| 209237 | U-multitask joint probe | Held until the native cache is verified |
| 209239 | Four-arm production/evaluation | Pending all four replacement probes |

Both fresh fitting checks reached the initializer, evolution, ERA5 adapter and
completion head. These one-example fitting numbers are not scientific skill
results. The original login-host readiness bridge is superseded; the authorized
monitoring loop will release only the replacement patch jobs after checking the
cache marker and hashes.

Transfer job 114235 completed at **11:30 a.m. ET** with all 385,269 files /
1,594,021,374,932 bytes verified. CPU cache attempt 114311 failed immediately
because its environment lacked cftime. An isolated cftime 1.6.4.post1 ARM runtime
was staged without modifying shared environments. Attempt 114399 then exposed
an overly strict exact-five-day assertion in the new cache builder.

Live metadata checks established that quarter-degree and global OM4 timestamps
are **identical**, including Julian calendar/units and 16 six-day gaps across the
full 1958–2022 series. The existing global task already uses nominal five-day
stepping over this series. Cache producer
`909a6d60cebcc24f7a4dd86ce4b8cde2ee5e9c9e` now requires exact timestamp agreement
with that baseline and records the five/six-day gap counts. It does not resample
or change the training timestep convention. A regression test rejects shifted
timestamps, differing calendars and unexpected gaps. All 19 quarter-degree depth
coordinates also match the shared state depths exactly, and the builder checks
that match. The cache-only correction does not alter training producer ba2242014.

Grace cache job **114412** is running with the corrected guard and isolated CPU
dependency. Native physical values, wet-finite checks and full cache readback
verification remain required before releasing patch probes.
