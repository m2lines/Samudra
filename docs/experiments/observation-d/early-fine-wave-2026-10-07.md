<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Earlier OM4, fine encoder/decoder, and recurrent memory

The user authorized this four-arm wave on Engaging on October 7. It tests whether
broader temporal coverage helps observation forecasts, whether learning from
fine-resolution examples improves transfer, and whether fine supervision makes
recurrent latent state more useful. No LLC data enter this wave.

## Runs and matched exposure

| Literal model name | Recent OM4 1° updates | Earlier OM4 updates | Observation updates | Extra recurrent channels |
|---|---:|---|---:|---:|
| U-multitask-early | 1,000 | 1,000 at 1° | 2,000 | 0 |
| U-multitask-early-latent | 1,000 | 1,000 at 1° | 2,000 | 10 |
| U-multitask-early-fine | 1,000 | 1,000 at ¼° | 2,000 | 0 |
| U-multitask-early-fine-latent | 1,000 | 1,000 at ¼° | 2,000 | 10 |

The external control is [U-global](extent-review-2026-10-05.md), which uses 2,000
recent OM4 updates and 2,000 observation updates. All new arms retain its U-Net
widths 128/192/256/384, seed 1729, AdamW LR 1e-4, weight decay 0.01, clipping 1,
eight accumulated examples, InstanceNorm, input task adapters, and no warmup.
They start fresh; fitting/probe weights are never loaded into production.
The same progressive mixed schedule gives 218/407/593/782 observation updates
in successive 1,000-slot blocks. Alternating OM4 slots select recent then early
examples. Their dates are paired across the four new arms by deterministic seed.
This matches updates, not FLOPs. The prior U-global ran on Beta; hardware/runtime
are not exactly matched and will be reported with the new results.

Recent training remains 1975-01-03 through 2013-10-04. Earlier windows draw from
1,241 five-day frames in 1958–1974: 1,217 possible starts with all 25 frames
strictly before 1975. Each example has 19 history frames and six forecast steps.
No documented scientific reason for the inherited 1975 cutoff has yet been
established; early-period spin-up/distribution differences remain an interpretation
caveat. One-degree and quarter-degree stores are different preprocessing releases,
although complete time vectors match. We do not attribute effects solely to
resolution without acknowledging this difference.

## Fine-task architecture and loss

The processor always operates globally on 180×360 cells. Fine examples supply
720×1440 histories of SST, SSH, forcing and validity. Their original 77 state
channels are U/V/T/S at 19 depths plus SSH. The optional 10 channels are unscaled
learned state, initialized and autoregressed on **every** task.

The fine-specific encoder consists of two stride-2, 3×3 convolutions with 32
features and GELU, followed by a pointwise projection to two coarse states. It
predicts a correction to the shared initializer's coarse-history estimate;
available coarse SST/SSH copying remains intact. No true full-depth states enter
initialization. All shared weights start identically between fine and coarse
arms with the same latent count; extra modules use isolated RNG state.

After each shared processor step, the fine decoder reads all predicted state
channels plus geography/season. A 3×3 convolution to 64 features, GELU and
pointwise projection to 77×16 channels followed by pixel shuffle produce a
fourfold upsampled residual over bilinearly interpolated coarse physical fields.
Longitude convolution and reference interpolation are periodic; latitude convolution padding is zero. Decoded fields
are masked at native depths. The recurrent path carries the **coarse predicted
state, including latent channels**; it never re-encodes true future fields.
Fine I/O modules are inactive on recent-OM4 and observation tasks.

Fine-task forecast loss is 0.5 × group-balanced native-field MSE plus
0.5 × group-balanced coarse-physical-state MSE, averaged over six leads. Coarse
targets are exact wet-area overlap means of the native fields using actual
spherical cell bounds, not assumed 4×4 nesting. The equal split retains the
physical interpretation of the shared state while supervising native outputs.
Initializer reconstruction uses the same 50/50 fine/coarse split, coefficient
0.1, excluding copied surface channels. Hidden-surface completion has coefficient
0.1 and is scored on decoded native initial surfaces. The encoder sees only
surface/forcing histories; full states are targets only.

Coarse early examples use the unchanged OM4 forecast/reconstruction/completion
objective. Physical normalization remains observation-training-derived; OM4
forcing normalization remains the existing 1° statistics. There is no new
per-cell physical normalization or spectral training loss.

## Selection and diagnostics

Selection remains the frozen U-global integrated-plus-spatial-spectral observation
validation metric, every 100 total updates plus observation milestones. The
reference SHA256 is `c94c0601e168858e4ba41423aed74d8604eaa1efbc68dd76370b32b8eeacd26c`.
The final report will include the selected forecast, its own initialized
persistence, all component metrics, and validation curves versus total and
observation updates. Held-out evaluation uses the same 96 monthly origins in
2015–2022, not one continuous eight-year rollout.

## Engaging execution and qualification

Data already present:

- `/orcd/data/abodner/002/jrusak/om4_onedeg_v3`
- `/orcd/data/abodner/002/jrusak/om4_quarterdeg`
- `/orcd/pool/008/jrusak/diffusion-interior-engaging/data/observation-bundles/samples`

Experiment root: `/orcd/pool/008/jrusak/early-fine-20261007`.
The isolated observation view is fully rehashed (350 NPZ files) before training;
source datasets are not modified. Native early chunks are checked for expected
sizes, identical timestamps and representative finite wet values. This is not a
full fresh native-payload checksum audit.

H200 qualification is required: the retained resident recent-OM4 cache alone is
about 55 GiB, so L40S is unsuitable without a separate cache change. Early data
are streamed with eight reader threads rather than resident on the GPU. The
existing x86 Apptainer image is pinned by SHA256; a private gsw overlay leaves
shared environments untouched. Runtime module and extension hashes are checked.

Each arm passes a ten-update observation fitting check, then a real mixed-task
GPU probe including fine I/O gradient checks and a disk checkpoint/optimizer/RNG
restore comparison. Throughput must predict at most 20 training hours before the
24-hour preemptible production job proceeds. Estimates exclude validation and
initial cache warming; these are measured in qualification logs. The global
cohort, producer, model options, data readiness hash and selection reference are
part of the qualification/resume contracts.

Production uses one GPU per arm, at most four concurrent array tasks. A checkpoint
is written every ten updates as well as at validation boundaries. SIGUSR1/SIGTERM
requests a checkpoint at the next update boundary, then requeues; abrupt preemption
can lose work since the last checkpoint. Jobs never reset scientific progress or
reuse another arm's checkpoint. Slurm dependencies gate production on successful
data preparation and all four qualifications. No additional seeds or variants are
included.

## Status

Checked at 2026-10-07T19:03:34.587072+00:00.

GPU implementation is pinned at `c4d726764be3909156ad5fbf1d5fc3ecb158c425`.
The independent data audit used `87d49ee7d72607ad96a90286d6e302075bb97aa2`;
qualifications bind its completed readiness manifest.
Local validation: **32 targeted tests passed**, Ruff and shell syntax checks
passed, and the pinned Engaging container imported the implementation and Rust
reader successfully. Tests verify identical shared initialization, correct
fourfold spatial reduction, wet-area conservation, fine encoder/decoder gradients,
latent-state recurrence/gradients, periodic decoder reference interpolation,
no gold-state leakage into initialization, and
existing task/resume behavior.

| Stage | Slurm job / array | Dependency | Last observed state |
|---|---|---|---|
| Data and container audit | 25185472 | None | Completed successfully (4m28s) |
| Four fitting/mixed-resume qualifications | 25185999_[0-3] | Successful data audit | Pending: Priority |
| Four production/evaluation runs | 25186000_[0-3] | All qualifications succeed | Pending: Dependency |

The completed audit verified all 350 observation NPZ checksums, exact time-vector
agreement, and 99,280 expected-size early chunks per OM4 source. Wet-value
finiteness was sampled at frames 0, 620 and 1240 across all 80 variables.
Original arrays 25185474/25185475 were canceled before starting (zero GPU use)
to correct periodic longitude interpolation before qualification; their receipts
are retained.

Array order follows the run table above. No GPU work or scientific result is
claimed yet. [Exact submission commands, paths and runtime hashes](artifacts/early-fine-2026-10-07/submission.json).

The Engaging image SHA256 is
`08e81775209ac46315f8e65e854750147644f806fd3af6a6dd5d9d4df8a54008`, with PyTorch
`2.12.0a0+0291f960b6.nv26.04.48445190`, NumPy 1.26.4, Zarr 2.18.7 and an isolated
GSW 3.6.20 overlay. This differs from the previous Beta runtime and will remain
visible in cross-campaign comparisons. The project pool has 488 GiB free at
submission; no new full-resolution dataset copy is being made.
