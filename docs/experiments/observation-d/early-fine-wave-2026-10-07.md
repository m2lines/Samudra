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

High-memory GPU qualification is required: the retained resident recent-OM4 cache alone is
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

## October 7 evening queue check

At approximately 21:37 ET, qualifications `25185999_[0-3]` were still pending
for scheduler priority. Production/evaluation `25186000_[0-3]` remained pending
on their success. No GPU work or model results exist yet; `squeue --start` gives
no start estimate. The passed data audit remains the only completed stage.

Durable CPU-only completion callbacks are installed: `25219446` depends on
`afterany:25185999`, and `25219447` on `afterany:25186000`. They notify this
conversation after either successful or unsuccessful termination; actual outputs
will be checked before reporting results. Endpoint authentication passed from the
login host; compute-node connectivity check job is `25219448`. Callback delivery
can itself wait for CPU scheduling. Credentials remain private and outside Git.

## Torch migration (October 7–8)

The user authorized checking and using Torch capacity. At 23:43 ET, a complete
24-hour, one-RTX, 16-CPU, 96-GiB request passed `sbatch --test-only` with an
immediate start estimate on gr101 in `rtx6000_lzanna` (scheduler preemption of
lower-priority work). The estimate is not proof of a running allocation.
Comment-routed H200 requests were rejected as unavailable under both tested
LZanna and general accounts.

Torch observation manifest/grid/statistics and both OM4 metadata SHA256 hashes
match Engaging. Data remain in place under `/scratch/jr7309/data`; a fresh full
observation verification and early-OM4 audit gate Torch qualification. Live
quota was 4.42 TB used of 5 TB, leaving about 0.58 TB. No full dataset transfer
is needed. The experiment root is
`/scratch/jr7309/runs/2026-10-08-early-fine-rtx`.

Producer `9efad7a45b1a9a328d19db888bb89f6b12d22291` changes only launcher image
and bind-path portability relative to the Engaging producer. Scientific model,
schedule, optimizer, selection and seed are unchanged. Each arm must qualify
fresh on RTX; no qualification marker is carried across hosts. Python modules
and compiled scientific/Rust extensions must match the pinned Engaging runtime;
the Torch SIF gets its own verified hash.

| Torch stage | Job / array | Dependency | Initial state |
|---|---|---|---|
| Data/runtime audit | 19410261 | None | Submitted |
| Four RTX qualifications | 19410263_[0-3] | Successful audit | Pending dependency |
| Four production/evaluation runs | 19410264_[0-3] | All qualifications succeed | Pending dependency |

Engaging arrays 25185999/25186000 were held while still unstarted, preventing
duplicate scientific runs. They remain available as a fallback until Torch
qualification succeeds.

Torch CPU-only completion callbacks 19410281/19410283/19410285 respectively
use `afterany` dependencies on the three stages. Login-host HTTPS requests to
the callback receiver time out, so a durable local service
`early-fine-torch-callback-relay.service` retrieves saved terminal-event payloads
over the authenticated Torch SSH session every ten minutes and submits the
identical payload locally. It exits after all three events are relayed. It emits
no periodic conversation updates. Relay delivery requires the SSH session;
remote event files persist if delivery is delayed. Compute-node HTTPS check
19410286 was also submitted. No callback credentials are included in the report.

[Exact Torch submission and callback receipts](artifacts/early-fine-2026-10-07/torch-submission.json)
include the source-only archive hash and bootstrap scripts. The initially copied
full repository archive was unnecessarily large because of prior report assets;
its extraction was stopped before job submission, and the jobs use a separate,
verified source-only archive of the same producer.

### Confirmed RTX execution at 00:01 ET October 8

Data/runtime audit 19410261 completed successfully in 4m36s. All four
qualification tasks were **running** on gr101/gr102/gr104/gr103 respectively.
Each passed its ten-update observation fitting check and advanced to loading
the 54.63-GiB recent-OM4 cache for the mixed-task/resume probe. Logs identify
NVIDIA RTX PRO 6000 Blackwell Server Edition. Full qualification, throughput
acceptance, production and held-out evaluation are still pending.

The Torch SIF SHA256 is
`b77c03ca4810dc13c7cb970b10edfbd279b2da17e16973d8df2bfbccae8b7d81`. It differs
from the Engaging SIF container file, while all checked scientific library
versions and compiled scientific/Rust binary hashes match. All 350 observation
NPZ checksums and the early-source audit passed again on Torch.
[Verified startup evidence](artifacts/early-fine-2026-10-07/torch-startup.json).

Compute-node callback HTTPS check 19410286 timed out too. The installed local
SSH relay supplies the completion-delivery route; CPU callback jobs save their
event before attempting HTTPS, allowing identical replay through that relay.

## October 8 qualification failure and bounds recovery

Callbacks `37c06a82-364e-47b3-b9da-5492fe283e3e` and
`0e550751-873b-467b-92c6-f6bf8a0efc7a` were handled together. Coarse arms
19410263_0/1 completed mixed-task and disk-resume qualification; their estimated
training times were 8.37 and 7.85 hours, respectively, with peaks of 71.68 GiB.
Fine arms 19410263_2/3 failed in coarsener construction with `KeyError: lat_b`,
before any joint training update. Production 19410264 was canceled by the
scheduler after the failed dependency and used zero GPU time. First-attempt
qualification usage was 0.6186 allocated GPU-hours, including both failures.
[Accounting, failures and successful coarse checks](artifacts/early-fine-2026-10-07/torch-first-qualification.json).

The consolidated OM4 stores retain center coordinates but omit corner geometry.
The loader incorrectly assumed `lat_b` and `lon_b` were present. Recovery reads
the original published Gaussian grid geometry from
[180×360](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/grids/gaussian_grid_180_by_360.zarr/.zmetadata)
and [720×1440](https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/raw/grids/gaussian_grid_720_by_1440.zarr/.zmetadata),
checks that its centers match the data arrays, and binds file hashes into the
readiness contract. The CPU audit now constructs the actual coarsener and checks
full-sphere area and constant preservation before allowing GPU qualification.
No inferred or interpolated replacement bounds are introduced.

Fix producer: `bace89544af6bee819e9da480d7280c8d5e2b51f`. **33 targeted tests
passed**, including published-grid, mismatch rejection, area, model, and resume
tests; Ruff passed. All four arms requalify fresh in
`/scratch/jr7309/runs/2026-10-08-early-fine-rtx/recovery-grid-bounds`. No old fit
weights or qualification markers are promoted to the new producer. Original
checkpoints, logs and contracts are preserved. Engaging jobs remain held and
would also require this fix before any reuse.

Recovery submissions: CPU audit **19411638**, qualification array
**19411640_[0-3]**, production/evaluation array **19411643_[0-3]**. Their
`afterok` chain again requires the audit and all four qualifications to succeed.
Callbacks **19411674/19411675/19411679** use the corresponding `afterany`
dependencies and the verified local
`early-fine-torch-grid-callback-relay.service`. Initial state is queued;
submission is not qualification or scientific success.
[Recovery submission receipts](artifacts/early-fine-2026-10-07/torch-bounds-recovery.json).

A direct lightweight CPU check in the pinned Torch container verified both
published grids against the actual stored `x`/`y` arrays to absolute tolerance
1e-10 degrees (180×360 and 720×1440). Full payload and coarsener preflight
still run in audit 19411638; this coordinate check does not replace that gate.
