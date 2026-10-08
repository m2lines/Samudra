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

At 00:28 ET, recovery audit 19411638 was running (4m26s elapsed) without an
error traceback. Qualification 19411640 and production 19411643 remained
pending dependencies. No new qualification or scientific result is claimed.

## October 8 replay-check recovery

Audit callback `3512e27d-5a43-47f2-a037-8eddab8ff2d7` confirmed data/bounds
audit 19411638 completed in 5m31s. Geometry and all data checks passed. The
next qualification attempt passed for coarse+latent and both fine arms, with
estimated training times 7.95, 11.59 and 12.15 hours and peak allocated GPU
memory about 71.68 GiB. These are throughput estimates, excluding validation
and startup, not trained-model results.

Coarse/no-latent arm 19411640_0 failed the existing numerical replay check.
Checkpoint serialization and restored model/optimizer/RNG states were exact.
Its one native repeat differed by RMS 6.04e-9 / max 4.92e-6, while serialized
replay differed by RMS 1.51e-7 / max 1.94e-4. This exceeded the single-repeat
control envelope, though comparable variation occurred between native repeats
in earlier successful probes. This evidence points to an inadequately sampled
native numerical-variation estimate; it does not justify ignoring the gate.
[Full qualification evidence and failure](artifacts/early-fine-2026-10-07/torch-bounds-qualification.json).

Producer `030c96ec7205ab87f38f9cdf132d07a282db7602` changes only the
qualification diagnostic: exactly five native and five serialized trials are
run from the same checkpoint, with exact restoration checked every time.
The maximum serialized difference must satisfy the same 5× factor and the
same RMS/max-absolute floors against the maximum native variation. All trials
are retained. The test never retries until passing. **35 targeted tests pass**,
including late native jitter and deliberately corrupted resumed-update tests;
Ruff passes. Scientific training, data, optimizer, schedule, and metric
selection are unchanged.

Fresh qualification array **19412713_[0-3]** and dependent production array
**19412714_[0-3]** use isolated root
`/scratch/jr7309/runs/2026-10-08-early-fine-rtx/recovery-replay`. They reuse the
passed, hash-pinned data/bounds audit unchanged, but no fit weights or old
qualification markers are promoted. The completed audit had aged out of live
Slurm dependency records, so its successful accounting and readiness hash
were explicitly verified before submission. New qualification waited for
prior array termination to avoid overlap; production requires all new tasks
to succeed. At 00:43 ET all four replacement qualifications were running and
had advanced past fitting to cache loading.

First two completed qualification attempts used **1.2378 GPU-hours** including
failures (0.6186 + 0.6192); neither production attempt ran. Old blocked
production 19411643 and its superseded callbacks were canceled. Callback
registrations `9aad14ae-ebf3-4301-b2d8-9a92956f6ec9` and
`119b303a-1c0f-49a5-aad8-40797f2fa9ac` were revoked as superseded; the old
grid-recovery relay was stopped. New callbacks **19412774/19412775** use
`afterany` and the active `early-fine-torch-replay-callback-relay.service`.
[Replacement receipts, final prior-attempt results, and startup evidence](artifacts/early-fine-2026-10-07/torch-replay-recovery.json).
Engaging jobs remain held until all Torch qualifications succeed.

## October 8: all qualifications passed; production running

Callback `e23115bf-fbcd-4029-b600-b00c365e05f9` was verified against accounting
and output markers. All four tasks in **19412713** completed successfully.
Each has exact serialization/restoration checks, all five native and five
serialized trials, and a throughput estimate below the 20-hour training gate.
The saved numerical differences were independently checked against the
acceptance thresholds. No failed gate was overridden.

All four production tasks in **19412714** started around **00:52 ET**. At
01:03 ET they had finite training losses and gradients and positive-size
`joint-last.pt` checkpoints (about 756–763 MB each):

| Model | Completed / 4,000 updates | Qualified training-time estimate | Peak qualification GPU memory |
|---|---:|---:|---:|
| U-multitask-early | 57 | 6.46 h | 71.68 GiB |
| U-multitask-early-latent | 64 | 7.02 h | 71.68 GiB |
| U-multitask-early-fine | 28 | 10.85 h | 71.68 GiB |
| U-multitask-early-fine-latent | 28 | 9.24 h | 71.68 GiB |

These short-probe timing estimates exclude validation, startup and final
evaluation; differences between arms include I/O variability. They are not
scientific comparisons or firm finish deadlines. Model selection remains
integrated-plus-spectral observation validation.

At the recorded accounting snapshot, actual allocated GPU time was **2.6278 h**:
**1.8311 h** across all three qualification attempts, including failures, plus
**0.7967 h** of current production. Earlier production attempts used zero GPU
time. [Complete qualification, startup, and accounting evidence](artifacts/early-fine-2026-10-07/torch-production-start.json).

After verifying qualification, held Engaging arrays 25185999/25186000 and
their CPU callbacks 25219446/25219447 were canceled; their two callback
registrations were revoked. Torch production completion callback 19412775
and the active SSH relay remain in place. No held-out results exist yet.

## October 8: root cancellation and I/O diagnosis

Production callback `4bac5bca-9172-41f6-863e-202c0d620c0b` was checked against
Slurm accounting and saved outputs. **All four production tasks were canceled
by UID 0 at approximately 03:00 ET**, after 2h09m. Each saved a signal-time
checkpoint. No arm has completed training or held-out evaluation.

| Model | Saved / 4,000 updates | Mean recorded GPU utilization |
|---|---:|---:|
| U-multitask-early | 1,800 | 34.6% |
| U-multitask-early-latent | 1,836 | 36.4% |
| U-multitask-early-fine | 758 | 20.9% |
| U-multitask-early-fine-latent | 746 | 21.1% |

Utilization comes from 513–514 offline W&B system samples per arm, read
locally from saved records. These measurements support investigating Torch's
reported low-utilization cancellation policy; UID 0 and timing alone do not
prove the administrator's reason. The fine job steps reported about 7.3–7.4 TB
of disk reads each, versus about 1.1 TB for each coarse arm. The synchronous
early-example loader is a candidate bottleneck; profiling is needed to
separate storage from CPU preprocessing and other idle time.

Actual allocated time before the diagnostic is **10.4211 GPU-hours**:
1.8311 across all qualifications plus 8.5900 for production. Failed attempts
remain included. Scratch quota was 4.44 / 5.00 TB. The inspected RTX node
provided 1.7 TB of free node-local SSD space, making local staging a possible
recovery without duplicating the dataset in shared scratch.

Bounded diagnostic **19421648** runs 24 updates from an independent,
SHA-256-verified copy of the fine+latent checkpoint at step 746. Its container
mounts production scratch read-only and overlays only the diagnostic copy as
writable. Diagnostic updates will not be promoted to production. It records
CPU profiling, loader timings and one-second GPU telemetry. Scientific producer
`030c96ec7205ab87f38f9cdf132d07a282db7602`, data and model settings are
unchanged. Production will resume from its own checkpoints after the bottleneck
is addressed; unchanged low-utilization jobs have not been resubmitted.

CPU completion callback **19421734** depends on `afterany:19421648`, with
registration `37121c37-5a02-4507-9e75-d5723c73f541` and the durable local
`early-fine-torch-io-callback-relay.service` for Torch's blocked HTTPS egress.
[Cancellation and utilization evidence](artifacts/early-fine-2026-10-07/torch-root-cancellation.json).

## October 8: measured loader stalls and prefetch recovery

Diagnostic callback `37121c37-5a02-4507-9e75-d5723c73f541` was verified:
job **19421648 completed successfully**, using 649 allocated GPU-seconds.
Its separate checkpoint copy advanced from 746 to 770; those updates are
**not** included in production progress. The 24 measured updates took
308.36 seconds. Timed loader calls accounted for 158.48 seconds (51.4%) in
fine OM4 sampling and 87.28 seconds (28.3%) in observation loading. Observation
cache misses included about 59 seconds of NPZ decompression. This is a short,
instrumented run with initially cold observation caches, not an estimate that
80% of a warmed production run is necessarily lost to loading. It nevertheless
identifies substantial avoidable GPU-idle work.

Tested producer **`b4c7c05f0a63c064980b91ec41962d88d313245c`** changes only
I/O execution and its checks:

- Prefetch CPU arrays for the next two early OM4 updates, using the exact
  existing per-example seeds and consumption order. CUDA preprocessing remains
  on the training thread; prefetch consumes no model RNG state.
- Copy native early-data chunks into each job's local SSD cache as they are
  requested. Every copied chunk is read back and compared byte-for-byte before
  it becomes visible. Source datasets remain unchanged.
- Warm the existing bounded observation cache in the background. No observation
  fields, masks, normalization, loss terms, or temporal splits change.
- Add an exact serial-versus-prefetched CPU-array check during qualification.
  Existing fit, gradient-routing, five-native/five-serialized replay, and
  throughput gates remain required.

**39 targeted tests pass**, including original channel/time layout, explicit
seed/replay order, unchanged NumPy RNG, cache readback/missing-key behavior,
and propagation of asynchronous read errors. Ruff and shell syntax checks pass.
Each GPU job now requests 192 GB host memory for the bounded lookahead and
observation cache. Torch rejected a 24-CPU request before allocation; a
16-CPU request passed `sbatch --test-only` and was used. No GPU time was
charged for that rejected submission.

The recovery uses isolated root
`/scratch/jr7309/runs/2026-10-08-early-fine-rtx/recovery-prefetch`:

| Stage | Job | Required outcome |
|---|---|---|
| All four fresh qualifications | 19422275 | Fit, I/O equality, numerical replay, and throughput pass |
| CPU checkpoint-copy audit | 19422370 | All four qualifications pass; original checkpoints remain unchanged |
| Four isolated utilization checks | 19422371 | 20 warmup + 80 measured updates; mean allocated-GPU utilization ≥50% |

Dependencies are `afterok` at each transition. **Production has not been
resubmitted.** The copy audit permits only a producer update and relocation of
four run/qualification paths in checkpoint manifests. It verifies all other
checkpoint contents exactly, including model weights, optimizer, CPU/CUDA/NumPy
RNG, task counts, and accumulated elapsed time. Original attempts remain intact.
Utilization checks receive separate verified copies, with production mounted
read-only; their weights will not be promoted. Resumed production must start
from the retained 1,800 / 1,836 / 758 / 746 update checkpoints after all gates pass.

At 03:39 ET, all four qualifications were running; migration and utilization
checks were waiting on dependencies. No new qualification or utilization
success is claimed. Completed attempts through the diagnostic account for
**10.6014 GPU-hours**. Including the running qualifications at the 03:39 ET
snapshot gives **10.9286 GPU-hours**; pending stages have consumed none.
CPU callbacks **19422419/19422425/19422426** and the verified local
`early-fine-torch-prefetch-callback-relay.service` cover each stage.
[Profile, submission commands, audit/performance scripts, and accounting](artifacts/early-fine-2026-10-07/torch-prefetch-recovery.json).

At 03:40 ET, both coarse qualification probes had finite real training losses
and gradients; the fine probes were still warming their OM4 device caches.
Fresh probe losses are qualification diagnostics, not resumed-model results.

## October 8: exact-reference replay qualification

Callback `a5e99721-56fc-410d-a43f-0b0446c479de` was verified against outputs
and accounting. Prefetch qualification **19422275 passed three arms**; the fine
arm without latent channels failed its numerical replay envelope. All four
serial/prefetched CPU-array comparisons passed. The passing arms' estimated
training times were 4.57 h (coarse), 4.11 h (coarse+latent), and 5.19 h
(fine+latent), excluding startup, validation and evaluation. These short-probe
estimates suggest an I/O improvement but do not establish sustained utilization.

The failed arm restored model, optimizer and RNG state exactly. Four native
repeats were mostly identical (largest RMS difference 2.28e-9), while one of
five serialized repeats differed by RMS 1.32e-7 / maximum 1.63e-4. Other
serialized repeats were identical or much closer. The native-variation envelope
therefore failed again. Its failed record is retained; the checkpoint was not
promoted and the gate was not overridden. The pinned PyTorch source explicitly
provides a deterministic decomposition for CUDA bilinear interpolation when
deterministic algorithms are enabled, addressing a known source of native
backward variability in this network.

Producer **`b111057c72373825f1e7b643d1ee8a03daa191c3`** retains the I/O
changes and replaces the noisy serialization comparison with a **deterministic
numerical reference used only during replay qualification**. Both native and
serialized trials use deterministic algorithms and cuDNN settings, with a
cuBLAS workspace configured at qualification launch. All five native and five
serialized trials must now agree **bitwise**; any nonzero parameter difference
fails the additional check. All prior state-restoration checks remain.
Numerical settings are restored even if the diagnostic fails. Production's
numerical backend, model, losses, data, seeds and update schedule are unchanged.

**43 targeted tests pass**, including exact-reference rejection of even tiny
injected drift and restoration of numerical settings on success and failure.
New all-four qualification array **19423076** uses isolated root
`/scratch/jr7309/runs/2026-10-08-early-fine-rtx/recovery-exact-replay`.
At initial bring-up all four tasks were running; the coarse tasks had completed
fitting and entered cache warmup. No new replay success is claimed yet.

The blocked migration 19422370, utilization array 19422371, and their callbacks
19422425/19422426 were canceled without starting. Their callback registrations
were revoked and the superseded relay stopped. Copy-audit and isolated
utilization scripts have been staged for the new producer, but those jobs and
production have **not** been submitted. The copy audit additionally requires
the new `deterministic_reference_bitwise_exact` evidence. Original production
checkpoints at 1,800 / 1,836 / 758 / 746 updates remain untouched.

Completed attempts now total **11.1950 GPU-hours**, including the additional
0.5936 h for all four prefetch qualifications; the running replacement is
additional. Callback **19423241** (`afterany:19423076`), registration
`203b8dc7-568f-4b64-9efe-3fa6a1b88c87`, and the durable
`early-fine-torch-exact-replay-callback-relay.service` cover the replacement.
[Full prior failure, passing probes, replacement receipts and staged audit scripts](artifacts/early-fine-2026-10-07/torch-exact-replay-recovery.json).

## October 8: exact replay and checkpoint-copy audit passed

Callback `203b8dc7-568f-4b64-9efe-3fa6a1b88c87` was verified. **All four
qualifications in 19423076 completed successfully.** Every native repeat and
serialized replay had zero RMS and zero maximum parameter difference under
the deterministic reference. Exact state restoration and CPU I/O equality also
passed in every arm. Qualified short-probe training-time estimates were 4.80,
4.72, 6.14 and 6.16 hours for coarse, coarse+latent, fine and fine+latent,
respectively. They include the diagnostic's deterministic replay work and are
not production completion estimates.

CPU job **19423495 completed the copy audit in 68 seconds**. It verified all
non-manifest checkpoint contents exactly and confirmed unchanged original
checkpoint hashes. Only the new producer and four run/qualification paths
changed in the copied manifests. Verified resume positions are:

| Model | Global updates | OM4 updates | Observation updates |
|---|---:|---:|---:|
| U-multitask-early | 1,800 | 1,272 | 528 |
| U-multitask-early-latent | 1,836 | 1,291 | 545 |
| U-multitask-early-fine | 758 | 610 | 148 |
| U-multitask-early-fine-latent | 746 | 601 | 145 |

The audit also created and hash-verified independent diagnostic copies.
Utilization array **19423496_[0-3]** depended on successful audit completion;
all four tasks are now running. Each measures 80 real updates after 20 warmup
updates on its diagnostic copy, using the unchanged production numerical
backend. Production is mounted read-only during these checks. Their weights
and additional updates will not enter the comparison. Production has not yet
been submitted; sustained utilization remains the last recovery check.

Completed GPU attempts through exact qualification total **11.8131 GPU-hours**,
including every failed qualification, canceled production allocation and the
original I/O diagnostic. Running utilization checks are additional; the
checkpoint-copy audit used CPU resources only. Completion callback **19423518**
(`afterany:19423496`), registration `98c13e5b-3d1c-4a2d-8972-c71f1c3796d0`,
and `early-fine-torch-utilization-callback-relay.service` provide follow-up.
[Exact replay, checkpoint hashes/state audit, and utilization submission receipts](artifacts/early-fine-2026-10-07/torch-migration-and-utilization.json).
