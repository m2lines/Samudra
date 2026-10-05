<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global mixed latent diffusion: 16,000 updates

Authorized October 1: run the full 8,000 OM4 + 8,000 observation updates at effective
batch eight, with intermediate checkpoints and early evaluations. This replaces
the proposed stop for approval at 2,000 observation updates in the
[grain diagnostic report](grain-diagnostics.md). Full exposure is authorized;
actual allocated GPU-hours, including qualification and restarts, will be reported.
After reviewing the approximately four-day estimate, the user approved proceeding
with early results the following morning. The communicated allowance for this new
run is 900 GPU-hours, including qualification, restarts and evaluation; this
supersedes the earlier campaign cap for this run.

Use seed 1729, the comparator's quadratic mixed schedule and per-task epoch-shuffled
sample streams. Preserve 19 history bins, six native forecast leads, the supplied
six/seven observation bins and calendar-weighted monthly interiors. Use the exact
243/9/96 observation cohorts, observation normalization, zero normalized missing
inputs, explicit validity, observed-only anchoring, artificial structured hiding,
and source-specific identity input adapters. No future ocean observations are
forcing inputs. All losses and evaluation use global finite wet support and cosine
latitude weights. The ERA5 adapter is active only on observation tasks.

Keep the width-192 physical diffusion decoder and deterministic width-128 latent
processor. Use half white/half spatially correlated Gaussian noise, 32 Heun steps
for differentiable observation sampling, clean-target spatial-difference MSE for
OM4, and spatial-increment fair CRPS for observations. Spatial terms have weight
0.5 per direction, periodic longitude, and no latitude wrapping. Observation
forecast group weights are 0.8 interior and 0.1 each SST/SSH; monthly initializer
reconstruction and hidden-surface completion each have weight 0.1. Native fitting
uses channel-balanced denoising, as in previous diffusion runs, with separate
forecast, initial interior (0.1), and hidden-surface (0.1) terms. This native
probabilistic objective differs from the comparator's deterministic group MSE.

Qualification measures the **complete** batch-eight objective on H200, checks
all latent-path gradients, global support, zero placeholders and strict checkpoint
reload. Training uses AdamW (1e-4, weight decay 0.01), gradient clip one, no warmup.
Synchronous ranks sum gradients scaled by 1/8 after processing disjoint parts of
the same eight samples; changing rank count preserves effective exposure. All
training noise and missingness draws are keyed by task update and microbatch.

Save rolling optimizer checkpoints every ten total updates, and immutable weights
at observation updates 10, 25, 50, 100, 250, 500, 1k, 2k, 4k, 6k and 8k. Record exact
OM4 counts; the 2k observation milestone is global update 6,949 with 4,949 OM4
updates. Early evaluations use the nine validation origins, eight-member means,
and the comparator's frozen global integrated/spectral reference, plus member
CRPS and spatial diagnostics. Compare 32 and 64 evaluation steps on validation
before fixing the test setting. Retain fixed-exposure weights separately from
validation selection. Final reporting includes the 96 test origins and annual
cases, with member spectra and calibration rather than ensemble means alone.

Torch and Engaging are reachable (Engaging requires normal SSH rather than a
batch-mode initial authentication). Torch has the
same staged observation manifest and an existing compatible container. The new
run root is `/scratch/jr7309/diffusion-global-v1`. H200 preemptible routing gives a
substantially earlier estimate than the regular queue. Production submission
requires successful qualification and full observation payload checksum verification.

Monitor in this chat, with intervals chosen around checkpoints, job boundaries
and evaluation completion. Notify through Pushover when blocked or results are
ready to review.

## Current milestone

The October 4 update adds a [cosine LR cooldown within the final 2,000 updates](global-cooldown.md)
of the existing 16k run, with no extra training exposure. Save a full optimizer
checkpoint at step 14k, then taper from 1e-4 to 1e-6 by step 16k. The original
quadratic task schedule and exact 8k/8k exposure remain unchanged. The new trainer
is prepared for a normal job-boundary handoff; cooldown has not started.

The [2,000-observation-update report](global-2000-results.md) is available. The
composite is 1.2458 versus 0.6268 for the matched deterministic control. Grain is
substantially reduced, but surface ensembles are underdispersed; 64 inference
steps offer little benefit over 32. On October 4 at 20:25 ET the optimizer had
reached 2,776 observation / 5,764 OM4 updates. Ordinary four-H200 continuation
24761666 is running on Engaging after several automatically recovered preemptions.
Allocation through the 2k evaluations was 190.303 GPU-hours; that figure excludes
this ongoing job. Torch is reachable again, but no new Torch writer is active.
The next detailed report is planned at 4k observation updates. The following
hardware and migration entries preserve the earlier execution history.

## Qualification and launch status

### Alternative compute: Torch

The seed-fund reservation is excluded at the user's request, and its queued job
was canceled with zero allocation. The user authorized looking across other
available resources. Full-objective hardware tests on Torch both passed finite
encoder/processor/decoder gradients and strict model/optimizer reload:

| Hardware | Job | Warm OM4 seconds/sample | Warm observation seconds/sample | Peak allocated GPU memory |
|---|---|---:|---:|---:|
| RTX PRO 6000 Blackwell Server Edition | 19026429 | 1.687 | 43.930 | 43.72 GiB |
| B200 | 19026430 | 1.363 | 24.338 | 43.71 GiB |
| H200 NVL, original Engaging qualification | 24596676 | 5.112 | 37.156 | 43.71 GiB |

Each warm number averages the last seven samples of the same eight-sample
qualification objective; the first includes compilation. Native I/O/cache differs
across hosts, so native timing is not a pure GPU comparison. Steady later Engaging
H200 observation samples were about 31–34 seconds. B200 is about **1.8× faster than
RTX** in the observation test. These are single-GPU timings, not an eight-GPU
scaling measurement. RTX/B200 qualifications consumed 556/366 allocated seconds,
or **0.2561 GPU-hours combined**; their disposable updates do not enter the run.

The full **1,845,796,913-byte optimizer checkpoint** at global step 713 passed
source and destination SHA256 verification before installation on Torch. Transfer
used an authenticated SSH stream and rclone on `dtn011`; that node's software module
mount was absent, so the existing trusted Torch rclone executable was copied to
scratch and run there. Optional OSN staging denied writes and was not used.
Observation payloads and the metric reference are identical to the previously
verified Torch copies. Native OM4 uses each cluster's existing store under the
user's prior authorization for similar per-cluster versions; full native payload
identity is not claimed.

**Eight-RTX job 19027018 completed on `gr104`**, using Lzanna's normal RTX
partition, 128 CPUs and 1,400 GB host memory. It resumed the same optimizer for
200 updates, reaching global step 913 (780 OM4 / 133 observation), with effective
batch eight and `NCCL_P2P_DISABLE=1`. Both tasks have finite gradients and fresh
optimizer checkpoints. Recent median complete update times are 1.93 seconds OM4
and 44.91 seconds observation. Its 36m25s allocation consumed 4.856 GPU-hours;
the new run has consumed 16.40 GPU-hours through this segment, including all
earlier qualification, failed/preempted attempts and three early evaluations.
The broad scheduler/QOS listings were insufficient to infer its limit; actual
execution confirms all eight GPUs.

The [250-observation-update results](global-early-results.md) are now published.
Eight-RTX continuation **19028464** reached global step 1,550 in 6,365 seconds,
followed by single-RTX evaluation **19028475** in 563 seconds. Allocation through
that evaluation totals **30.70 GPU-hours**. Continuation **19032747** is running
on eight RTX GPUs toward 500 observation updates; evaluation **19032748** depends
on it.

B200 requests used the
[documented preemption-only comment](https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/).
The normal eight-B200 request was rejected, whereas preemption routing was
accepted. Slurm initially labeled it `b200`/`gpu48`, then changed it to `gpuplus`
after its dependency cleared. Accepted requests are not proof of allocation.
B200 jobs **19027159** and **19028465** were canceled before allocation because
their estimated starts were about two days away. A fresh 10 a.m. October 2
scheduler check estimated October 3 for four B200s or eight H200s, and October 4
for eight B200s. RTX training therefore continues now; alternative capacity will
be rechecked at segment boundaries. No concurrent jobs write the optimizer.

- [RTX complete-objective qualification](global-assets/rtx6000-qualified.json.gz)
- [B200 complete-objective qualification](global-assets/b200-qualified.json.gz)
- [Checkpoint migration identity and transfer receipt](global-assets/torch-migration.json.gz)
- [Hardware qualification launcher](../../../scripts/slurm_diffusion_hardware.sbatch)

### Earlier Engaging launch sequence

The [25-, 50- and 100-observation-update results](global-early-results.md) are
published. Bounded four-H200 continuations reached step 713, including successful
preemption recovery. The subsequently proposed eight-H200 reservation was canceled
at the user's request and consumed no allocation. The remaining original launch
notes preserve qualification and provenance; current compute status is above.

H200 qualification **24596676** completed 0:0 in 14 minutes. Training producer:
`4ecba36a40b7b581edeaa2292c90d1953fca7aa0`. Both full effective-batch-eight
objectives have finite, nonzero encoder/processor/decoder gradients. The global
support and zero-placeholder checks pass, as does strict model/optimizer reload.
Peak allocated GPU memory is **43.7 GiB**. CUDA allocator reserved memory is
higher; this is not a claim that the complete job fits a 48-GB GPU.

Warm microbatches average **5.11 seconds OM4 / 37.16 seconds observation**.
Observation timing includes forecast, completion and monthly reconstruction,
two members, 32 Heun steps and backward passes. The earlier 8.54-second H200
benchmark covered forecast only at 16 steps with unused initial readouts removed.
For a seven-lead month, decoder calls increase from `14 × 31 = 434` to
`30 × 63 = 1,890`: approximately **4.35×**. H200 and decoder compilation are
still enabled; completion now needs the initial readout. The workload increase
explains the new measured timing without implying that compilation regressed.

This projects approximately **752 GPU-hours** before evaluation, checkpoint I/O,
restarts and distributed overhead. Eight H200s would imply roughly four training
days, plus queue time. The approved planning allowance is 900 GPU-hours for this
new run, including overhead and evaluation.

Production job **24597458** requests eight H200s, 64 CPUs, 512 GB host RAM and
12 wall-clock hours on Engaging `mit_preemptable`. It depends on a smaller prefix
job **24598325**, requesting one H200 for up to five hours and stopping after
global update 194 (169 OM4 + 25 observation updates). This is the same scientific
run, with unchanged effective batch eight and producer. The bounded prefix uses
launcher revision `41bdcdc16`; changing rank count preserves the sample schedule
and resumes the optimizer. Evaluation **24598449** depends on that prefix and
requests one H200 for up to two hours, exporting all nine validation origins with
eight members and 32 sampling steps. It may run alongside the continuation.

The target is an initial report on October 2 morning, Eastern time, with maps,
member spectra, mean scores, CRPS and spread. Queue time remains uncertain; no
production updates have completed at this update. Checkpoint/evaluation directories
and observations are on scratch. Count all preempted/requeued allocation time
against the authorized remaining budget.

Earlier qualification attempts are retained: **24595210** failed after four seconds
because the container has no standalone `torchrun`; **24595596** failed after
95 seconds because host `CC`/`CXX` settings reached the container. The launcher now
uses `python -m torch.distributed.run` and explicitly selects container GCC/G++.
The unused pending Torch qualification **18990883** was canceled without allocation.
Qualification plus these failed attempts consumed **0.2608 GPU-hours**.

The observation scratch copy contains **350 files / 21,888,642,966 bytes**. Every
source-content hash and destination read-back hash matched the frozen comparator
manifest; eight parallel readers completed it in 150.4 seconds. Native OM4 remains
on the existing verified Engaging store. No data preparation or target changed.

- [Complete qualification measurements and contract](global-assets/qualification.json.gz)
- [Observation scratch transfer and read-back receipt](global-assets/data-staging.json.gz)
- [Training implementation](../../../src/samudra/experiments/diffusion_global_train.py)
- [Member-level evaluation implementation](../../../src/samudra/experiments/diffusion_global_evaluate.py)

Twenty-nine focused loss/model/noise/distributed-accumulation checks pass. Repository
lint, types, schema, secret scanning and REUSE checks pass. Early evaluations will
also record pixel and spatial loss magnitudes and gradients with respect to decoded
fields; those are supervision-strength diagnostics, not parameter-gradient norms.
