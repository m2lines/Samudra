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

## Qualification and launch status

**Update October 2:** [First 25-observation-update results](global-early-results.md)
are complete. The eight-H200 request was unschedulable because Engaging
`mit_preemptable` caps this account at four GPUs. It was canceled without allocation
and replaced by four-H200 continuation **24611683**. The previous four-day estimate
assumed eight GPUs; actual continuation timing and Torch availability are being
checked. The launch details below preserve the original sequence.

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
