<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Global mixed latent diffusion: 16,000 updates

Authorized October 1: run the full 8,000 OM4 + 8,000 observation updates at effective
batch eight, with intermediate checkpoints and early evaluations. This replaces
the proposed stop for approval at 2,000 observation updates in the
[grain diagnostic report](grain-diagnostics.md). Full exposure is authorized;
actual allocated GPU-hours, including qualification and restarts, will be reported.

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
ready to review. No new goal object was requested.
