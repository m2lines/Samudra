<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Early-data reproduction at the original 4k budget

Two fresh 4,000-update runs test whether the original early-data advantage
repeats when training length and mixture match **U-global-repro4k**. The
[6k comparison](early-rmse-6k-2026-10-10.md) stretched the mixture as well as the
training budget, so its intermediate update 4,000 is not this control.
The new runs passed data and model qualification; their results are pending. The curves below include completed historical runs and the new
4k reproductions through updates **2,300 / 1,800** (October 11, 13:46 ET).
The new curves are partial; do not interpret their current minima as final.

Contents: [Experiment](#experiment) · [Common-metric historical comparison](#common-metric-historical-comparison) ·
[Curves](#curves) · [Execution and provenance](#execution-and-provenance).

## Experiment

| Model | Why reproduce it | Recent OM4 1° | Earlier OM4 ¼° | Observation updates | Extra recurrent channels |
|---|---|---:|---:|---:|---:|
| U-multitask-early-fine | Best validation in the completed 6k RMSE wave | 1,000 | 1,000 | 2,000 | 0 |
| U-multitask-early-fine-latent | Best validation in the original 4k wave, including after RMSE-only rescoring | 1,000 | 1,000 | 2,000 | 10 |

Both have a global 1° processor with the existing encoder/decoder for the ¼°
earlier-data task. They start fresh at seed 1729 with fixed LR 1e-4, eight-example
accumulation, unchanged losses, normalization, data splits, and runtime. The
4k progressive mixture ends at exactly 2,000 observation updates; earlier and
recent OM4 alternate in the 2,000 OM4 slots. These are independent executions
of the same seed, not additional independent seeds. No cooldown is used.

**U-global-repro4k** is the completed recent-only control: 2,000 recent OM4 and
2,000 observation updates under the same 4k mixture and seed. **U-global** and
**U-global-seed1730** from the RMSE wave are the completed 6k controls, with
3,000 updates per task and seeds 1729/1730. The original U-global is the prior
4k Beta run; original early arms are the 4k Torch runs, with their recorded
recovery history. Literal model names remain unchanged; curve labels identify
the wave and budget so these different executions cannot be confused.

Select new checkpoints using the same five-component, full-field RMSE blend
and frozen validation denominators as the 6k wave. No spectra, detrending,
deseasonalization, or test data enter selection. Validation is saved every 100
total updates and observation milestones. Report minima and fixed endpoints,
physical components, and curves against total updates and observation exposure.

## Common-metric historical comparison

The original saved physical validation components were rescored using the
unchanged RMSE reference. Grid, data/statistics hashes, global support, strict
velocity support and component definitions match. This is a retrospective
curve comparison; it does not replace historical test scores with test scores
from un-evaluated checkpoints.

| Completed run | Best RMSE blend | Update | Final RMSE blend |
|---|---:|---:|---:|
| Original U-global, 4k | 0.8815 | 4,000 | 0.8815 |
| U-global-repro4k, RMSE reproduction | 0.8861 | 4,000 | 0.8861 |
| Original early-fine, 4k | 0.8704 | 4,000 | 0.8704 |
| Original early-fine+latent, 4k | 0.8559 | 4,000 | 0.8559 |

The original fine+latent gain over original U-global is **2.9% on RMSE alone**.
It cannot be dismissed as purely a spectral-score effect. Conversely, the
weaker separation at 6k does not establish that the 4k gain was noise: the
stretched training/data mixture may alter relative performance. The requested
reproductions isolate the 4k schedule under the current common Torch producer.
Cross-runtime differences remain relevant for the historical Beta control.

The 6k minima also hide a late-training difference: the final five scheduled
validations average **0.8524/0.8502** for early-fine/early-fine+latent, versus
**0.8830/0.8808** for the two U-global seeds. These correlated validation points
are a descriptive stability diagnostic, not five independent trials. Report
the mean at budget−400, −300, −200, −100 and the endpoint for every completed
run, alongside the minimum and endpoint; do not use it to change checkpoint
selection. This diagnostic was added before the new 4k results were available.

## Curves

All lines below use the same RMSE blend, including historical runs originally
selected with the integrated-plus-spectral score. Markers are actual validation
events; no smoothing or interpolation chooses minima. The late-stage views
explicitly restrict the displayed score range; the full view and CSV retain
all events. The budget is stated in each panel and legend.

![Matched schedules and late validation](artifacts/early-rmse-4k-2026-10-11/validation-late.png)

[Full training curves](artifacts/early-rmse-4k-2026-10-11/validation-full.png) ·
[Observation-exposure curves](artifacts/early-rmse-4k-2026-10-11/validation-observation-exposure.png) ·
[All values and five physical components](artifacts/early-rmse-4k-2026-10-11/validation.csv) ·
[Minima and endpoints](artifacts/early-rmse-4k-2026-10-11/minima.csv).

## Execution and provenance

Torch submission: data audit **19580149**, two-arm qualification **19580150**,
then production/evaluation **19580151**. Dependencies require every audit and
qualification to pass. Each arm requests one RTX6000, 16 CPUs and 192 GB RAM;
at most two GPUs run concurrently. No qualification weights enter training.
Qualification measured roughly 5.3–5.5 training hours, excluding validation
and startup; queue time is additional.

The unchanged successful training producer is
`eae2b5c56da44eb3686a6eaf72a80aa419a4b5a4`. The source archive and runtime are
verified again; source data are shared read-only, with new output and
qualification directories. Estimated new outputs are at most 14 GB, within
the live quota headroom; no data or checkpoints were deleted.

Cluster root: `/scratch/jr7309/runs/2026-10-11-early-rmse-4k`.
Its `paths.json`, `reproduction-contract.json`, `runtime-contract.json`,
`selection-reference.json` and `submission.json` retain the launch contracts.
Original saved validation bundles are retained in its `reporting/` directory.
[Curve provenance](artifacts/early-rmse-4k-2026-10-11/provenance.json).
Reporting script: `scripts/plot_early_rmse_reproductions.py`.
