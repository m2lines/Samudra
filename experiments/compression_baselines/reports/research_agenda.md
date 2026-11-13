<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# How this direction serves the Samudra research agenda

This note connects the compression-baseline experiment to three things: the
current goal of better short-term, single-step prediction; the known Samudra
failure modes; and the reviewer feedback on the Samudra 2 paper (meeting deck
"Meeting 11.3 — Samudra 2 revision", 40 slides, cited below by slide number).

Each claim is marked with its status. **Result** means a number from a run in
this folder ([v1](v1_om4_twodeg.md), [v2](v2_om4_twodeg.md)). **Proposed** means
an idea that has not been tested. Nothing here has been run on a Samudra
checkpoint yet.

## What the experiment provides

1. **A score.** Bits per value is the cost of encoding the true next state under
   a model's predictive distribution, which is the log score of a one-step
   forecast. In v2, the tolerance it codes to is set per variable from the spread
   of five-day changes: the normalization GraphCast-style models use for
   time-delta targets (Keisler 2022).
2. **A baseline ladder.** These are low-parameter, per-cell forecasters taken from
   lossless coding: persistence, AR(1), linear prediction, context-coded scale,
   kernel analogs, and a mixture (`mix`). They are transparent, they run in
   minutes on a CPU, and they report a number for every channel.
3. **Coder-based diagnostics.** A coder trained on the true ocean can score how
   plausible a forecast's step-to-step changes are. The blosc compressibility
   ratio is a cheaper relative of the same idea.

## 1. Single-step prediction quality

This is the current focus.

- **A floor beyond persistence and climatology (Result).** On 2° OM4, `mix` beats
  persistence in bits on all 77 channels. Its RMSE skill against persistence is
  0.04–0.16, depending on variable ([v1](v1_om4_twodeg.md); v2 numbers are in
  [v2](v2_om4_twodeg.md)). The CLAUDE.md rule "beat persistence and climatology"
  can be made sharper: beat `mix`, channel by channel, in bits and in RMSE. A
  single-step change that loses to a 63-parameter regression somewhere has a
  located defect.
- **What Samudra misses, and where (Proposed).** Turn Samudra's deterministic
  forecast into a calibrated distribution with EMOS, the standard
  post-processing method in weather forecasting (Gneiting et al. 2005): its
  forecast becomes the mean, and a context-dependent scale is fit on validation.
  Then add it to `mix`. The bits that `mix(Samudra + ladder)` saves over Samudra
  alone are information that cheap predictors find and Samudra does not. Mapping
  those bits by depth and region turns a scalar into a list of places to fix.
- **One-step artifacts (Proposed).** Slide 31 shows that the deep-salinity
  variance artifact persists when the true history is refreshed every step
  ("artifacts exist at one step"). One-step bits on truth-refreshed forecasts
  measure exactly that regime. A forecast with spurious small-scale structure
  costs more bits under a coder trained on truth.

## 2. Known failure modes

Jesse's summary of the goal is to "resolve known issues in Samudra, like noisy
fields, sawtooth trends, jumping to and sticking in 'attractor' mean states far
from target state." Each failure mode has a distinct signature for a coder
trained on the true ocean. **All three detectors are Proposed.** They are
validated first on deliberately degraded truth, then on Samudra.

| Failure mode | What a truth-trained coder sees | Cheap check | Deck evidence |
|---|---|---|---|
| Noisy fields | Step-to-step changes cost *more* bits than true changes, concentrated where the noise is | Excess bits per value of the forecast over truth, by depth and region; blosc ratio above 1 | Deep salinity variance 2.2–3.2× OM4 (slide 31); jagged deep means (slides 33–34) |
| Sawtooth trends | Successive changes flip sign: lag-1 autocorrelation of the change goes negative where the truth's is positive | Compare the AR(1) coefficient of the change, truth vs forecast, per channel. The ladder already fits it | Deep T and S time series (slides 3, 12–13, 18–19) |
| Sticking in attractor states | Changes collapse toward zero while the state sits far from truth: the forecast is *too cheap* to encode, and persistence explains it too well | Bits below truth's at long lead, together with a growing mean-state error; blosc ratio below 1 | Persistent subsurface cold bias and OHC drift (slides 7–8, 40) |

Two facts from v1 make the sawtooth check concrete. True deep tracer changes are
positively persistent: `ar1` alone has RMSE skill 0.205 at `thetao_18` and 0.220
at `so_18`, compared with persistence (Result). So a sign-flipping forecast at
depth is easy to tell apart from the truth.

### Fit with the observation metrics system

`samudra.metrics` implements the "Observation Evaluation Summary" document. Its
metrics are all computed on long rollouts (2015–2022) against observations:
- RMSE totals and maps, with block-bootstrap uncertainty;
- EKE and SSTA spatial spectra, and KE temporal spectra;
- OHC for [0, 700) m and [700, 2000) m;
- linear trends, and residual variance after removing the trend and seasonal
  cycle, as global series and maps.

Against that set, the coder diagnostics fill four gaps:

- **One step.** Every metric in the document scores a multi-year rollout, so
  nothing isolates one-step quality, which is the current focus. The ladder and
  the bits score are one-step by construction.
- **Dynamics, not amplitude.** Residual variance can catch noisy fields when the
  noise inflates variance. It cannot tell a forecast whose variance is right but
  whose steps are unpredictable noise from one whose steps follow the true
  dynamics. A truth-trained coder can, because it scores each step given the
  previous one.
- **The deep ocean.** The metrics stop at 2,000 m (the OHC layers), and the
  observations do too. The deck's deep artifacts (slides 3, 31–37) sit below
  that. OM4 truth, and so the coder, covers all 19 levels down to 6,000 m.
- **Every variable, one unit.** The spectra cover surface velocity and SST. Bits
  per value covers T, S, u, v and SSH at every level, so budgets add up across
  variables.

There is overlap. Temporal KE spectra see energy at the shortest periods, so a
surface sawtooth in velocity would show up there too. The coder's claim is
breadth (all variables and depths, per step) and attribution (where), not a
signal that spectra cannot see at all. That claim needs the validation step
before we lean on it. If the detectors pass, they would sit in
`samudra.metrics` as pure kernels next to `spectra`, following the package's
pattern. They would be computed against OM4 rather than observations, which is a
different reference from the rest of the document.

## 3. Reviewer concerns

| Concern | What this experiment contributes | Status |
|---|---|---|
| Editor: compare with "important baselines, including a physics-based one and recent ML ones" (slide 1) | A transparent statistical baseline family, scored per channel, as a floor under Samudra 2 and Neptune. It is **not** physics-based and **not** recent ML, so it does not answer the request by itself; it gives the comparison a common bottom rung | Result at 2° (ladder); Samudra not yet scored |
| R3 major 3 and R2 major 1: losses in normalized space and normalized time-delta targets, GraphCast's inverse temporal-difference variance (slides 10, 16) | v2's bins *are* that normalization: one spread of five-day changes per variable (`sigma_delta` in `results.json`). Scoring Samudra 2 and the normalized-delta variant (slide 11) in the same bits gives the ablation in the normalized space the reviewers name, with a proper score instead of MSE | Bins: Result. Scoring the variants: Proposed |
| R2 minor 1 and R3 major 4: how much skill comes from following the prescribed forcing (slides 6, 20–23) | Fit the ladder with and without `tauuo`, `tauvo` and `hfds`. The bits saved estimate how much information the forcing carries about the next state, per variable, depth and region, independently of any emulator. It complements the deck's forcing-sensitivity rollouts, which measure Samudra's *use* of the forcing over years | Proposed; next step |
| R3 detailed 16: decompose into trend and anomaly (slide 10) | Weak link. The true deep change is persistent (`ar1` skill above), which is consistent with a slow, predictable drift and with the background-anomaly target improving subsurface means (slides 12–13) | Result (`ar1` skill); interpretation is a hypothesis |
| R2 major 6: are the equatorial deep variance bands physical or artificial? (slides 30, 35–37) | Compare forecast bits under a truth-trained coder in the equatorial band and elsewhere. If OM4's own deep equatorial changes are cheap to encode and Samudra's are costly, the band is an artifact of the emulator | Proposed |
| Input context for deep T/S (slide 32: deep-only inputs improve short-lead RMSE) | Cheap feature-group ablations: drop other variables, then neighbouring levels, then velocities, and measure how many bits each group carries about deep changes. A model-free guide to input selection | Proposed |
| R1 substantial 3: multiple rollouts and error bars (slide 38) | The ladder is cheap enough to refit on many training windows, giving a spread on the baseline itself | Proposed |

## 4. Limits

- **It is not a physics baseline.** The editor's physics-based request needs
  something else, for example a coarse OM4 run or a damped-persistence model.
- **Pointwise bits against truth stop meaning anything once a rollout
  decorrelates from truth.** For rollouts, use the truth-trained coder applied to
  the rollout's own steps (section 2), not bits of truth given the rollout.
- **At a per-variable tolerance, deep five-day changes are often smaller than one
  bin.** Every model then costs about 0 bits there, so deep levels need RMSE or
  the coder diagnostics. This is a property of the score, not a defect.
- **Everything so far is 2°, one-step, without forcings.** No Samudra checkpoint
  has been scored.

## 5. Order of work

1. **Forcings in the ladder.** Use Samudra's exact `tau_hfds` inputs and timing
   convention. This answers R2 minor 1 and R3 major 4 in bits, and gives the
   ladder the same inputs as Samudra.
2. **Failure-mode detectors, validated on degraded truth.** Make synthetic
   versions of each failure on OM4 truth: added noise, blur, an alternating-sign
   (sawtooth) perturbation, and relaxation toward a wrong mean state. Check that
   each detector responds monotonically and distinguishes the failures, and
   compare with the existing spectra. This is the headline use case, and it
   needs no checkpoint.
3. **Causal spatial context rung.** Code each cell given its already-coded
   neighbours, as JPEG-LS does. This measures the joint-coding gap directly, and
   it is the coder that can see spatial artifacts (R2 major 6).
4. **The new 2° Samudra model** (needs a checkpoint on the `om4_demo.yaml`
   splits):
   - one-step forecasts through EMOS into `mix`;
   - truth-refreshed one-step forecasts and full rollouts through the detectors.
5. **Optional:** input-group ablations (slide 32) and multiple training windows
   (R1).
6. **Then scale up** to 1° and compare against Samudra 2 and its variants
   (slide 11).
