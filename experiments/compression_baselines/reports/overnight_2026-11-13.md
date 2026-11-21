<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Overnight results, 2026-11-13: the compression ladder and Samudra 2 at 1°

**Status: written before the 10:00 meeting.** The 1° forcing run finished
overnight and is in section 4b. Every number comes from a file in this folder,
named in each section. Terms are introduced in the [overview](../README.md),
and the mathematics in [background](background.md).

## What we ran

| Run | Data | Output folder |
|---|---|---|
| Ladder, one step, protocol v2 | Official `v2026-09` 1° store, Samudra's splits | `om4_onedeg_v2_results/` |
| Samudra 2 (paper checkpoint `ema_ckpt.pt`), one step, true history refreshed every step | Same store and test window | `samudra2_onestep_onedeg_results/` |
| `samudra.metrics` against observations, 2015–2022: the paper's Samudra 2 rollout, and a persistence rollout | Same pipeline for both | `metrics_onedeg_results/` |
| Ladder rollouts at 2° | Diverged; root cause in [rollout_blowup_rca.md](rollout_blowup_rca.md) | `rca_twodeg_results/` |
| Ladder with forcing, one step, 2° | Paired with the v2 run | `om4_twodeg_v2_forcing_results/` |
| Ladder with forcing, one step, 1° | Paired with the 1° v2 run | `om4_onedeg_v2_forcing_results/` |

## Headlines

1. **At one step, Samudra 2 is the best predictor of ocean currents, and worse
   than persistence for temperature and salinity at every depth.**
   - *Velocity:* its RMSE skill against persistence is about 3× the ladder's.
   - *Temperature and salinity:* it loses to persistence even at the surface
     (−0.08 and −0.12), and badly at depth (−26 and −36 at 6,000 m).
   - *The ladder,* by contrast, beats persistence on every variable.
2. **Over 2015–2022 against observations, a frozen ocean (persistence) scores
   better than Samudra 2's rollout on velocity RMSE and heat-content RMSE.**
   Samudra 2 wins clearly on SST and on the spatial pattern of variability. So
   velocity RMSE and per-area OHC RMSE barely tell these models apart.
3. **The ladder can't be rolled out yet.** It tracks persistence for about
   three months, then diverges. The root cause is an unconstrained grid-scale
   response that grows 1.38× per step, accelerated by quadratic advection
   features. That is a general caution for any model trained on short
   horizons.
4. **The official 1° release is bit-identical to `om4_onedeg_v3`** for the
   prognostic fields: maximum difference 0, or about 2 × 10⁻⁹ for velocity. The
   paper model runs on it, with Samudra deriving `hfds_anomalies` itself.
5. **Forcing at Samudra's input times adds little at one step, at 2° or 1°**:
   at most 0.10 bits at 2° and 0.13 bits at 1°, both on near-surface
   temperature. The large surface joint-coding gap remains at both
   resolutions.

---

## 1. One step at 1°: Samudra 2 against the ladder, on the same truth

Sources: `samudra2_onestep_onedeg_results/samudra2_vs_ladder.md` and
`rmse_by_channel.json`, and `om4_onedeg_v2_results/results.json`.

**Samudra 2's forecasts.** 299 one-step forecasts over 2014-10 to 2022-12, one
every other test step, since the model predicts two five-day steps per call and
we keep the first. Each starts from the true states (t−1, t). They are scored
on every wet cell.

**The ladder's.** Scored on 8,000 area-sampled rows per channel, over every test
step. That's the same population, but a different sample.

**Bits** use the ladder's 1° tolerance, ε = 1/4 of each variable's σ_Δ. For
Samudra 2 that's minimal EMOS: one Laplace spread per channel, fit on its
validation errors.

**Alignment check.** Persistence bits recomputed on Samudra 2's forecast times
match the ladder's full-field persistence bits to within 0.004 bits on all 77
channels. Forecasts are closest to the truth at their labelled time t+1, not at
t or t+2 (checked on 5 channels).

### Per variable

Wet-area-weighted over levels, surface level in brackets:

| Variable | Persistence bits | Ladder `mix` bits | Samudra 2 bits | RMSE skill, ladder `mix` | RMSE skill, Samudra 2 |
|---|---:|---:|---:|---:|---:|
| `thetao` | 2.59 [4.77] | 2.02 [4.32] | 3.51 [4.96] | 0.153 [0.132] | −1.281 [−0.083] |
| `so` | 1.99 [4.45] | 1.48 [3.80] | 3.20 [5.13] | 0.124 [0.036] | −2.916 [−0.118] |
| `uo` | 2.90 [4.99] | 2.45 [4.75] | 2.45 [4.76] | 0.107 [0.061] | 0.352 [0.188] |
| `vo` | 2.97 [4.79] | 2.42 [4.47] | 2.36 [4.41] | 0.166 [0.132] | 0.445 [0.305] |
| `zos` | 3.68 | 3.45 | 3.96 | 0.037 | −0.076 |

### By depth

RMSE skill against persistence, where 0 is persistence, positive is better and
negative worse:

| Depth (m) | 2.5 | 105 | 550 | 1,050 | 2,400 | 4,000 | 6,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `thetao`, Samudra 2 | −0.08 | −0.42 | −0.41 | −0.55 | −3.29 | −1.36 | −26.3 |
| `thetao`, ladder | 0.13 | 0.10 | 0.14 | 0.27 | 0.07 | 0.11 | 0.22 |
| `so`, Samudra 2 | −0.12 | −0.77 | −1.44 | −2.76 | −9.50 | −6.72 | −36.2 |
| `so`, ladder | 0.04 | 0.14 | 0.19 | 0.22 | 0.12 | 0.06 | 0.16 |
| `uo`, Samudra 2 | 0.19 | 0.44 | 0.45 | 0.40 | 0.27 | 0.24 | 0.29 |
| `uo`, ladder | 0.06 | 0.10 | 0.16 | 0.15 | 0.08 | 0.08 | 0.08 |
| `vo`, Samudra 2 | 0.31 | 0.53 | 0.54 | 0.51 | 0.34 | 0.31 | 0.35 |
| `vo`, ladder | 0.13 | 0.16 | 0.22 | 0.18 | 0.14 | 0.12 | 0.10 |

**How to read it.**
- *A skill of −26* means an RMSE 27 times persistence's. Deep five-day changes
  are tiny, so any added error dominates.
- *Velocity:* Samudra 2 is genuinely better than cheap statistics.
- *Tracers and SSH:* its single steps add more error than the change they
  predict. That fits the deck's slide 31 ("artifacts exist at one step") and
  the noisy deep time series on slides 33–34.
- *Caveat on bits:* one EMOS spread per channel handicaps Samudra 2 in bits,
  because the ladder's `lpc_ctx` adapts its spread to the situation. RMSE has
  no such handicap.

**This is the most important result for your current focus on single-step
quality.** Before acting on it, it should be confirmed independently with the
team's own tooling, for example a one-step validation loss on these years.

## 2. Rollouts against observations, 2015–2022

Sources: `metrics_onedeg_results/paper_samudra2_metrics.log` and
`persistence_rollout_and_metrics.log`. Both rows were scored by
`python -m samudra.metrics` with the bundled `samudra_om4_v2/eval.yaml`, the
official 1° data and the bundled `obs.yaml` (`v2026-07` observations).

**What "persistence" means here.** The persistence rollout repeats the true
state of 2014-10-15 for every step through 2022-12-24. It is the zero-skill
reference.

| Metric | Samudra 2 (paper rollout) | Persistence (frozen ocean) |
|---|---:|---:|
| Surface geostrophic velocity, total RMSE (m s⁻¹) | 0.2031 | **0.1994** |
| Instantaneous surface EKE, total RMSE (m² s⁻²) | **0.0456** | 0.0468 |
| SST, total RMSE (°C) | **1.182** | 2.622 |
| SST residual-variance map RMSE | **0.468** | 0.600 |
| SST residual-variance pattern correlation | **0.520** | −0.148 |
| OHC 0–700 m, per-area total RMSE (J m⁻²) | 4.32 × 10⁹ | **4.22 × 10⁹** |
| OHC 700–2,000 m, per-area total RMSE (J m⁻²) | 3.04 × 10⁹ | **2.69 × 10⁹** |
| Upper-700 m OHC residual-variance map RMSE | 5.60 × 10¹⁷ | **4.21 × 10¹⁷** |
| Upper-700 m OHC residual-variance pattern correlation | **0.669** | 0.070 |

**Reading it.**
- **Velocity and per-area OHC RMSE barely discriminate.** A frozen ocean scores
  better than Samudra 2. Velocity RMSE is dominated by eddies that no
  non-eddy-resolving forecast can place correctly. OHC RMSE is likely dominated
  by Samudra 2's known cold drift (deck slide 40).
- **SST and the variability-pattern metrics do discriminate.**
- **On the OHC residual-variance *map* RMSE, Samudra 2 is worse than predicting
  zero variability**, even though its pattern correlation is far better. That
  reads as too much variability, in roughly the right places, consistent with
  noisy fields.

**These numbers do not match the metrics doc.** The doc gives a Samudra 2
velocity RMSE of 0.2102 and SST of 0.9357; this run gives 0.2031 and 1.182.
OM4's own velocity RMSE also differs (doc 0.2090, this run 0.1940). So the
difference is in the pipeline or observation versions, not the predictions.
Comparisons *within* this table are consistent; agreement with the doc is not
claimed. The OM4 row printed only its velocity metric in this run, so it's
omitted.

**The ladder rollout couldn't be scored.** The full ladder rollout became
non-finite at step 33, and the linear one overflows float32 at step 297. The
[root-cause analysis](rollout_blowup_rca.md) explains why. A stabilized ladder
(fit on multi-step rollouts, or with grid-scale damping) is needed before it
can serve as a rollout baseline.

## 3. The ladder at 1°

Source: `om4_onedeg_v2_results/tables.md`, `sanity.json`,
`eps_sweep/eps_selection.json`. ε = 1/4, chosen by the same rule as at 2°,
which there gave 1/8. So **absolute bits are not comparable across resolutions;
bits saved within a resolution are.**

| Variable | Bits saved by `mix` over persistence | RMSE skill of `mix` | Est. surface joint-coding gap (1°) | Same, at 2° (v2) |
|---|---:|---:|---:|---:|
| `thetao` | 0.58 [0.45] | 0.153 | 1.10 | 0.98 |
| `so` | 0.51 [0.63] | 0.124 | 0.35 | 0.19 |
| `uo` | 0.46 [0.23] | 0.107 | 1.19 | 1.06 |
| `vo` | 0.56 [0.32] | 0.166 | 0.53 | 0.55 |
| `zos` | 0.23 | 0.037 | 1.32 | 1.19 |

**Checks.**
- Persistence recomputed by hand matches the run to within 0.0004 bits.
- Bin widths match to 6 × 10⁻¹⁵ (relative).
- `mix` is worse than its best component on 3 channels, all at the deepest
  levels where every model costs about 0 bits: `thetao_18` by 0.0003, `so_17`
  by 0.0009, `so_18` by 0.0001.

## 4. Forcing at 2°

Source: `om4_twodeg_v2_forcing_results/compare_vs_no_forcing.md`. The
comparison is paired on identical test rows, verified by identical persistence
bits.

Adding `tauuo`, `tauvo` and `hfds` at the input times, as Samudra receives them:

| Variable | Bits saved by forcing [surface] | RMSE skill of `mix`, before → after [surface] | Surface gap, before → after |
|---|---:|---:|---:|
| `thetao` | 0.02 [0.10] | 0.141 → 0.187 | 0.98 → 0.92 |
| `so` | −0.01 [0.01] | 0.111 → 0.115 | 0.19 → 0.19 |
| `uo` | 0.01 [0.02] | 0.070 → 0.078 | 1.06 → 1.05 |
| `vo` | 0.00 [0.03] | 0.165 → 0.178 | 0.55 → 0.54 |
| `zos` | 0.02 | 0.051 → 0.053 | 1.19 → 1.19 |

Forcing at the *input* times carries little extra information about the next
five-day mean, and it doesn't explain the correlated surface errors. One
hypothesis, not yet tested: the forcing *during* the forecast interval matters,
and Samudra's convention doesn't give it. That would cost something only where
future fluxes are knowable, not in coupled runs with ACE. A ladder variant with
t+1 forcing would measure it.

## 4b. Forcing at 1°

Source: `om4_onedeg_v2_forcing_results/compare_vs_no_forcing.md`. Paired with
`om4_onedeg_v2_results/` on identical test rows; `compare.py` checks that the
persistence bits match and refuses otherwise. Same ε (1/4) as the 1° ladder.

| Variable | Bits saved by forcing [surface] | RMSE skill of `mix`, before → after [surface] | Surface gap, before → after |
|---|---:|---:|---:|
| `thetao` | 0.02 [0.09] | 0.153 → 0.163 [0.132 → 0.182] | 1.10 → 1.03 |
| `so` | −0.01 [0.01] | 0.124 → 0.122 [0.036 → 0.037] | 0.35 → 0.35 |
| `uo` | 0.00 [0.02] | 0.107 → 0.107 [0.061 → 0.071] | 1.19 → 1.18 |
| `vo` | 0.00 [0.04] | 0.166 → 0.167 [0.132 → 0.144] | 0.53 → 0.52 |
| `zos` | 0.01 | 0.037 → 0.039 | 1.32 → 1.31 |

What this adds to the 2° result:

- **The same picture at 1°.** The largest saving is 0.13 bits on `thetao_2`
  (22.5 m), and surface temperature's RMSE skill rises from 0.13 to 0.18. That
  is where surface heat flux should act. Elsewhere forcing is worth a few
  hundredths of a bit at most.
- **Forcing slightly *hurts* below the mixed layer.** Held-out bits get worse
  by up to 0.044 on `thetao_7` (250 m), 0.040 on `so_9` (550 m) and 0.032 on
  `thetao_8`. Three extra regressors carrying no information about those
  levels add fitting noise, and the `mix` weights don't fully remove it. A
  model should get forcing only where it's informative, or be regularized so
  that it can ignore it.
- **The surface gap is unchanged** (temperature 1.10 → 1.03 bits). Forcing at
  the input times doesn't explain the correlated surface errors at either
  resolution. The t+1-forcing hypothesis in section 4 is still the open test.

## 5. For the meeting: decisions and next steps

1. **Confirm the one-step tracer result with the team's own tooling.** If it
   holds, it's the clearest target for single-step work:
   - first, where the error comes from, i.e. noise against bias (the
     residual-variance map numbers suggest noise);
   - then whether the normalized-delta or background-anomaly variants (deck
     slide 11) fix it. `one_step.py` scores any checkpoint in about 30 minutes
     on the DTN CPU.
2. **Metric calibration.** Velocity RMSE and per-area OHC RMSE can't separate
   Samudra 2 from a frozen ocean, so they shouldn't carry headline weight in the
   revision without a persistence reference beside them.
3. **Rollout stability diagnostic.** The power iteration from the RCA only
   needs Jacobian–vector products, so it would work on Samudra via autograd. It
   would give Samudra's leading error-growth factor and mode cheaply.
4. **Baseline roadmap:**
   - a stabilized ladder for rollouts (multi-step fit);
   - a t+1-forcing variant;
   - the input-group ablation for deep T/S (revision item 5).

## Provenance

- **Machine:** the DTN (`dtn012`), CPU only. Logs are in `/scratch/am16581/agent-logs/`, and copies of the key ones are in each results folder.
- **Code:** commits are recorded in each `git_hash.txt` and run log:
  - 1° ladder: `8c8d9629`;
  - one-step Samudra 2: `27260885`;
  - metrics: `1acc9b39`;
  - scoring: `4c65c332`;
  - 2° forcing: `0f43c79f`;
  - 1° forcing: `4c65c332` (rerun after a shell-quoting failure in the chained job; log in the results folder).
- **Runtimes:**
  - 1° ladder: data 341 s, ε sweep 1,523 s, run 7,592 s (`timings.tsv`);
  - Samudra 2 one step: about 4 s per forecast on CPU.
- **Not committed:** the caches (1° about 18 GB), the forecast arrays (`pred.npy`, 5.6 GB for the test window) and the rollouts. They stay on DTN scratch, under `/scratch/am16581/compression_baselines/`.
