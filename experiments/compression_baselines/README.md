<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Compression baselines: scoring ocean forecasts in bits

This is an experiment record, not intended to merge. This page explains, from
first principles, what the experiment does and why. The results are in
[`reports/`](reports/):

| Report | What it is | Status |
|---|---|---|
| [`v2_om4_twodeg.md`](reports/v2_om4_twodeg.md) | Current protocol: per-variable tolerance bins, Samudra's 2° splits | Complete |
| [`v1_om4_twodeg.md`](reports/v1_om4_twodeg.md) | First OM4 run, with a fixed 8,192-bin quantization | Frozen record |
| [`research_agenda.md`](reports/research_agenda.md) | How this serves Samudra's goals, the reviewer feedback and the metrics system | Draft |
| [`losses_from_bits.md`](reports/losses_from_bits.md) | Training losses from the bits view: Samudra 2's dynamic loss as a likelihood, per-cell spreads, tolerance-based losses, field-level losses (#981) | Proposal for discussion |

## 1. Why: the problem this addresses

Samudra is a neural network that emulates an ocean simulation. Given the ocean
state now, it predicts the state five days later. It then feeds that prediction
back in as input, again and again, to produce a multi-year *rollout*. The
simulation it learns from is OM4, GFDL's global ocean model. The state is
temperature (`thetao`), salinity (`so`) and the two horizontal velocity
components (`uo` eastward, `vo` northward) at each of 19 depth levels, plus sea
surface height (`zos`): 77 two-dimensional fields, called *channels*.

Every rollout is a chain of single steps, so errors in one step become inputs to
the next. Three failure modes are known in Samudra's rollouts:
- **noisy fields**, which jitter from step to step;
- **sawtooth trends**, where successive steps overshoot in alternating
  directions;
- **attractor states**: rollouts drift to a mean state far from the truth and
  stay there.

Reviewers of the Samudra 2 paper also asked for comparisons against simpler
baselines, and for evidence of how much skill comes from the prescribed
atmospheric forcing rather than the ocean dynamics the network learned.

The existing evaluation (`samudra.metrics`) scores multi-year rollouts against
observations: RMSE, spectra, heat content and residual variance. That is the
right final test, but it leaves three gaps:

1. **No bar for a single step.** Improving one-step quality is the current
   priority, and the only one-step reference is *persistence*, the forecast that
   nothing changes. Persistence is easy to beat. It doesn't tell you whether a
   model has learned what is cheaply learnable.
2. **No common unit.** RMSE is in °C for temperature and m s⁻¹ for velocity, and
   it scores only a single guess. It cannot add up across variables or say how
   confident a forecast should be.
3. **No direct test of step-to-step dynamics.** A field can have the right
   variance and still evolve implausibly from one step to the next.

## 2. The idea: a forecast is a compressor

### A forecast as a probability

A *probabilistic forecast* assigns a probability to each possible next value,
rather than a single guess. One standard way to score it is the *log score*:
−log₂ of the probability the forecast gave to what actually happened. A
confident, correct forecast costs little. A confident, wrong one costs a lot.

The log score is a *proper scoring rule*: a forecaster minimises its expected
score only by reporting its true beliefs. So it rewards both accuracy and honest
uncertainty, and it can't be gamed by hedging.

### Why "bits" and "compression"

Shannon's coding theorem connects the log score to data compression. If a sender
and receiver share a forecast P, then *arithmetic coding* can transmit the true
value in almost exactly −log₂ P(value) bits. So:

> **the average log score of a forecaster equals the size of the compressed
> future, given the past.**

A better forecaster is a better compressor. This turns forecasting into a
problem with fifty years of engineering behind it. Lossless audio and image
codecs (FLAC, JPEG-LS, PAQ) are, at their core, cheap and carefully tuned
predictors of the next sample.

This gives us one score that applies to any forecaster with a predictive
distribution: **bits per value**. Lower is better. Bits are additive: they sum
over cells, levels and variables, so the same number breaks down into maps,
depth profiles and per-variable budgets.

### Tolerance: what "lossless" means for real numbers

Compressing a real number losslessly needs a precision. Each value is rounded to
a bin of width *w*, and the forecast is scored on the probability it gives to
the true value's bin. *w* is the *tolerance*: we transmit the ocean to within
*w*.

Two regimes matter:
- **Fine:** *w* is much smaller than the forecast's spread. Halving *w* adds
  exactly one bit to every model's cost, so *differences* between models don't
  depend on *w*. They equal the difference in log-likelihood.
- **Coarse:** *w* is comparable to the actual five-day change. Then most true
  next values stay in the current bin, every model costs close to 0 bits, and
  the score stops telling models apart. This happens in the deep ocean, where
  little changes in five days.

### Choosing the tolerance: Keisler normalization

Weather and climate emulators face the same problem when they combine errors
across variables in a training loss. Keisler (2022, *Forecasting Global Weather
with Graph Neural Networks*, arXiv:2202.07575, §3.3.3) scales each variable by
the spread of its *time differences*, using one scale per variable shared across
all levels, so "1 K of temperature error should count the same" at every level.
GraphCast-style models use the same family of normalizations, and Samudra 2's
reviewers asked for it.

We set the tolerance the same way:
- σ_Δ is the area-weighted standard deviation of a variable's five-day change,
  pooled over all its levels and the training years.
- The bin width is *w* = ε·σ_Δ, where ε is a small number.
- ε is chosen by a rule fixed in advance: the largest value at which the surface
  channels are in the fine regime (`select_eps.py`).

The result is one physically meaningful precision per variable. The deep ocean
then often lands in the coarse regime, which is the honest answer: at that
precision, little happens there in five days. RMSE remains the score that tells
deep forecasts apart.

## 3. The baselines: a ladder from lossless coding

A *baseline* is a simple forecaster that a complex one should beat before its
complexity is justified. We build a *ladder* of them, each a small step up from
the last, borrowing the predictors lossless codecs use.

Every rung predicts the five-day change in one channel at one grid cell, its
*tendency* x[t+1] − x[t]. The inputs are the two most recent states (t−1 and t)
in a 3×3 neighbourhood, plus the same cell's recent tendencies in the other
variables and in the levels above and below (20 features). Each rung outputs a
full probability distribution, so each can be scored in bits.

| Rung | What it does | Borrowed from |
|---|---|---|
| `persistence` | Predicts no change, with a fitted spread | |
| `ar1` | Continues a fraction of the last change | Autoregressive models |
| `lpc` | A linear combination of the 20 features, fit per latitude band | FLAC's linear prediction |
| `lpc_ctx` | `lpc`, with a spread that depends on the local situation (gradients, recent change) | JPEG-LS and CALIC context modelling |
| `analog` | Finds the 32 most similar situations in the training data and uses what happened next | Kernel analog forecasting |
| `lpc_analog` | `lpc`, plus the errors `lpc` made in those similar situations | |
| `mix` | A weighted mixture of all rungs, with weights fit on validation data | PAQ context mixing |

`mix` is the bar to clear. Each rung has about a hundred parameters or fewer per
channel (plus a library of past examples for the analog rungs), and the whole
ladder fits on a CPU in minutes.

**Scoring a deterministic model.** Samudra outputs one state, not a
distribution. To score it in bits, its forecast becomes the centre of a
distribution whose spread is fit on validation data. This is *EMOS* (ensemble
model output statistics; Gneiting et al. 2005), the standard way weather centres
calibrate deterministic forecasts. Adding that rung to `mix` measures what the
cheap predictors know that Samudra does not. This is planned, not yet run.

## 4. Diagnostics built on the same idea

- **Bits saved over persistence, by depth and as maps.** These show where each
  model finds information.
- **Joint-coding gap.** The ladder codes each cell independently. If its errors
  are correlated between neighbouring cells, a coder that sees the whole field
  could do better. We estimate the gap from the correlation ρ between neighbours'
  errors as −½·log₂(1−ρ²) bits. This is a Gaussian approximation; a *causal
  context* coder, which codes each cell given its already-coded neighbours as
  JPEG-LS does, would measure it directly. That coder is planned.
- **Blosc compressibility.** Blosc is the generic compressor used inside our
  Zarr data stores. The ratio of a forecast field's compressed size to the true
  field's is a cheap smoothness check: below 1 means too smooth, above 1 too
  noisy. It only becomes informative on rollouts.
- **Failure-mode detectors (planned).** A coder trained on the true ocean can
  score a forecast's own step-to-step changes:
  - noise makes them more expensive than the truth's;
  - a sawtooth flips the sign of their step-to-step correlation;
  - sticking in an attractor state makes them too cheap.

  See [`research_agenda.md`](reports/research_agenda.md).

## 5. Protocol

| | v1 (frozen) | v2 (current) |
|---|---|---|
| Data | OM4 2°, `v2026-09/om4_twodeg` | same |
| Channels | 77: `thetao`, `so`, `uo`, `vo` at 19 levels, plus `zos` | same |
| Inputs | States t−1 and t; no atmospheric forcing | same (forcing is the next step) |
| Bins | 8,192 per channel over that channel's range of values | Unbounded, width ε·σ_Δ per variable (Keisler) |
| Train | 2003–2004 | 2009-10 to 2013-09 (the last 4 years of Samudra's training window) |
| Validation | 2010 | 2013-10-05 to 2014-10-05 (Samudra's) |
| Test | 2015 | 2014-10-10 to 2022-12-24 (Samudra's inference window) |
| Scoring | 8,000 test rows per channel, sampled in proportion to cell area; cheap rungs also on every wet test cell | same |

v2 replaced v1 for two reasons:
- **Bins.** v1's bins came from each channel's range of *values*, so their width
  had nothing to do with how much the ocean changes in five days. Surface
  bins were far finer than the typical five-day change, while deep temperature
  and salinity bins were wider than it.
- **Splits.** v2 uses the splits in `src/samudra/configs/data/om4_demo.yaml`
  (the 2° preset), so a Samudra model trained on that preset can be scored on
  exactly the same test window.

v1 was originally designed to match an earlier all-fields classifier
experiment; that constraint was dropped.

## 6. Files

| File | Role |
|---|---|
| `data.py` | Caches OM4 (or a synthetic toy) as one `.npy` file per split and channel |
| `coding.py` | Tolerance bins, interval probabilities, the bits score, blosc helpers |
| `models.py` | The 20 patch features and the ladder |
| `run.py` | Fits the ladder per channel, tunes on validation, scores on test, computes the diagnostics |
| `select_eps.py` | Picks ε from a sweep by the fixed rule |
| `report.py` | Tables and figures from `results.json` |
| `sanity.py` | Independent checks: recomputes σ_Δ, bin widths and persistence bits with plain numpy |
| `run_all.sh` | Runs everything end to end; `SYNTHETIC=1` for the smoke test |
| `test_coding.py` | Unit tests of the bit accounting |
| `om4_twodeg_v2_results/` | v2 outputs, including the ε sweep |
| `om4_twodeg_results/` | v1 outputs (frozen) |
| `synthetic_smoke/` | Toy-model outputs from before OM4 was reachable; not about the ocean |

## 7. Run it

```
pip install xarray zarr numcodecs fsspec aiohttp dask cftime scikit-learn scipy matplotlib
bash run_all.sh                              # OM4 2°: cache, eps sweep, all 77 channels, report, checks
SYNTHETIC=1 bash run_all.sh /tmp/c /tmp/o    # smoke test, about 2 minutes
python -m pytest -c /dev/null test_coding.py # -c /dev/null skips the repo's pytest config
```

`cftime` is required because the store's time axis uses a Julian calendar.
The v2 cache takes 4.5 GB of disk. Runs so far used the DTN (32
cores, 251 GB RAM) with Python 3.12 and the package versions listed in each
report.
