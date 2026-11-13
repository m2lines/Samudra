<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Training losses from the bits view

**Status: proposal for group discussion.** Only the algebra in section 2 has
been checked, against the code. Nothing here has been trained. The terms
(bits, tolerance, the ladder) are introduced in the [overview](../README.md).

## 1. Likelihood, NLL, and why MSE is one

A probabilistic model gives a *likelihood* p(y) to each possible outcome y. The
*negative log-likelihood* (NLL) of what actually happened is −log p(y). It is
the training-loss form of the log score this experiment reports:

- **NLL in base 2 is exactly bits.** "Minimize bits per value" and "minimize
  NLL" are the same objective, up to the choice of logarithm.
- **MSE is a special case.** For a Gaussian forecast with mean μ and spread σ:

  NLL = (y − μ)² / (2σ²) + log σ + ½·log 2π.

  If σ is fixed, the last two terms are constants, so minimizing NLL is
  minimizing (y − μ)²: mean squared error. *Training with MSE is maximum
  likelihood under a Gaussian with one fixed spread.*

Everything below relaxes that assumption, one step at a time. First comes one
spread per channel, then one per cell and situation, then a precision floor,
then fields scored jointly instead of cell by cell.

**What is already established.** Training networks on NLL with a predicted
spread is old (Nix & Weigend 1994). Discretized likelihoods are standard in
image models (PixelCNN++). Probabilistic weather emulators train on proper
scores: AIFS-CRPS and FGN use CRPS, a score that is gentler on overconfidence
than the log score. The value here is in the specific connections to Samudra,
not in "a likelihood loss" as such.

## 2. Samudra 2's dynamic loss is close to a per-channel likelihood

**Status: algebra checked against the code. The interpretation needs the
group's review before it goes in the paper.**

Give each channel c its own spread σ_c, and sum the Gaussian NLL over
channels:

  L = Σ_c [ MSE_c / (2σ_c²) + log σ_c ].

1. **Best spread.** For fixed predictions, the σ_c that minimizes L is
   σ_c² = MSE_c.
2. **Substitute it back:** L = Σ_c ½·log MSE_c + constant.
3. **The gradient** with respect to the network's parameters is
   Σ_c ∇MSE_c / (2·MSE_c).

That is MSE with each channel weighted by **1 / (its current MSE)**. Kendall,
Gal & Cipolla (2018) make this point for multi-task learning.

`DynamicLoss` (`src/samudra/utils/loss.py`, `update`) does almost exactly this:

| Likelihood with optimal per-channel σ | `DynamicLoss` |
|---|---|
| Weight = 1 / MSE_c on the current batch | Weight = 1 / MSE_c, averaged over the history steps in the batch |
| Weights unbounded | The largest weight is clamped to 20 × the smallest (`limit`) |
| Weights track the current error exactly | Exponential moving average over about 25 steps (`N_WINDOW`) |

So the dynamic loss can be read as **maximum likelihood with a learned spread
per channel, regularized by a cap and smoothing**. This bears on reviewer 2's
major comment 1, "clarify what is new about the proposed dynamic weighting"
(revision deck, slide 16):
- **What it isn't:** new as an objective. It is a likelihood, and Hu et al.'s
  adaptive weighting sits in the same family.
- **What it is:** the cap and the smoothing are design choices with a clear
  role. They stop near-perfect channels (tiny MSE, huge weight) from
  dominating, which an unregularized likelihood would allow.

That reading also predicts reviewer 3's control (slide 16): fixing the final
weights should do about as well, because at convergence the weights settle
where the likelihood puts them. The deck reports that fixed final weights
"nearly match" Samudra 2 (slide 17).

**A documentation discrepancy.** `docs/utils/loss.md` says "Channels with
**higher error** receive **higher weight**". The code gives *lower*-error
channels higher weight, and the derivation agrees with the code. The doc
should be fixed in a separate change.

## 3. A spread per cell and situation

**Status: proposed.**

One spread per channel is the crudest likelihood. The ladder's results show
where the bits are. `lpc_ctx` has the *same mean* as `lpc`, so the same RMSE,
but predicts a spread from the local situation: gradients and recent change.
It is the largest single parametric gain on 44 of 77 channels in v2
([v2 report](v2_om4_twodeg.md), finding 1).

The network analogue adds an output head that predicts a spread for every
cell, trained on Gaussian or Laplace NLL.

- **What it buys:**
  - **Weighting from the data.** Calm, predictable regions get tight spreads
    and large effective weights; fronts and eddies get loose ones. No schedule
    is needed.
  - **A calibrated uncertainty map** with every forecast, which coupling to ACE
    (SamudrACE) or ensemble initialization could use.
- **Known risks:**
  - The log score punishes overconfidence harshly. A cell that collapses its
    spread can produce exploding gradients.
  - Standard remedies: a floor on σ, the Laplace distribution (lighter
    penalty), or a CRPS loss instead of NLL.

## 4. Training at a physical tolerance

**Status: proposed.**

The score in this experiment is the probability of the true value's *bin* at a
per-variable tolerance: ε × that variable's typical five-day change, as in
Keisler (2022). The same discretized likelihood can be a loss. The probability
of the true bin is a difference of two cumulative distribution values, which is
differentiable, as PixelCNN++ does for pixels.

The property that matters for Samudra is the coarse regime:
- **Small changes become cheap.** A change smaller than the tolerance costs
  almost nothing, so deep channels contribute in proportion to how much
  actually happens there at a physically chosen precision.
- **A different answer to deep-ocean weighting.** It gives a principled
  alternative to upweighting slow channels (which the dynamic loss does) or
  hand-set weights.

Whether it helps the deep-salinity artifacts (deck slides 31–37) is untested.
One hypothesis is that the network spends capacity fitting changes far below
any meaningful precision. This loss would stop rewarding that. The hypothesis
is a guess.

## 5. Beyond per-cell losses

**Status: proposed. Tracked in #981.**

Every loss above, MSE included, scores each cell independently: the total is a
sum over cells. That throws away one thing the bits view measured directly.
Surface forecast errors are spatially correlated, enough that a coder seeing
whole fields could save about as much again as the whole per-cell ladder
does:
- 0.98 bits per value for surface temperature, against 0.50 saved by `mix`;
- 1.06 bits for `uo`, against 0.21 ([v2 report](v2_om4_twodeg.md), finding 3).

A per-cell loss can't reward a model for getting that shared structure right.

Losses that score fields jointly can:
- **Ensembles with CRPS:** several samples, scored per cell, but sample
  diversity carries joint structure.
- **Ensembles with the energy score or variogram score:** these score the
  joint distribution directly.
- **Autoregressive spatial likelihoods:** each cell is predicted given the
  already-predicted ones, as the planned causal context coder does.

This matters for the known failure modes in a specific way. Neither MSE nor a
per-cell NLL fixes blur, because the best single forecast is still the average
over possible futures, and averages are smooth. Sharpness needs sampling. Naive
per-cell sampling adds uncorrelated noise, which is another known failure.
Field-level losses are the standard way to get samples that are both sharp and
spatially coherent.

## 6. How to test

Training runs need sign-off; this is a sketch. The comparison would reuse the
new 2° Samudra model being coordinated, on the `om4_demo.yaml` splits: same
architecture and data, three losses.

| Run | Loss |
|---|---|
| (a) | The current loss (dynamic) |
| (b) | Gaussian or Laplace NLL with a per-cell spread head (section 3) |
| (c) | Discretized NLL at the Keisler tolerance (section 4) |

Score all three on:
- one-step RMSE and bits (EMOS for (a), the model's own distribution for (b)
  and (c));
- the failure-mode detectors, once validated;
- the existing rollout metrics in `samudra.metrics`.

Section 2 needs no training. It is an analysis that could go into the revision
as it stands, once the group agrees with the reading. Section 5 is a larger
project; its issue describes the problem and first steps.
