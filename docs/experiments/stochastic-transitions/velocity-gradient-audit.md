<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Velocity gradient audit at 2k updates

A has a deterministic initializer and processor; B has a stochastic initializer
and deterministic processor.

At identical stochastic-CRPS initializer outputs, changing only the loss to MSE
increases the output-gradient magnitude in the strongest 1% of reference currents
by about tenfold. The same change slightly decreases gradient magnitude in the
weakest half. This supports a loss-dependent emphasis on strong-current errors,
rather than a uniform shortage of gradient. It does not by itself establish that
this difference causes the held-out velocity gap.

The audit used raw 2k checkpoint parameters from deterministic MSE, deterministic
MAE, stochastic CRPS, and stochastic per-member MSE runs. Each checkpoint saw
three fixed OM4 training examples and three fixed observation training examples.
Stochastic probes used four members, as in training; each probe was one example,
not an effective-batch-eight accumulated update. No optimizer steps were taken.
BF16 autocast and FP32 loss matched training. Activation checkpointing bounded
memory without changing the intended objective. These are current local gradients,
not gradients recorded during the original training trajectory.

## Identical-output loss comparison

For each raw initializer output, the audit detached and re-enabled gradients on
the output tensor, then evaluated per-member-averaged MSE, per-member-averaged MAE, and fair CRPS on those identical
values. The 0.1 reconstruction coefficient, normalization, wet mask, cosine-latitude
weights, and all-channel averaging remained unchanged. Reference speed is
`sqrt(u² + v²)` in physical units. Speed quantiles used wet cells only and were computed separately at
each depth and example. Table entries average over three examples, u and v, and
19 depths. They are gradients with respect to normalized model outputs.

| B-CRPS output, substituted objective | Weakest 50%: mean absolute gradient | Strongest 1%: mean absolute gradient |
|---|---:|---:|
| MSE | 6.44e-9 | 1.088e-7 |
| MAE | 1.158e-8 | 1.155e-8 |
| Fair CRPS | 7.10e-9 | 1.101e-8 |

The strongest-current bin has average absolute normalized residual 4.53, versus
0.267 in the weakest half. Squared loss emphasizes those large residuals.
MAE's output gradient does not grow with residual magnitude. The CRPS spread term
changes the member-specific gradient further.

About 37% of fair-CRPS wet-cell velocity member output gradients are smaller than 1e-12 overall;
about 27% are in the strongest-current bin. For four untied members, the lowest
member below the target and highest member above it have zero derivative under
the fair estimator. This finite-ensemble cancellation is expected, and is not
evidence that expected CRPS is improper or that the whole network receives no
gradient. Other members, locations, and examples still train shared parameters.
See the [analytic and toy checks](velocity-diagnostics/toy-mechanisms.md).

## Gradients reaching the initializer

The audit separated the actual weighted loss into OM4 initializer velocity,
OM4 initializer other fields, OM4 forecast, observation initializer surface, and
observation forecast. The components reproduce the original total loss within
numerical tolerance. The table averages the per-example L2 norms for OM4 losses.
Norms below cover initializer parameters, excluding the
processor and forcing adapter; the JSON files also retain adapter/shared norms.

| Trained checkpoint/objective | Mean velocity reconstruction gradient norm | Mean forecast gradient norm |
|---|---:|---:|
| A: MSE | 0.0554 | 0.2334 |
| A: MAE | 0.00692 | 0.2163 |
| B: CRPS | 0.0196 | 0.2671 |
| B: per-member MSE | 0.0601 | 0.2607 |

These rows differ in learned parameters as well as objective. The identical-output
comparison above isolates the loss derivative more directly.

For B-CRPS, OM4 forecast and velocity reconstruction gradients have positive
cosines 0.28–0.51. These are positive local directional alignments on these three
examples. The cross-source cosine between OM4 velocity reconstruction and
observation forecast is -0.067, 0.054, and 0.001. These probes do not show strong,
consistent opposition by the forecast objective. Different OM4/observation
examples were paired; this is not an estimate of the full-data expected gradient.

No clipping was applied. The reported shared-parameter norms exclude processor
gradients, so they cannot determine whether the original global gradient clip
at one activated, or reproduce an Adam update with its accumulated moments.

## A controlled follow-up

At the same B-CRPS outputs, an auxiliary ensemble-mean velocity MSE restricted
to OM4 initialization has initializer-gradient norms 0.0307, 0.0275, and 0.0199.
It retains the original 0.1 coefficient and 77-channel denominator, so adding it
with coefficient one is a precisely specified intervention. Its gradient cosine
with the existing velocity CRPS term is 0.50–0.70.

The combined velocity gradient norm for CRPS plus that unit auxiliary matches
multiplying the existing velocity CRPS term by 2.19, 2.27, and 1.83 on the three
training probes. A fixed multiplier of 2.2 is therefore a useful magnitude
control for local initializer-gradient magnitude, selected from training gradients
rather than validation scores. It does not match the full-objective gradient or
the Adam parameter update.
Compare continuation from the same B-CRPS checkpoint with unchanged training,
with the velocity-only mean-MSE auxiliary, and with velocity-only CRPS multiplied
by 2.2. Leave every forecast, observation, and non-velocity term unchanged.

Finite-member ensemble-mean MSE still contains a variance penalty proportional
to 1/member-count in expectation for conditionally independent draws. Preserve member CRPS, spread, and maps in the
follow-up; improved mean RMSE alone cannot demonstrate a useful stochastic fix.

[Exact JSON inputs/results and CSV summaries](results/velocity-diagnostics/gradients/)
retain checkpoint and script fingerprints. Torch job 21382062 completed in
2m23s; the first successful audit took 2m13s. A preceding diagnostic array/list
indexing error cost 1m05s. Total allocated GPU time was about 0.095 GPU-hours.
The process-local THP and cuDNN workarounds were retained. No training was run.
