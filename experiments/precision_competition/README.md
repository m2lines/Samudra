<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Precision and competition in point-query transformers

The [completed first-wave report](results/README.md) contains all three paired
seeds, figures, metrics and execution receipts.

This is an experiment/results record, not intended to merge. The working hypothesis is
that improvements to a high-precision output are expensive relative to improvements
to other tasks sharing the same parameters. The primary measurement is additional
precision lost under joint training, not a requirement for exponential scaling.

For a fixed source temperature scale `s`, effective RMS bits are `-log2(RMSE/s)`.
Here `s` is the analytic population standard deviation of the zero-mean source T:
`sqrt((1+f^2)/(48*(1+f)^2)) = 0.1292985189` for fine amplitude f=0.125.
Bits lost are `log2(RMSE_joint/RMSE_solo)` at the same update or compute budget:
one lost bit means twice the RMSE. Negative values indicate positive transfer.
Every joint arm has a matching single-task configuration. Exact zeros are recorded
explicitly, not replaced by an arbitrary small error. These are empirical RMS and
tail measurements on held-out samples, not worst-case error bounds.

## First-wave protocol

The checked-in manifest contains 223 configurations: three paired seeds for 74
conditions and one untrained wire control. Each training arm runs up to 3,000
updates or 600 seconds of measured training time, whichever comes first. Batch
size is 64. Thus 192,000 fresh training draws is the maximum exposure per arm;
the training-time cap is 37 GPU-hours across all 222 trained arms. Validation,
test, startup, checkpoints, failures and preemption replay are additional costs;
report scheduler allocation separately. At most four GPU array tasks run at once.
This is a screening budget. Unreached precision thresholds remain censored; they
do not establish impossibility, a numerical floor, or a scaling law.

| Comparison | Values in the first wave |
| --- | --- |
| Fixed head-width growth | 2/4/8 heads of width 64; model widths 128/256/512; 4 blocks |
| Fixed total-width repartition | Model width 256; 2/4/8 heads; 4 blocks |
| Depth | 2/4/8/16 blocks; width 256; 4 heads |
| Dependence | Solo, independent auxiliary, dependent auxiliary |
| Main target | Copy, affine `-T+0.1`, nonlinear `sin(pi*T)/pi`, near-persistence forecast |
| Near-persistence | Nominal alpha 0.1/0.01/0.001, plus the copy experiment at zero change |
| Output formulation | MSE; CE with 16/64/256/1024/4096 uniform bins; bin-center MSE and tolerance loss at 256 bins |
| Arithmetic | FP32 primary; selected FP64 and BF16 repeats; TF32 disabled |
| Auxiliary weight | 0/0.1/1/10; separate switch-on at update 1000 |
| Copy path | Current value embedded in query; retrieval from context; physical-space residual skip |
| Other controls | Filter before versus after product; coarse-only versus detailed velocity; final RMSNorm removed; zero blocks; exact wire |

This table lists staged comparisons, not a factorial design. Capacity varies on
the copy task across solo/independent/dependent conditions. Forecast alpha varies
only at base size. CE bin counts vary only at base size for solo/independent joint
training. Other controls change one factor from base; `suite.py` and the manifest
specify every intersection. An auxiliary-only transport arm at base size estimates
standalone auxiliary improvement cost; it has main-loss coefficient zero and is
excluded from bits-lost comparisons.

The base model has width 128, 4 blocks, 2 heads, and 128 context tokens plus two
query tokens. All arms include both query tokens and both output heads. The query
receives the four current center values and position, analogous to Skyline's
explicit current-value input. Retrieval removes current temperature from the query
but leaves it in context. Both queries participate in bidirectional attention.
Blocks use pre-RMSNorm attention and a SwiGLU MLP with residual connections; final
RMSNorm precedes two separate output heads. These are synthetic scalar point queries,
not the full Skyline architecture or a compressed-latent model.

Main-loss coefficient remains one except in the explicit auxiliary-only control.
Auxiliary weight changes only its additive
loss term. Shared/trunk and auxiliary-head initialization do not change with main
bin count. Generator RNG is separate from model RNG, so paired examples match
even when heads differ. Parameters and approximate dense training FLOPs are logged;
head-count changes at fixed total width do not materially increase parameters.

AdamW uses no weight decay, 100-update linear warmup, then constant learning rate
0.0003. There is no dropout, gradient clipping, adaptive task weighting, or EMA.
This first wave has one optimizer recipe; architectural conclusions remain
conditional on it. Fixed-scale RMS, bias, p99/max errors, threshold violation rates,
auxiliary error, gradient norms/cosines, and actual paired Adam-update effects are
recorded. The Adam probe starts both updates from identical weights and optimizer
states, uses the same training batch, measures a separate held-out batch, and
restores all state before continuing training.
It compares a main-only Adam update with a main-plus-weighted-auxiliary Adam
update and reports held-out main-loss and physical MSE changes. It is an immediate
local effect, not an attribution of cumulative training failure.

## Manufactured data and dependence controls

Each example contains independent fields T, S, u, v on a periodic one-dimensional
grid. T/S have identical distributions; u/v have identical distributions. All four
are supplied to every arm. Low Fourier modes are 1-4 and high modes 12-15. Fourier
coefficients are bounded independent uniforms; high temperature amplitude is 0.125
relative to its low component, and high velocity amplitude is 1 by default. The
coarse-velocity control sets the latter to zero. Total coefficient amplitude is
normalized by the deterministic construction, not by the individual realization.
Data and reference calculations use FP64 in all arithmetic arms. The maximum product
mode is 30, below Nyquist, so reference multiplication does not alias.

Define `g(A,u) = -P_K d(A*u)/dx / K`, with K=4. This is a bounded, smooth 1-D
conservative-flux analogue; it is not a physical ocean solver or a 2-D
incompressible-advection experiment. Both fields and g are bounded by one.
The main target is T, a simple transformation of T, or `T + alpha*g(T,u)`.
Alpha is a nominal coefficient, not the achieved increment/state RMS ratio:
report measured persistence RMSE divided by the fixed source scale.
Forecast error divided by that measured persistence RMSE must accompany state-scale
bits: values below one beat persistence. High state precision alone does not show
that the small increment was learned. Auxiliary normalized RMSE uses its measured
target RMS on the fixed evaluation population; comparison of standalone fractional
error-reduction costs uses each task's own reference scale.

For copying and simple transformations, the auxiliary target is `g(S,u)` in the
independent arm and `g(T,u)` in the dependent arm. For near-persistence, the main
target is unchanged across arms; auxiliary targets are `F(v)` and `F(u)`, where
`F(w) = roll(w,-1) + 0.1*g(w,w)`. Thus auxiliary marginal distributions and
operation complexity match while sharing with the main target changes.
"Independent" refers to the underlying target-generating fields; spatial query
locations, supplied inputs, attention and weights remain shared. The forecast
control uses independent v for the auxiliary target while its main target uses T/u.

The filter-before variant additionally low-passes velocity before multiplication,
while retaining the final low-pass in both variants. This preserves a smooth output
in both and isolates whether high-frequency velocity contributes through products.
The analytic test `T=cos(12x), u=cos(13x)` produces `sin(x)/8` after product filtering
and zero after filtering velocity first. Contiguous assembled predictions on eight
fixed synthetic fields provide error spectra; random point samples are never
treated as a spatial field. Temperature-only reconstruction, followed later by real
Skyline forecasting, remains the intended transfer experiment.
The spectral diagnostic queries all 128 locations on each of eight fixed periodic
fields, computes FFTs of prediction-minus-target error with forward normalization,
and averages power across fields with conjugate-pair weights. High-band RMS sums
modes above K. No land mask, detrending, or window is needed for this periodic
manufactured domain. Forecast-increment error equals full-target error pointwise.

## Discrete outputs and precision

Uniform bins cover fixed support [-2,2] in every arm. No target clipping is allowed.
CE uses argmax-center predictions as its primary discrete output, and separately
reports continuous probability-weighted means. The oracle nearest-center RMSE is
measured against the original continuous target. Bin-center regression changes
the labels; tolerance regression uses `max(abs(error)-half_bin_width,0)^2`.
These controls separate target coarsening, finite tolerance, and categorical loss
effects, but do not by themselves isolate every head-capacity or gradient-scale
effect. CE also records accuracy, NLL and multiclass Brier score. A larger bin count
does not imply that its nominal bit count was learned. The 16-20-bit regression
thresholds are separate stress measurements beyond this bin sweep's resolution.

## Evaluation and reporting

Validation uses 1,024 examples at initialization and every 100 updates, plus the
actual final update. Test uses 8,192 disjoint examples only at the final endpoint.
Seeds for training, validation, gradient probes, spectra, and test are separate.
Three consecutive validation measurements must meet a precision threshold before
it is reported as reached. Preserve missing/censored runs and their last observed
budgets. Time-capped final endpoints are not interchangeable with 3,000-update
endpoints. `bits-lost.csv` reports exact common update checkpoints and explicitly
marks unavailable scheduled comparisons. Learning curves retain approximate FLOPs
and measured training time for compute-matched analysis. Kernel, normalization,
optimizer, validation and data-generation FLOPs are not included in the dense
FLOP approximation; measured training time includes reference generation.

The first figure is explicitly a one-seed illustration. Scientific summaries must
use the three paired seeds, retain individual results, and compare auxiliary skill.
Report the median and range of the three seed-level penalties; three seeds do not
support a precise population confidence interval. Primary contrasts are capacity,
copying versus nonidentity targets, and independent versus dependent joint losses;
other controls are exploratory. Relative standalone improvement cost and joint
penalty are separate measurements; neither alone establishes the proposed mechanism.
The dependency contrast is a particular controlled sharing relationship, not all
possible notions of task dependence. Claims of negligible effects require a
predeclared practical margin and powered confirmation, not a nonsignificant pilot.
Numerical retraining differences do not replace same-checkpoint arithmetic checks.

This first wave does not include the full block-placement experiment, a standalone
auxiliary learning-rate survey, gradient-scale-matched CE runs, parameter-matched
depth/width confirmation, or larger-model bin sweeps. Those are targeted follow-ups
after the screen, with budgets set from measured throughput and observed effects.
No expensive continuation is automatic merely because a threshold was not reached.

## Running and recovery

Run tests with a PyTorch environment, without requiring the Samudra data stack:

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest -c /dev/null -p no:cacheprovider experiments/precision_competition/test_benchmark.py
python -m experiments.precision_competition.suite --output experiments/precision_competition/manifest.json
python -m experiments.precision_competition.benchmark --manifest experiments/precision_competition/manifest.json --index 0 --output /scratch/precision-results
python -m experiments.precision_competition.report --manifest experiments/precision_competition/manifest.json --source /scratch/precision-results --output /scratch/precision-report
```

GPU qualification precedes the array. Code is staged in an immutable commit archive
and the manifest hash is verified at launch. Runs preserve atomic model/optimizer,
data/model RNG, step, measured training time, and validation-history checkpoints.
Configuration mismatches refuse resume; completed arms are skipped. Torch uses the
lzanna account, broad compatible GPU routing and preemptible/requeue support, with
scratch-backed outputs. A CPU afterany callback and local fallback reminder cover
the whole array; the callback is a wake-up, not evidence of success.

Source, container, job, runtime, GPU and result receipts are recorded with the
[submission](submission.md) and [results](results/README.md). Reports must distinguish code validation, observed execution, and
scientific results.
