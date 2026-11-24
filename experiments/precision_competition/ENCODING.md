<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Scalar encoding and auxiliary interference

This wave tests whether scalar output encodings preserve copying accuracy when a
shared transformer also learns a clean or noisy auxiliary task. It is an
experiment/results record, not intended to merge. No outcome is assumed.

## Targets and architecture

Reuse the synthetic fields and width-128, depth-4, two-head transformer from
`benchmark.py`. The current temperature value is present in the query. Normalize
that field by the generator's analytic standard deviation (approximately
0.1292985189), round all its input positions to the nearest fixed-point grid
value with spacing 1/4096, and copy the query's quantized scalar. Ties use
round-to-even. Codes 0 through 65535 decode as `code/4096 - 8`. The generator's
analytic support lies strictly within this range; reject escaped original
values. The quantized field, multiplied back by the fixed scale, is the backbone input; the
copy head's target/output uses normalized units. All arms see identical inputs
and targets; no head receives integer codes as inputs. This is fixed-point
encoding, not prediction of IEEE float16 bit patterns.

The auxiliary is `(0.1 - S(query))/scale`, where S is an independent field.
S has the same generator distribution and analytic standard deviation as the unquantized temperature. Here `scale` is that same analytic standard deviation. Its objective is ordinary MSE in normalized units. The noisy arm adds fresh
independent unit-standard-deviation Gaussian noise to its training targets,
never to inputs or clean evaluation targets. The auxiliary uses the second query representation and shares the transformer.
Each sample has 128 periodic positions and four independent fields T,S,u,v,
with Fourier modes 1-4 and 12-15. T/S fine-mode amplitude is 0.125 relative to
coarse modes; u/v fine-mode amplitude is one. Each token contains four physical
field values and sine/cosine position features. Two query tokens contain the
same selected position and its four values, with different learned role vectors.
All 130 tokens attend together. The first query feeds the copy decoder; the
second feeds the auxiliary head. The affine auxiliary is directly recoverable
from the input.

Run the backbone once per example. All copy decoders have a width-128 SiLU hidden
layer. Parallel heads emit their logits at once. Autoregressive heads reuse one
MLP four times, conditioning on the main representation, center/8,
`log2(radius/8)/16`, and a four-element level indicator. Parameter counts are
recorded; decoder parameter counts and compute differ between arms.

## Encodings

| Name | Output and objective |
| --- | --- |
| mse | Scalar, normalized MSE |
| asinh_mse | Scalar, MSE against asinh(target), sinh decoding |
| radix | Four independent 16-way digit predictions, mean CE |
| parallel_nested | Four independent 16-way asinh subdivision predictions, mean CE |
| ar_uniform | Four conditioned uniform subdivisions, mean CE |
| ar_nested | Four conditioned asinh subdivisions, mean CE |
| ar_overlap | Four conditioned overlapping center selections, hard CE |
| ar_twohot_hard | Same overlapping geometry, two-hot CE, hard selection |
| ar_twohot_soft | Same overlapping geometry, two-hot CE, weighted-mean selection |
| flat_twohot | 64 asinh-spaced centers, scalar-coordinate two-hot CE, mean decoding |
| flat_gauss | 64 bins uniform in asinh space, Gaussian histogram CE, mean decoding |

Radix emits the four base-16 digits from most to least significant, reconstructs
`4096*d0 + 256*d1 + 16*d2 + d3`, then applies the fixed-point inverse.
Flat two-hot centers are `8*f(linspace(-1,1,64))`. Nested and overlapping
hard predictions decode to the final selected center.

Start center=0, radius=8. The local asinh-spacing map is
`f(u)=sinh(u*asinh(3))/3`, for uniform u on [-1,1]. Nested geometry uses 17
boundaries, selects the containing interval (boundary ties go right), and
recenters/rescales to that interval. Uniform geometry replaces f(u) by u.
Overlap geometry uses 16 candidate centers including endpoints, selects the
nearest center (ties use first index), and divides radius by eight after each
selection. Adjacent next ranges overlap throughout the parent range.

Two-hot weights interpolate the two surrounding centers in normalized scalar
coordinates. Hard decoding chooses the largest probability (ties use first
index). Soft decoding uses the weighted mean of candidate centers. Every later
head conditions on the center produced by its own decoding rule. Prefix states
and targets are detached: later losses do not differentiate through earlier
center choices. They still train the shared representation and decoder.

Autoregressive training uses target-derived prefixes for the first 1,000 of
12,000 solo updates, then model-generated prefixes. The warm-up uses nearest
centers for overlapping hard arms, containing intervals for nested arms, and exact two-hot means for the soft arm. Evaluation and
all continuations always use model-generated prefixes. Targets are recomputed
relative to each actual state; out-of-range targets receive endpoint labels.
Log range misses. Parallel nested targets always follow the canonical true
nested path; inference follows the four predicted subdivisions with no feedback
to later heads. Prediction is tested for independence from supplied targets.

For flat_gauss, 65 edges uniformly span [-asinh(8), asinh(8)], Gaussian sigma is
one bin width in that transformed coordinate, and target masses are Gaussian
CDF differences, renormalized after truncation. Decode by the probability mean
of sinh-transformed bin midpoints. This introduces smoothing/transform and edge
bias, which the numerical audit reports; it is not a lossless encoding.

`encoding_audit.json` evaluates all 65,536 words using float64 perfect-target
predictions. This is the greedy encoder's reconstruction error, not a search for
the best possible overlapping path or a neural-network precision bound. Radix is
exact. Uniform nested midpoint decoding is offset by half a grid step from this
fixed-point grid. Overlap sacrifices hard resolution for correction range.
Two-hot soft decoding can interpolate exactly but requires precise probabilities.
The uniform-over-words audit differs from the concentrated training distribution.

## Loss scaling and training

Average four categorical losses rather than summing. At initialization, match
main-loss shared-backbone gradient RMS to a fixed, seed-matched linear normalized
MSE reference, measured over four independent batches of 64 examples. Exclude the
auxiliary head from this norm. The reference is a seed-matched, randomly initialized linear scalar head on the
same initial backbone. Sum squared gradient components over shared parameters
and the four batches, then use `weight=sqrt(reference_sum/main_sum)`.
The joint objective is `weight*main_loss + auxiliary_MSE`; clean and noisy
auxiliary coefficients are both one. Auxiliary-only probes use auxiliary MSE
without the main term. Keep the resulting coefficient fixed throughout
all phases and record it and both measured squared norms. This matches an initial
scale, not subsequent directions, noise, or effective Adam updates. A coefficient
sensitivity study remains a follow-up, not part of this launch.

Use FP32 with TF32 disabled, AdamW with zero weight decay, fixed batch 64, learning
rates 1e-4 and 1e-5, and initialization seeds 11, 22, 33. All arms use matched
training examples and separate, matched noise streams. Raw weights are primary;
EMA 0.99 is reported separately. Hold the GPU type to L40S.

There are 66 bundles (11 encodings x 2 LRs x 3 seeds). Each starts with 12,000
solo updates. Qualification requires at least four RMS bits in each of the last
three validations. Fit only the auxiliary linear head by minimum-norm least
squares on 4,096 separate examples; require clean auxiliary normalized RMSE <0.2
on 1,024 separate validation examples. Record head weight norm/bias because fits
can differ in conditioning. Qualification never uses final test metrics.

Qualified bundles continue from identical prepared weights for 3,000 updates in
three arms: main-only, clean joint, and noisy joint. Each starts fresh Adam state
and EMA. All three start with the same fitted auxiliary head. Up to 198 such
continuations run; failures to qualify remain reported copy-learning or auxiliary
qualification failures.

Save the first post-warm-up checkpoint sustaining 4 and 6 RMS bits for three
validations. Separately qualify its auxiliary head and run paired 256-update
main-only, clean auxiliary-only, and noisy auxiliary-only probes, recording
1,4,16,64,256 steps. Up to 396 short probes run. Actual starting precision and
training age must be reported; threshold labels are not exact precision matches.
The same precision threshold across encodings does not imply equal model state.
Save the third checkpoint of the first qualifying sequence. Probes use the
bundle LR with fresh optimizer/EMA, matched data/noise streams, and update every
parameter receiving an auxiliary gradient (shared backbone and auxiliary head;
copy decoder has no auxiliary gradient). Gates use raw weights.

Validate on 512 fixed examples every 100 updates and at probe milestones; final
independent test uses 8,192 examples. Measure normalized RMSE, RMS bits
`-log2(RMSE)`, bias, p99 and maximum absolute error, exact value and rounded-word
accuracy, stage errors, stage accuracy, range misses, probability margins and
entropy, and stage error conditional on an incorrect first choice. Stage accuracy compares argmax with the current nearest-center/containing-bin
label; two-hot accuracy uses nearest centers too. Range-miss rates are fractions
of examples at each stage. Continuous/flat arms have only one stage; radix
reports digit accuracy plus final numerical error. Shared gradient norms and
main/clean-auxiliary and main/noisy-auxiliary cosines use a fixed diagnostic
batch at initialization, each 1,000 updates, and endpoints; they are not
estimates of gradient-noise covariance. Saturating
rounded-word accuracy is secondary; numerical error remains authoritative.

The main interference metric is paired `log2(RMSE_joint/RMSE_main-only)` at equal
updates within encoding/LR/seed. Also report absolute accuracy and loss from the
fork point. A low-resolution method can appear stable by remaining inaccurate.
Report all seeds; three-seed ranges are not confidence intervals. Keep raw/EMA
results, encoding limits, acquisition failures, and interference separate.
Clean/noisy comparisons estimate the effect of adding label noise with this
loss and optimizer; they do not alone identify gradient conflict as its mechanism.

## Operations

`encoding_smoke.json` tests every encoding for 20 solo and 5 updates per
continuation, with reduced validation/test sizes. Smoke explicitly bypasses the
scientific qualification gate and must never enter scientific results.
`encoding_manifest.json` fixes the scientific bundle specifications.

Checkpoints every 100 updates include model, optimizer, EMA, data/noise RNG states,
curves and elapsed time. Each phase resumes its latest checkpoint; completed
phases restore their final state. Requeued tasks reuse the same output folder.
Source archive and manifest hashes, Slurm job IDs, Torch version, GPU identity,
and W&B group identify the run. Permit at most one infrastructure retry per failed
bundle; numerical failures require inspection, not silent continuation. Bound
allocation at 66 one-hour scientific slots plus 11 ten-minute smoke slots;
actual usage is expected to be lower and must be measured before any expansion.

The monitor must verify all outputs, retrieve curves and provenance, create
learning and interference figures plus machine-readable tables, and publish them
to the existing draft experiment PR #967. A completed array alone is not a
scientific result. Keep unqualified arms visible in the report.
