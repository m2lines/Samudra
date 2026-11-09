<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Auxiliary updates after copying has been learned

This experiment tests whether unlearned or irreducibly noisy auxiliary targets
perturb a learned copy function. It reuses the constant-LR copying models from the
[previous qualification](followup_results/README.md). The six starting checkpoints
are the first sustained raw 4- and 6-bit checkpoints for seeds 11, 22 and 33.
Their actual precision differs from their threshold label and will be reported.
No matched raw 8-bit checkpoints exist. All computation uses FP32, TF32 disabled,
and L40S GPUs, with the original width-128, four-block transformer and 128-position
synthetic periodic fields. Copying predicts T at the query position. The query
includes current field values, so this is not a separate-token retrieval test.

The auxiliary is an affine function of a second independent temperature-like field:
`auxiliary target = 0.1 - S(query position)`. T and S are independent and identically
distributed. Losses are MSE divided by the fixed temperature variance,
`0.1292985189 squared`. Noise is fresh independent Gaussian noise added to the
auxiliary target with standard deviation `sigma * 0.1292985189`. It is never
included in the model input. Metrics compare auxiliary predictions with the clean
affine target, even when training labels are noisy. The constant predictor 0.1
has population normalized RMSE one for this auxiliary.

## Preparation

Keep the copying backbone and output head fixed. Fit the auxiliary linear output
head by minimum-norm least squares on 4,096 separate calibration examples.
Qualification requires clean auxiliary normalized RMSE below 0.2 on a separate
1,024-example validation set. Save both the original untrained auxiliary head and
the fitted head. Changing this head cannot change the copy output because the two
heads operate separately after the shared transformer. This directly checks
learnability on the fixed representation; it replaces the need for a separate
auxiliary model when qualification passes. A failed fit is reported, and that
checkpoint's learned/noisy conditions are skipped; unlearned perturbations and
main-only controls can still run. Do not adapt the auxiliary target or fit the
backbone in response to a failed fit.

Each preparation job also checks an accumulated batch-1024 noisy Adam update on
a disposable model clone. Source checkpoint hashes are verified before loading.

## Perturbation and recovery

| Sweep | Values |
| --- | --- |
| Learning rate at batch 64 | 3e-4, 1e-4, 3e-5, 1e-5 |
| Batch at LR 3e-4 | 64, 256, 1024 |
| Auxiliary conditions | Unlearned clean; learned clean; learned sigma 0.1, 0.3, 1 |
| Controls | Main-only; learned clean plus constant loss offset 1 |
| Auxiliary-only updates | 256, with measurements at 1, 4, 16, 64, 256 and additional matched-example points |
| Recovery | 256 more updates using main loss only |

The overlapping LR/batch configuration runs once. Every trajectory begins with
fresh AdamW state, zero weight decay and constant LR, preventing inherited main-task
momentum from contaminating the primary auxiliary-only intervention. All paired
conditions use the same starting backbone, data order and separate noise RNG.
The only initial learned-versus-unlearned difference is the auxiliary head.
Batch size is implemented by accumulating 64-example microbatches and taking one
optimizer step. This preserves example ordering across batch sizes. A constant
loss offset enters the differentiated loss scalar as well as the reported
auxiliary loss; its derivative is zero. Keep this numerical negative control distinct from noisy targets.

Measure raw and EMA (decays 0.99 and 0.999) copy RMSE, bias, tail error and clean
auxiliary error. EMA starts at each fork and does not alter raw training. Effective
copy bits are `-log2(RMSE / 0.1292985189)`. Report absolute bits lost from the
starting checkpoint and the difference from the corresponding main-only
trajectory separately. Evaluate trajectories on 256 fixed validation examples;
final independent test uses 8,192 examples after recovery. The unrecovered endpoint
at 256 steps is preserved in validation history and the trajectory's checkpoint
history. Report comparisons at equal updates and equal examples, and compare
copy damage at comparable auxiliary accuracy where their achieved ranges overlap.
Do not extrapolate unmatched auxiliary accuracy.

## Frozen-checkpoint probes

For each checkpoint, condition and LR/batch setting, sample 16 independent
auxiliary batches. Restore the identical checkpoint before every update and
measure the resulting copy-output displacement on 64 fixed held-out inputs.
Use plain SGD, fresh Adam, and inherited Adam as separate diagnostics. Inherited
Adam loads the original copy optimizer moments but overrides LR. An additional
zero-gradient coasting probe exposes movement from those inherited moments.

Save all displacement vectors, their mean displacement RMS, their average
per-output sample variance, alignment with the starting copy error, and actual
mean copy-MSE change. The MSE change decomposes exactly into twice error times
displacement plus squared displacement. These are finite-update effects, not an
assumption that the model is locally linear or that parameter-gradient cosine
explains interference. Sixteen batches give noisy variance estimates; report their
limitations and the three training seeds separately. The same probe RNG seeds
pair noise conditions and optimizer comparisons.

## Longer joint training and limits

After perturbation artifacts verify, run the raw 6-bit checkpoints for 3,000 joint
updates across the same six LR/batch settings. Include main-only, learned-clean
joint, and learned sigma-1 joint. Auxiliary weight is one. This fixed second stage
compares noise with the clean baseline without selecting favorable effect sizes.
Only checkpoints whose auxiliary fit qualified are eligible for this stage.

There are six preparation bundles, 36 perturbation bundles (up to seven paired
trajectories each), and up to 18 joint bundles (three trajectories each). Each
bundle requests one L40S for at most 30 minutes, at most four concurrently. The
upper allocation bound is 30 GPU-hours before bounded infrastructure retries;
actual runtime is expected to be much lower and must be measured. Allow at most
two infrastructure retry submissions and two extra allocated GPU-hours. No
capacity, tolerance-loss, dependent-target or real-ocean extension is authorized
by this protocol.

Atomic checkpoints retain model, Adam, EMAs, data/noise RNG states, histories and
step count. A preempted bundle resumes its current trajectory and skips completed
trajectories and probes. Work since the most recent checkpoint can repeat; paired
RNG state makes the resumed sequence reproducible. Per-stage immutable manifests,
source commit, parent hashes, scheduler receipts and prepared-model hashes preserve
provenance. All results remain an experiment record in PR #967, not production code.

## Definitions for interpretation

The original field construction is specified in [benchmark.py](benchmark.py):
independent random Fourier coefficients at modes 1–4 and 12–15, with mean-zero
T and S and variance equal to the fixed normalization above. Queries are uniform
spatial positions. The model attends over all positions plus two task-query tokens;
the scalar heads consume their respective final query representations. Exact parent
paths and hashes are in [the preparation manifest](noise_manifests/prepare.json).
Sustained crossings used three consecutive 1,024-example validations spaced 100
updates apart; the third checkpoint is saved.

Recovery retains optimizer moments and EMAs from the perturbation endpoint. The
main-only control uses the fitted auxiliary head and receives 512 uninterrupted
main updates, with the same recorded boundary at 256. Bias is mean signed error;
tail error is the 99th percentile absolute error across example/query predictions.
Copy bits lost from the start are starting bits minus current bits; the matched
control penalty is main-only bits minus intervention bits, positive when worse.
Both raw and EMA clean auxiliary errors are recorded.

Probes use only the prepared starting checkpoints, not trajectory endpoints. Learned
conditions use the fitted auxiliary head. Inherited Adam retains every original
parameter's moments and step counter, including those of the previously unused
auxiliary head. Coasting uses the zeroed auxiliary-loss gradient: the shared trunk
and auxiliary head have explicit zero gradients, and the main head is unused.
For displacement vectors d over batches, mean displacement RMS is RMS(mean(d));
variance is mean over outputs of unbiased batch variance. Alignment is
mean_outputs(2 * starting_error * mean(d)). The saved vectors permit uncertainty
analysis beyond these summaries.

Joint training starts anew from the prepared 6-bit checkpoint with fresh Adam and
EMAs initialized to that model, not from recovered weights. Validation is every
100 updates plus powers-of-four landmarks and stage endpoints. The 4,096- and
16,384-example perturbation budgets and 65,536- and 131,072-example joint budgets
have exact common measurement points across all batches. Recovery can also be
compared at equal examples since its boundary, while retaining its different
preceding perturbation exposures. Final joint test uses the independent 8,192
examples. Frozen probes are included in bundle allocations: 30 GPU-hours is the
base allocation maximum, or 32 including the bounded retry allowance.
