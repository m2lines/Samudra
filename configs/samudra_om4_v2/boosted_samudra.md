<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Boosted Samudra v2 — Path-Cyclic Gradient Boosting (PCGB)

Author: Alexander S. Merose, Claude

Date: 2026-05-05.

A research experiment training Samudra v2 with **gradient-boosting principles
applied to the implicit ensemble already inside the network**. The output is a
single trained model whose forward pass is a standard ResNet/UNet — no
inference-time ensembling.

## Problem formation: Why is this experiment relevant?

In climate science, and likely AI 4 Science in general, we are in a weird circumstance
where our training data is large is bytes, but small in instances. In the context
of Samudra, our datasets can range from terabytes to petabytes in size, yet only cover
a relatively small number of overall samples of the state of the Earth (say, as measured
as points in time). When trying to use the right tool for the job, we find ourselves in
an awkward situation:

- When data is large in bytes, deep neural networks are the ideal tool.
- When working with a small number of samples, boosting based methods often excell where others fail.
  - xgboost in particular has been a huge success in solving problems involving tabular data or similar.
  - However, the literature on GBM nearly always involves trees which are ill fit when each example is large.

This experiment attempts to apply the meta algorithm of [gradient boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
to the workhorse that is [residual neural networks](https://en.wikipedia.org/wiki/Residual_neural_network).
The hope is that we can combine the strengths of both methods to create a new recipe for training
ML models within scientific domains.

## Why this should work in theory

The premise rests on three observations from the literature plus one
neural-network-specific consequence.

**1. ResNets are ensembles of shallow paths, indexed by skip patterns.**
Veit, Wilber & Belongie (2016, [arXiv:1605.06431](https://ar5iv.labs.arxiv.org/html/1605.06431))
show that a depth-`n` residual network admits an "unraveled view" containing
`2^n` implicit paths, indexed by which residual modules are traversed vs.
skipped. Their lesion experiments show that paths behave with ensemble-like
characteristics: errors degrade smoothly under module deletion or reordering,
unlike VGG-style chained networks. **For a U-Net, the natural path-indexing
knob is the set of skip connections** — toggling skip `i` changes whether
multi-scale information flows around the bottleneck or only through it,
producing genuinely distinct functions of the input. Our backbone has 4 skips,
so 16 distinct skip-masks are available.

**2. Most paths in a trained ResNet contribute almost no gradient.**
Veit et al. measure the gradient magnitude as a function of path length and
find it decays exponentially. In a 54-block network, *less than 1%* of all
paths receive 95% of the gradient; the remainder is effectively wasted
capacity. This is wasted capacity by accident, not by design — there is no
mechanism in standard SGD that *forces* the model to recruit longer or
different paths.

**3. The theory of gradient boosting provides useful affordances for deep learning**
I recommend Wikipedia's entry on [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting),
especially their informal introduction. Boosting is a meta algorithm lets one
turn "weak" learners into a strong learner via an ensemble. One intuition for how boosting works:
it lets you use previous "weak" learners to figure out the hardest examples in the training
dataset, and then trains a new, better learner adept at addressing that new class of "hard" examples.

The algorithm works iteratively: After training an initial model `F_m(x)`, one calculates the _residual_
of its prediction `R = y - F_m(x)`. The next iteration produces a better ensemble of learners
`F_{m+1}(x) = F_m(x) + h(x)` where `h_m(x)` is an approximation of the residual `R`. You can repeat this process
`M` times until you have a strong ensemble of "weak" learners.

Gradient boosting generalizing boosting to use any arbitrary loss function. This works because `h_m(x)` for a
given model are proportional to the negative gradients of the MSE loss function with respect to `F(x)`. Because
of this, gradient boosting can be generalized to near any gradient descent algorithm.

This present an opportunity to combine with training ResNets: A residual block in a ResNet is the same
functional form as a gradient-boosting round.

**4. Neural-network base learners turn boosting into a *training-time*
intervention, not an inference-time combiner.** Classical gradient boosting
expresses the prediction as `F_T(x) = Σ_t γ_t · h_t(x; θ_t)` — `T` separately
parameterized learners summed at inference. That formulation is forced by tree
base learners, which are *not* closed under composition with an arbitrary
loss `L`. Neural networks *are* closed under that: SGD on
`min_θ Σ_i L(y_i, F(x_i; θ))` directly minimizes the loss without a
pseudo-residual / line-search detour.

This matters because Veit's path decomposition tells us that a ResNet's
prediction *already is* an additive ensemble: `F(x) = Σ_p y_p(x)` over the
`2^n` implicit paths. The "stitch" between rounds-of-boosting and
paths-of-an-ensemble is the residual addition `y_i + f_i(y_i)` itself —
literally a `+`. If the ensemble already exists in the network's forward
pass, we don't need a second outer summation around it. We need a *training
algorithm* that uses gradient-boosting principles — example reweighting and
mask-cycling — to shape how SGD allocates capacity across the existing path
ensemble.

**Synthesis.** A baseline Samudra v2 trained with [stochastic skip-drop](https://github.com/Open-Athena/Ocean_Emulator/blob/0dbc093ad4d11adcd37bcb5c76d342e1c062dc76/src/ocean_emulators/models/modules/blocks.py#L108)
([reference run](https://wandb.ai/ocean_emulators/default/runs/9kfd76fz),
drop_path_rate=0.5 during the first 55 of 70 epochs) gives empirical evidence
that the path lattice is alive — every one of the 16 skip-masks saw training
signal — but standard SGD nevertheless concentrates gradient flow on the
dominant length-5–17 paths and leaves the rest under-utilized. PCGB is a
training intervention that pushes back on that concentration: at each round
we (i) **adversarially select** the skip-mask currently performing worst on
the reweighted training distribution `D_t`, (ii) run K SGD steps under that
mask with per-sample loss weights from `D_t`, and (iii) update `D_{t+1}`
based on residuals of the *unmasked* network. The scientifically clean test
is **cold-start**: train a fresh `θ` with PCGB and compare against a
baseline-trained `θ` of the same architecture and matched total compute. The
output is a single trained network whose forward pass is the standard
ResNet/UNet — no inference-time ensembling, no extra parameters, no T× test-
time cost.

## Path-Cyclic Gradient Boosting (PCGB)

```text
Initialize θ from random init (cold start).
D_1(i) = 1 for all training samples i  (uniform example weights, mean = 1).
M_1 = the all-keep mask (round 1 is plain weighted SGD; D_1 is uniform anyway).

For t = 1..T (boosting rounds):
    1. For k = 1..K (SGD steps under M_t):
        Sample a batch (x_b, y_b, idx_b) from the training set.
        With backbone.with_skip_mask(M_t):
            ŷ_b = model(x_b)
        per_sample_loss[i] = ‖ŷ_b[i] - y_b[i]‖²   (label-mask weighted)
        loss = mean( D_t[idx_b] · per_sample_loss )
        loss.backward(); optimizer.step()

    2. End-of-round streaming pass over the calibration scoring subset
       (~25% of the calibration set; one streaming pass produces both
       per-sample residuals and per-mask scores):
       a. Per-sample residuals under the *unmasked* network — this is
          what we ultimately ship, so reweighting tracks deployed error:
              r(i) = ‖y_i - F(x_i; θ_now)‖
       b. Per-mask weighted MSE under D_t:
              S(M) = Σ_i D_t(i) · ‖y_i - F(x_i; θ_now, mask=M)‖²

    3. Update example weights via AdaBoost.R2-proper (Drucker, 1997):
           L_i  ← |r(i)| / max_j |r(j)|              # range-normalize ∈ [0, 1]
           L̄    = Σ_i D_t(i) · L_i                    # weighted mean loss
           β_t  = L̄ / (1 - L̄)                         # adaptive confidence
           D'(i) ∝ D_t(i) · β_t^(1 - L_i)             # easy ↓, hard ↔
           D_{t+1} ← (1 - λ) · D_t + λ · D'           # EMA smooth (λ ≈ 0.3)
           D_{t+1} ← clamp(D_{t+1}, max/min ≤ limit)  # prevent collapse
           D_{t+1} ← D_{t+1} · n / Σ D_{t+1}          # renormalize, mean = 1
       (No abort-on-failure rule. EMA + limit clamp are lifted from the
       precedent in src/ocean_emulators/utils/loss.py:DynamicLoss, which
       does the analogous adaptive reweighting along the channel axis.)

    4. Adversarially select next mask:
           M_{t+1} = argmax_M S(M)
       The mask currently performing worst on the reweighted training
       distribution gets next round's K SGD steps. This is the boosting-
       theoretic choice (analog of "weak learner with highest weighted
       error" — the one with the most slack to recover).

    5. Every 2 rounds, save a per-round checkpoint (diagnostics).

Save the final θ.

Inference: standard ResNet/UNet forward pass on θ. Single model. Single
forward per step. No mask, no ensemble combination.
```

<details>

<summary> Why the adversarial scan is amortized, not a 16× tax.</summary>

The end-of-round streaming pass over the scoring subset (step 2) computes both step 2a and
step 2b in one sweep — for each batch, we run 16 forwards (one per mask) plus
1 unmasked forward, accumulating both per-sample residuals and per-mask
scores. There is no separate "mask selection pass" before round t+1; the
information needed to pick `M_{t+1}` is already produced by the round-t
post-amble. Net cost: one scoring-subset pass per round, ~17× a single
forward over that subset (16 masked + 1 unmasked).

</details>

### Why each step matters

- **Per-sample loss weighting (step 1)** is AdaBoost's reweighting mechanism
  imported wholesale. We use loss multipliers rather than weighted resampling
  because the former is more stable for deep nets (avoids over-fitting to a
  small set of "hardest" examples).
- **Residuals under the unmasked network (step 2a)** aligns the boosting
  target with inference-time deployment. We reweight examples by what `θ`
  is failing on *as it would be used at inference*, not what it fails on
  under the most-recent mask.
- **Reweighting (step 3)** mirrors AdaBoost.R2's regression-boosting update
  (Drucker, 1997) — exponential in normalized residual, normalized so the
  weighted training distribution stays a probability measure with mean 1.
- **Adversarial mask selection (step 4)** is the lever that addresses Veit's
  wasted-capacity finding. Each round directs SGD into the sub-path with
  the highest weighted MSE on `D_t` — the path with the most slack to
  recover. This is the boosting-theoretic analog of "pick the weak learner
  with the highest weighted error and improve it." Cost is amortized into
  the same scoring pass as step 2; selecting the next mask is essentially
  free given that pass already runs.

### What's preserved from the boosting-theory canon, and what isn't

| | Status under PCGB |
|---|---|
| Gradient-boosting *recipe* (residual fit + adversarial reweighting) | preserved |
| Per-round weak learner with its own `θ_t` and `γ_t` | abandoned (single shared `θ`) |
| Freund-Schapire training-error bound | does not apply (shared `θ` breaks weak-learner independence) |
| Single-model deployment / no T× inference cost | gained vs. inference-time ensembling |
| Training-time capacity reallocation across the path lattice | gained vs. plain SGD |

This is "boosting-flavored curriculum learning" with a strong inductive bias
toward the path-ensemble structure. The theoretical guarantee is sacrificed
for the practical win of a single deployable model.

## Experiment

Two from-scratch training runs of the same Samudra v2 architecture on
matched compute, comparing standard SGD vs. PCGB:

| Setup | Baseline | PCGB |
| --- | --- | --- |
| Initial `θ` | random init | random init (same seed) |
| Training | 70 epochs, plain SGD with stochastic depth | T rounds × K steps each, adversarial mask selection + reweighting |
| Total SGD-equivalent compute | matched | matched |
| Inference | standard | standard (no masking) |
| Forward-passes per inference step | 1 | 1 |

The baseline reference run already exists
([wandb](https://wandb.ai/ocean_emulators/default/runs/9kfd76fz)) and serves
as the comparison. PCGB needs one new from-scratch training run.

Falsifiability: PCGB-final val/test MSE meaningfully *below* baseline =
positive signal. PCGB ≈ baseline = the path lattice is too well-utilized
already, no slack for boosting-flavored capacity reallocation to recover.

### Eval

Two runs, both single-model rollouts at matched test-time compute:

| Predictor | Where the checkpoint comes from |
| --- | --- |
| Baseline | `/scratch/.../samudra_om4_v2_drop_path_new_data/saved_nets/ckpt.pt` |
| PCGB | `${output_dir}/saved_nets/pcgb_final.pt` |

`scripts/launch_boosted_samudra.sh` chains: PCGB train → eval(PCGB-ckpt)
via SLURM `--dependency=afterok`. Baseline eval is a separate, optional
invocation against the original ckpt.

## Hyperparameters

Defaults are starting points and will be revisited after the first design pass.

| Knob | Default | Rationale |
| --- | --- | --- |
| Mask pool | enumerate all 2⁴ = 16 skip-masks | small enough to enumerate; covers the path lattice |
| Mask schedule | adversarial (argmax weighted MSE on D_t) | boosting-theoretic; cost amortized into the end-of-round scoring pass |
| Rounds T | 32 | each mask sees 2 expected rounds; enough for D_t to drift |
| Steps per round K | 2 epochs of full train | matches baseline compute when T·K = 64 epochs ≈ baseline 70 |
| Scoring subset | 25% of the calibration set | bounds the per-round forward-only cost to keep ≥80% GPU util |
| Calibration set | `data_percent: 0.2` of train period | for the unmasked-residual reweighting target |
| Reweighting | AdaBoost.R2-proper, adaptive β_t | no manual coefficient; matches `DynamicLoss` precedent in this repo |
| EMA smoothing on D_t | λ = 0.3 (≈ effective window of 3 rounds) | borrowed from `DynamicLoss.N_WINDOW`; avoids per-round whipsaw |
| `D_t` clamp | `max(D)/min(D) ≤ 20` | borrowed from `DynamicLoss._limit`; prevents distribution collapse |
| Optimizer | Adam, lr cosine 6e-4 → 0 | matches baseline schedule across T·K total steps |
| Per-round checkpoint cadence | every 2 rounds | space is fine; minimize utilization dips |

## Compute budget

Cold-start training, multi-GPU DDP (8 × rtx6000 baseline).

- Per round (8 GPUs): K=2 epochs × ~1850 batches/epoch ÷ 8 ≈ **460 SGD
  batches/GPU** + adversarial-scoring pass ≈ 105 forward-only batches/GPU
  ≈ **565 batches/GPU/round**.
- T=32 rounds × 565 sec/round ≈ **~5 h on 8 × rtx6000** at ~1 sec/batch.
- Eval rollout (single GPU): ~15 min.
- Per-round SGD time / total round time ≈ 460/565 ≈ **81% effective GPU
  utilization** (above the 50% target with margin).

Recommended HPC config: **8 GPUs, 12 hours**. Fits in a 24 h slot with
warmup, I/O variance, and the eval chained behind it.

Single-GPU PCGB cold-start would be ~40 h — does not fit. DDP is required.

## How to run

```bash
# PCGB cold-start training (multi-GPU)
torchrun --nproc_per_node=8 -m ocean_emulators.pcgb \
  configs/samudra_om4_v2/boosted_pcgb.yaml

# Eval the PCGB-trained checkpoint
python -m ocean_emulators.eval configs/samudra_om4_v2/boosted_eval.yaml \
  --ckpt_path=${OUTPUT_BASE}/${EXPERIMENT_NAME}/saved_nets/pcgb_final.pt
```

On HPC, use `scripts/launch_boosted_samudra.sh` which submits the
multi-node/multi-GPU train job and queues the eval behind it via SLURM
dependency.

## Outputs

```text
${base_output_dir}/${experiment.name}/
├── saved_nets/
│   ├── pcgb_final.pt          # Final θ. Drop-in replacement for any Samudra ckpt.
│   └── pcgb_round_<t>.pt      # Per-round checkpoints (diagnostics).
├── round_metrics.json         # Per-round: mask, mean weighted loss, residual stats, D entropy
└── eval/                      # Rollout zarrs and aggregator metrics
```

`pcgb_final.pt` has the same state-dict layout as a regular Samudra
checkpoint, so any consumer (`eval.py`, `viz`, etc.) loads it without changes.

## References

- Veit, Wilber, Belongie. *Residual Networks Behave Like Ensembles of
  Relatively Shallow Networks*. NeurIPS 2016.
  [arXiv:1605.06431](https://ar5iv.labs.arxiv.org/html/1605.06431)
- Freund, Schapire. *A Short Introduction to Boosting*. JJSAI 1999.
  [PDF](https://cseweb.ucsd.edu/~yfreund/papers/IntroToBoosting.pdf)
- Friedman. *Greedy Function Approximation: A Gradient Boosting Machine*.
  Annals of Statistics 2001.
- Drucker. *Improving Regressors using Boosting Techniques*. ICML 1997
  (AdaBoost.R2).
- Huang, Sun, Liu, Sedra, Weinberger. *Deep Networks with Stochastic Depth*.
  ECCV 2016. [arXiv:1603.09382](https://arxiv.org/abs/1603.09382)
- Huang, Liu, Lanckriet, Belkin. *Learning Deep ResNet Blocks Sequentially
  using Boosting Theory*. NeurIPS 2018.
  [arXiv:1706.04964](https://ar5iv.labs.arxiv.org/html/1706.04964)
