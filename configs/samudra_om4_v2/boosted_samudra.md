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

    4. Adversarially select next mask via the configured MaskSearcher:
           candidates = searcher.candidates_for_round(t)   # which masks to score in step 2b
           searcher.update(scored)                         # learn from this round's scores
           M_{t+1} = searcher.select_next(scored)          # argmax over candidates
       The mask currently performing worst on the reweighted training
       distribution gets next round's K SGD steps. (See "Searching the
       path lattice" below for the v1 vs. v2 searcher choice.)

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

## Searching the path lattice

The `MaskSearcher` abstraction owns three responsibilities each round:
generating the candidate set to score (`candidates_for_round`), learning
from the scored results (`update`), and picking the next mask
(`select_next`). Two implementations ship today; both target the same
algorithmic role — only the candidate-generation strategy and learned
state differ.

### v1 — `EnumerateSearcher` (default; UNet skips only)

Pool size `2^num_skips = 16`. Every mask is scored every round; selection
is argmax over the round's scored set (or round-robin if so configured).
Stateless across rounds. This is the configuration in
`configs/samudra_om4_v2/boosted_pcgb.yaml` and the experiment we run first.

| Knob | Default | Rationale |
|---|---|---|
| `num_skips` | 4 | Matches `len(unet.ch_width)`. |
| `schedule` | `adversarial` | Argmax weighted MSE on `D_t`. |
| `no_repeat_window` | 0 | "Trust the algorithm" first; raise to 2–3 if mask selection collapses. |

### v2 — `MixtureSearcher` (future experiment; full path lattice)

Pool size `2^(num_skips + num_blocks) = 2^14 = 16,384` — too large to
enumerate. Each round samples `num_candidates ≈ 256` masks from a
three-way mixture and fits a per-bit *importance* surrogate online:

| Mixture branch | Default share | What it does |
|---|---|---|
| **Structured prior** | 60 % | Per-bit Bernoulli with `p_b = σ(α · w_b)`, where `w_b` is the running per-bit importance. Concentrates candidates on bits that, when set, predict high weighted MSE — i.e. plausibly the next argmax. |
| **Stratified by Hamming weight** | 25 % | Pick `k ∈ {1, …, d-1}` uniformly, then sample a mask with exactly `k` drops. Hedges against the linear surrogate underfitting bit-interaction effects. |
| **Uniform** | 15 % | Per-bit Bernoulli(0.5). Exploration tail; prevents the prior from collapsing onto a narrow region of the lattice. |

The surrogate is a linear regression `S(M) ≈ Σ_b β_b · M[b]` fit each
round to the round's `(mask, weighted-MSE)` pairs, then EMA-blended into
the running estimate `w_b ← γ · w_b + (1-γ) · β_b`. Linear-model bias is
the price of cheapness; the stratified branch hedges against missed
bit-interactions. The closed loop is: dropping load-bearing blocks
raises `S(M)` → those bits get higher `w_b` → the next round's prior
samples them more often → SGD trains θ to compensate by recruiting other
blocks. **Veit's wasted-capacity argument made operational.**

| Knob | Default | Notes |
|---|---|---|
| `num_blocks` | 10 | 4 down + 1 middle + 4 up + 1 final ConvNeXt blocks. Address conv-block residual drops. |
| `num_candidates` | 256 | Forward-pass budget per scoring round. |
| `surrogate_decay` | 0.7 | EMA retention for `w_b` (≈3-round effective window). |
| `prior_temperature` | 1.0 | Greediness of the structured-prior Bernoulli. |
| `seed` | 0 | Per-rank RNG seed for the mixture sampler. |

v2 also enables i.i.d. stochastic depth on conv-block residuals at
training time (`core_block.residual_drop_rate=0.1`) so the network's
training distribution covers the deterministic-mask regime the searcher
exercises at scoring time.

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
| Mask searcher | `enumerate` (v1) | See "Searching the path lattice" — v1 enumerates 16 skip masks; v2 swaps in `mixture` for the full 14-bit lattice. |
| Rounds T | 70 | One round per baseline epoch — total SGD compute exactly matches the 70-epoch reference run. |
| Steps per round K | 1 epoch of full train | Fresh `D_t` between rounds (avoids the staleness of K=2 epochs). Total T·K = 70 epochs of SGD. |
| Scoring subset | 25% of the calibration set | Bounds the per-round forward-only cost to keep ≥80% GPU util. |
| Calibration set | `data_percent: 0.2` of train period | For the unmasked-residual reweighting target. |
| Reweighting | AdaBoost.R2-proper, adaptive β_t (clipped) | No manual coefficient; matches `DynamicLoss` precedent in this repo. |
| `reweight_beta_max` | 1.0 | Caps β to keep the reweighting *direction* correct when L̄ > 0.5 (raw β > 1 inverts AdaBoost.R2 and up-weights easy examples). Sign-safe replacement for the canonical abort-on-failure rule. |
| `reweight_enabled` | `true` (set to `false` in the no-reweight ablation) | When false, D_t stays at 1 forever — collapses to plain mean MSE so we can isolate the boosting signal from deterministic mask cycling. |
| EMA smoothing on D_t | λ = 0.3 (≈ effective window of 3 rounds) | Borrowed from `DynamicLoss.N_WINDOW`; avoids per-round whipsaw. |
| `D_t` clamp | `max(D)/min(D) ≤ 5` | Tightened from `DynamicLoss._limit=20` to bound the "Arctic-2018 collapse" risk where a few spatially-extreme samples dominate D_t. A `WARNING` fires if `H(D)/log2(N) < 0.8`. |
| Optimizer | Adam, lr cosine 6e-4 → 0 | Matches baseline schedule across T·K total steps. |
| Per-round checkpoint cadence | every 2 rounds | Space is fine; minimize utilization dips. |

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
# v1 PCGB cold-start training (multi-GPU): enumerate-skips searcher
torchrun --nproc_per_node=8 -m ocean_emulators.pcgb \
  configs/samudra_om4_v2/boosted_pcgb.yaml

# v2 PCGB (future experiment): mixture searcher over the full path lattice
# torchrun --nproc_per_node=8 -m ocean_emulators.pcgb \
#   configs/samudra_om4_v2/boosted_pcgb_v2.yaml

# Eval the PCGB-trained checkpoint
python -m ocean_emulators.eval configs/samudra_om4_v2/boosted_eval.yaml \
  --ckpt_path=${OUTPUT_BASE}/${EXPERIMENT_NAME}/saved_nets/pcgb_final.pt
```

On HPC, use `scripts/launch_boosted_samudra.sh` which submits the
multi-node/multi-GPU train job and queues the eval behind it via SLURM
dependency. Override `CONFIG=configs/samudra_om4_v2/boosted_pcgb_v2.yaml`
to launch the v2 experiment.

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

## Follow-up experiments

The first pass launches two runs concurrently:
- **V1 / E1 warm-start** — `boosted_pcgb_e1.yaml`, `EnumerateSearcher`,
  16-mask skip lattice, finetuned from the dense+dilated E1 ckpt. This
  is the "does PCGB extract slack on top of an already-strong arch"
  test.
- **V2 cold-start** — `boosted_pcgb_v2.yaml`, `MixtureSearcher` over the
  full 14-bit lattice. This is the "does PCGB beat plain SGD from
  scratch" test.

The follow-ups below assume that pair has finished. They are ordered by
cost — Tier 1 is config-only, Tier 2 is new training runs, Tier 3 is
post-hoc analysis.

### Tier 1 — config-only, runnable next

**A. No-reweight ablation.** Config exists at
`configs/samudra_om4_v2/boosted_pcgb_no_reweight.yaml` (sets
`reweight_enabled: false`). Holds `D_t = 1` forever, so step 1 collapses
to plain mean MSE; PCGB reduces to "deterministic mask cycling + SGD"
with the AdaBoost.R2 machinery off. Isolates the *reweighting*
contribution from the *mask-cycling* contribution. If A ≈ V1 the
boosting machinery is dead weight; if V1 > A the R2 reweighting is
doing real work.

**B. Round-robin schedule.** Set `mask_searcher.schedule: round_robin`
in the V1 config. Walks the 16-mask pool in order rather than picking
the argmax-weighted-MSE mask. Isolates "adversarial selection matters"
from "coverage of the lattice matters." If B ≈ V1 the lattice is
uniform enough that selection order doesn't matter; if V1 > B the
argmax is finding genuinely informative weak learners.

**I. Bit-3 lesion run.** The pre-launch 16-mask diagnostic on the E1
ckpt produced a *bimodal* 188.8% spread — 8 bit-3-dropped masks
clustered at MSE ≈ 0.52; 8 bit-3-kept clustered at MSE ≈ 0.025. Either
(a) bit-3 (the innermost/deepest U-Net skip) is carrying nearly all
the bottleneck-bypass capacity, or (b) dropping it breaks the network
architecturally (resolution mismatch through the bottleneck). Two short
fine-tunes — one with `M = drop-bit-3-only`, one with `M =
drop-all-except-bit-3` — would disambiguate and tell us whether the
adversarial selector is doing meaningful work or just chasing one load-
bearing skip every round. Drives the `no_repeat_window` default and
informs whether to even include bit-3 in the search pool.

### Tier 2 — new training runs

**E. Half-degree PCGB.** Same algorithm on the 0.5° dataset
(`om4_halfdeg_v4`). Tests whether the boosting signal scales with data
resolution: at 0.5° each batch carries ~4× more spatial information, so
the path lattice may have less slack to recover. Roughly 4× the
per-round wallclock; fits in a 24 h slot at 8 GPUs.

**F. Depth-banded adaptive reweighting.** Extend AdaBoost.R2 from
per-sample to per-(sample, depth-band). The depth axis has known
imbalance — surface variables dominate gradient norms relative to deep-
ocean variables. Reweighting along depth applies the same boosting
recipe to Veit's wasted-capacity claim along the *depth* axis instead
of the sample axis. The `DynamicLoss` precedent in `utils/loss.py` does
the analogous per-channel adaptation and is a copy-target for the
implementation.

### Tier 3 — post-hoc analysis

**G. Per-mask depth-banded MSE on the final PCGB ckpt.** Run the
16-mask forward scan over the validation set, breaking per-mask MSE out
by depth band. The hypothesis PCGB is selling is "level the depth
distribution of error"; this is the most direct check, and it's free
once a PCGB ckpt exists.

**H. Rollout head-to-heads vs. Samudra-2 paper baseline.** Invoke
`scripts/compare_to_paper.py` against the paper baseline checkpoints on
S3. Settles whether PCGB beats the reference on multi-step rollout
RMSE — the actual deployed metric — rather than just on the train/val
MSE we monitor during PCGB rounds.

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
