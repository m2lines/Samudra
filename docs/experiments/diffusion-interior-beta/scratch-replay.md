<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Fresh direct and residual models with replay

Status: campaign complete. The residual model reached 10,160 updates of each task and finished its scheduled cooldown; [final results](scratch-replay-final.md) and [error maps](scratch-replay-final-maps.md) are published. The direct arm was stopped at the user's request after persistent annual instability. The separate [4k observation-only cooldown](residual-4k-cooldown-results.md) did not replace the main trajectory. Historical launch notes below describe the original plan.

The question is whether training stable evolution from scratch makes annual error growth more comparable to the deterministic control, while retaining plausible individual diffusion members. Beating the deterministic control's absolute score is not the primary criterion.

| Arm | Processor | OM4 replay | Latent bound | Beta job |
| --- | --- | --- | --- | --- |
| Direct | Predict next latent directly | Yes | None | 212418 |
| Residual | Predict latent increment | Yes | Existing soft excess penalty, weight 0.1 | 212419 |

Both start with identical random tensors, seed 1729, with no pretrained checkpoint loaded. Each receives **10,160 OM4 and 10,160 observation updates**, 25% more than the preceding 8,128-per-task parent. The original quadratic mixed schedule is retained. These are two useful candidate systems; their comparison changes both residual prediction and the bound, so it does not isolate either effect.

## Training and checkpoints

- Effective batch eight on eight Beta GPUs, two hosts per arm. Each rank processes one logical example; gradients sum after division by eight. Globally unused parameters retain `grad=None`, preserving AdamW behavior when replay bypasses a module.
- Width-192 diffusion decoder, width-128 latent, depth-four processor, 45×90 latent grid and 180×360 output. BF16 autocast with FP32 parameters, compiled decoder, correlated noise 0.5, original spatial loss weight 0.5.
- Full 32-step Heun sampler and two-member observation CRPS. Larger decoder batches, channels-last, and the one-step surrogate are not used: the [profile](full-sampler-performance.md) did not establish their benefit for the full training objective.
- Original full-latitude observation masks, existing normalization, data splits, forcing and losses. Data fingerprints are checked at startup.
- Half the OM4 batch uses fresh starts; half uses four persistent replay chains with 30/90/180/300-day burn-in, six differentiable evolution steps, and at most four uses before refreshing. Replay is detached between updates and bounded by the existing maximum age.
- AdamW, learning rate `1e-4`, weight decay 0.01, gradient clipping at 1.0. Final 2,540 mixed updates cool to `1e-6` with a cosine schedule.
- Full restart checkpoint every 100 mixed updates, including optimizer, all rank RNGs, and every replay shard. Immutable evaluation snapshots at observation counts 2k/4k/6k/8k/final, plus mixed step 4k and cooldown start.

**The requested early checkpoint is 2,000 observation updates: mixed step 7,561, with 5,561 OM4 updates.** Training continues while evaluation runs. This schedule exposes observations gradually; 2k observations is not mixed step 2k.

The initial estimate is roughly five to six days per arm once running, and approximately a day to the early checkpoint, excluding queue time. This extrapolates the observation-only performance profile; native/replay timings during qualification and production will refine it. Each production allocation has eight GPUs and a seven-day limit; both are independently scheduled.

## Qualification and monitoring

Qualification job **212417** runs both arms for 16 mixed updates, restores the saved checkpoint, and continues each through update 32. Production jobs depend on successful qualification and a source-bound `QUALIFIED.json` receipt. Checks include continuous updates, finite module gradients, optimizer moments, all replay shards, exact exposure counts, and matching initial tensors. Qualification weights are never loaded into production.

Local durable services monitor the qualification and each production job through Slurm, with callback delivery on terminal state or the early checkpoint receipt. The completed qualification callback is retired; both production watchers remain active with two-hour fallback reminders. Online W&B is verified for [direct](https://wandb.ai/ocean_emulators/default/runs/diffscratch-diffusion-scratch-replay-v1-direct) and [residual](https://wandb.ai/ocean_emulators/default/runs/diffscratch-diffusion-scratch-replay-v1-residual). Each production run has eight process logs confirming cross-host `P2P/MNNVL` transport.

The new trainer and evaluator passed 21 local tests, including two-process gradient equivalence with rank-dependent unused parameters and optimizer/replay restoration. Ruff, mypy and shell syntax checks passed. GPU qualification completed successfully in 33m31s (4.47 allocated GPU-hours). Both arms completed 28 OM4 + 4 observation updates, including the actual restart at mixed step 16. The qualification receipt confirms identical initial tensors, complete replay/RNG state, finite module gradients and the expected source ID. All 32 process logs across the four invocations show `P2P/MNNVL` transport. Production checkpoints start fresh, independently of these smoke weights.

## Early evaluation

Direct saved `step-07561.pt` with exactly 5,561 OM4 + 2,000 observation updates on 2026-10-13. The checkpoint SHA256 matched its atomic receipt before submission of Beta evaluation **212865**. Outputs will be retained under `evaluation/direct-obs2000`. This job runs the three probes on one production Beta host, with a separate durable callback. Residual has not yet reached the same exposure.

For each immutable checkpoint, run the existing fixed nine-month validation cohort, native OM4 diagnostics, and annual observation rollouts. Retain eight members and 32 denoising steps, the same three January starts in 2015/2018/2021, and the established climatology-normalized four-component blended RMSE at 30/365 days.

Report individual components and error growth, ensemble mean and individual-member maps/spectra, and native leads 0/30/90/180/365 days. Compare both candidates with the existing matched deterministic and historical diffusion results, explicitly recording exposure differences. Restarted evaluations verify checkpoint hashes and payload receipts before reusing outputs.

## Reproduction and provenance

- Campaign: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/diffusion-scratch-replay-v1`.
- Immutable staged source: `code/scratch-replay-5c2a70055fcd`; archive read-back verified before extraction.
- `checkpoints/data-reference-contract.json` contains reference data fingerprints only for matching. No parent weights were staged.
- Observation samples: `/projects/ny/lz1955/multiscale/jrusak/data/obs_full_range/d-observation-pilot/samples`.
- Annual samples: same parent, `annual-instance-v1`.
- OM4: `/projects/ny/lz1955/multiscale/jrusak/data/diffusion-interior-beta/data/om4_onedeg_v3`.
- Entrypoints: [trainer](../../../src/samudra/experiments/diffusion_scratch_distributed.py), [evaluator](../../../src/samudra/experiments/diffusion_scratch_evaluate.py), [qualification](../../../scripts/qualify_diffusion_scratch_distributed.py), [Beta launcher](../../../scripts/slurm_diffusion_scratch_beta.sbatch).

The Beta launcher defaults to two production hosts with long QoS. Qualification overrides partition/QoS/time to `beta_test`/`test`/two hours. Evaluation overrides nodes/segment to one, QoS to standard, and time to 24 hours; it runs validation/native/annual on three GPUs of that full host. All distributed launches use `NCCL_MNNVL_ENABLE=1` and the verified routable `bond0` bootstrap interface. Transport is checked in runtime logs.
