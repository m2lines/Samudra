<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# All-fields model scaling

This experiment tests whether increasing the size of the all-fields, single-depth skyline model improves temperature forecasts under the same training recipe. Each crop queries temperature, salinity, and both velocity components at one sampled depth, plus SSH only when the sampled depth is the uppermost layer. We vary transformer depth while holding width at 1,024 and attention heads at eight. The completed 16-block run is the baseline.

| Blocks | Parameters | Relative to baseline | Run |
| --- | ---: | ---: | --- |
| 4 | 53,809,153 | 0.259× | New, fresh initialization |
| 8 | 105,197,569 | 0.506× | New, fresh initialization |
| 16 | 207,974,401 | 1× | Completed Torch job 22106889 |
| 32 | 413,528,065 | 1.988× | New, fresh initialization |

All new runs use the original training source commit `753edeac5faa46412957b70ce5e02d6337bdefbc`, container, and [all-fields configuration](../../../src/samudra/configs/temperature_skyline/comparison_all_fields.json), changing only `model.layers`. Each receives 25,000 optimizer updates and 6.4 million accepted crop draws (four GPUs × 32 crops × two accumulation steps). The rank-specific data-sampler seeds and frame-stream seeds are unchanged, so accepted date/depth/crop sequences match the baseline; different model shapes have different initial parameter tensors. Inputs are two states in a native quarter-degree 33×33 crop, with a five-day target. Coverage sampling, its importance correction, legacy full-grid per-channel MSE, AdamW, warmup and cosine schedule, EMA, and fixed validation cases remain as specified in the [output-task protocol](../output-task-comparison/README.md).

The primary comparison uses mean normalized temperature MSE across the 19 depth channels at update 25,000, using the legacy full-grid denominator on the same fixed validation dates and locations, reporting raw and EMA weights separately. We will show temperature error against parameter count, validation trajectories, and per-depth temperature RMSE. Lower errors are better. Other fields remain supervised and will be reported to expose tradeoffs. Each run has one seed. Updates and crop exposure are matched; FLOPs and runtime are not. The common learning rate is not tuned independently for each model size, so the result measures this fixed recipe's scaling behavior. The already-running whole-column experiment continues separately.

Before each long run can start, a shared qualification job runs all three new model sizes for four updates each and checks completion and bitwise agreement among DDP replicas for both raw and EMA parameters within each run. Submitted jobs use preemptible routing and restore optimizer, EMA, random-generator, and data-stream state from checkpoints. Total allocated runtime extrapolated linearly from the roughly 17.4-hour, four-GPU baseline totals about 48 summed wall hours across the three four-GPU runs, or 192 GPU-hours; fixed data/validation overhead and hardware differences make this a rough estimate. Qualification will provide timings for each size. Queue wait is additional.

Container: `ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-4f86be45a09cb9eba1b51b6273e0189941b02495.sif`. Source archive SHA-256: `be8ff1f2d47a91838fe3b8253b8c9a1be9f7abb9b7d2f71dff5a9e93ee936000`. These match the completed 16-block baseline.
