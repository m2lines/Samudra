<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Fresh 25,000-update soft-prior protocol

Two fresh runs have been released and submitted: temperature-only job **22617264** and all-fields job **22617266**. Both passed first-step logging and reached roughly 2,500 updates at the first check, with rolling GPU utilization 61.5% and 58.3%; no final 25,000-update results or new forecast maps are reported here. CPU callback job 22617267 tracks completion. This is an experiment record, not intended to merge. [Submission receipt](submission-receipt.json).

Both runs start from random initialization at update zero. They do not continue the earlier 2,000-update checkpoints. Each combines two-hot targets with smooth head initialization, the “both” arm of the [earlier pilot](../soft-priors/README.md). Two-hot targets divide probability between the two category centers bracketing the target. Smooth initialization correlates neighboring head rows with a Gaussian of sigma four bin indices; subsequent training leaves every row separately trainable.

The [temperature-only config](temperature-both-25k.json) and [all-fields config](all-fields-both-25k.json) retain the 17×17 quarter-degree input crop, four-block width-512 backbone, 18,523,648 parameters, five field queries, and 8,192 centers equally spaced in the asinh coordinate for each channel. Inputs are continuous ocean states at five days before initialization and at initialization; the target is five days later. The five fields are temperature, salinity, eastward velocity, northward velocity (19 depths each), and surface height (one channel), totaling 77 channels. Each field query requests one scalar for the sampled depth; surface height is supervised only at the surface. Values use per-channel dataset mean and standard deviation. The vocabulary centers are uniformly spaced after the inverse-hyperbolic-sine transform; supports come from 32 training-only dates with a 10% extrema margin and are recorded in the configs. Each point prediction is the fixed center of its argmax category, without a continuous mean or offset. Temperature-only training zeros the other fields' losses while retaining temperature's all-fields coefficient.

Both configs use seed 20261110 and fresh optimizer/EMA state. The learning rate warms up for 100 updates to 2×10⁻⁴, then follows a cosine schedule to 2×10⁻⁵ at update 25,000. Each arm stops at 25,000 updates or eight total allocated GPU-hours, whichever comes first. The replacement jobs have 28,792 s (temperature-only) and 28,788 s (all-fields) remaining after charged startup failures. The intended full budget is 6.4 million accepted crop examples per arm. EMA denotes the exponential moving average of model parameters; its decay is 0.999 with update-n decay min(0.999, (1+n)/(10+n)). Final update-25,000 EMA is the primary endpoint, with intermediate raw and EMA scores retained. Any arm stopped by the allocation cap is reported as incomplete and excluded from that endpoint; intermediate comparisons use matched update counts.

The accepted stream still consists of eight logical batches of 32 examples per optimizer update, for effective batch 256. The selected implementation groups each pair into a physical GPU batch of 64, prefetches batches, and caches validation frames. Coverage sampling, date reuse, examples per update, temperature loss coefficient and optimizer schedule match between the two new arms. Auxiliary loss coefficients differ by design. Both use the same optimized implementation. Each logical draw selects 32 examples from an 8,192-candidate pool: a half-base, half-geometric-coverage proposal uses eight input-feature groups per depth, with exact inverse-proposal correction. A frame supplies 2,048 accepted examples (eight optimizer updates) before advancing; the next frame is prefetched. See the [sampling implementation](../../../src/samudra/skyline/coverage.py) for the base distribution and correction.

## Performance qualification and its limit

Qualification used source `ea592e8a71ebf9b50285ef835d641aae1de4d3c9`, archive SHA-256 `c9a0e0ffa7aaa3dac98286b1fc5438e0ffaa825f842bbb4e2a423a6434a0d0c3`, on an NVIDIA RTX PRO 6000 Blackwell Server Edition. The submitted runs use that same source archive. The 100-update screen measured median update times of 0.24600 s for physical batch 32, 0.25139 s for batch 128, and 0.20222 s for prefetched batch 64. The selected batch-64 path then underwent a paired 2,000-update all-fields qualification in job 22615178.

| Measurement | Baseline batch 32 | Selected batch 64 |
|---|---:|---:|
| Median optimizer-update time (s) | 0.23598 | 0.20455 |
| Full training-loop elapsed time (s) | 587.34 | 449.35 |
| Mean GPU utilization after 120 s (%) | 44.19 | 60.04 |
| Final EMA temperature normalized MSE | 0.00489063 | 0.01431469 |
| Final EMA temperature MSE at 2,400 m | 0.00237986 | 0.17962154 |
| Final EMA all-channel hard-label NLL | 3.03538 | 3.03310 |

Elapsed time includes validation and data waits. These are one paired measurement on this hardware; they do not establish a universal speedup. Profiling records BF16 attention query/key/value projections and head computation, followed by FP32 logits and loss. In the selected 100-update probe, frame-advance waits totaled 0.819 s of 22.98 s wall time (3.6%). CPU crop extraction took 6.65 s; coverage sampling including extraction took 16.90 s. These overlap GPU work and must not be added as wall-time fractions. Current-frame reuse and next-frame prefetch already reduce disk waits. The attention kernels are fused CUTLASS memory-efficient scaled-dot-product attention, not FlashAttention. [Profile, training and validation data](qualification-run-data.json.gz).

The [qualification receipt](qualification-receipt.json) records identical initial parameter hashes, source/configuration apart from the three performance options, final NumPy/Torch/CUDA random-number-generator states and frame-stream state, and validation origins/counts/persistence. Both runs accepted 512,000 examples, reached 2,000 optimizer and EMA updates, and passed finite-loss/gradient/validation checks. The 100-update profiles also have identical accepted-input hashes.

Post-update numerical equivalence is false. Temperature MSE rose from 0.00489 to 0.01431, with most of the aggregate difference coming from depth index 14 (2,400 m). The all-channel negative log likelihood (NLL) changed only from 3.03538 to 3.03310. The combined implementation changes produced a different training trajectory; equality of initial states and sampled data does not establish equal forecasts. Release required matching initialization/data/configuration and sampler end states, finite execution, and improved utilization; it did not require equal point accuracy. The accepted-input hash covers the 100-update probes; final-state equality alone does not prove every 2,000-update input tensor was identical. The 2,000-update qualification tested all-fields only, not temperature-only numerical behavior. Both production arms share the selected implementation.

These qualification scores use one seed and four validation dates, with 16 points per depth per date: 64 sampled date-cells per channel. Temperature MSE averages 19 normalized channels; NLL averages all 77 channels. Each channel loss treats land as zero but retains the full rectangular-grid denominator, implemented by wet-fraction scaling and inverse-proposal correction; the [evaluator](../../../src/samudra/skyline/comparison_eval.py) specifies the sums. Lower is better. Sparse validation can make rare deep-water errors dominate; the NLL result does not establish calibrated probabilities. The completed performance work consumed **1,240 GPU-seconds**, including the failed diagnostic attempt, profiling and paired qualification. Initial production jobs 22617080 and 22617082 then failed before model/data initialization because the configs had not been copied to the cluster, consuming another 8 and 12 GPU-seconds. The files were copied and their hashes verified before the replacement submissions above. The 20 s is charged to the respective eight-hour caps. Both permitted engineering fixes have been used; the earlier fix corrected a profiling dtype assertion. Completed charged work at this stage is 1,260 GPU-seconds, excluding the replacement runs and later maps. Qualification costs are outside the production caps. One bounded infrastructure retry remains; any new engineering blocker requires a decision before more training.

## Matched quantization check

An oracle assigns each known OM4 target its nearest physical category center. Its RMSE measures the vocabulary's best possible pointwise quantization error; it is not a learned forecast. The check uses the actual 8,192 float32 centers, transformed into physical units in float64, and compares with the cached original temperature-only regression model at EMA update 25,000. Oracle and regression use exactly the same wet cells and area weights.

All nine full native regional patches use origin 4273, targeting 2016-07-21 at five-day lead. These full cached regions are larger than the planned centered 64×64 display patches. Every target lies inside the finite center support. Regression RMSE is 9.3–331.3 times the oracle RMSE on these patches; finite bin spacing alone therefore does not impose the observed regression error here. This does not show that a trained classifier can reach the oracle or establish support coverage on other dates, regions or fields. [CSV](matched-oracle.csv), [source normalization and oracle receipt](matched-oracle.json.gz).

| Region | Depth (m) | Wet cells | Oracle area RMSE (°C) | Original T regression area RMSE (°C) | Regression / oracle |
|---|---:|---:|---:|---:|---:|
| North Pacific | 2.5 | 9,600 | 0.001489 | 0.493321 | 331.3 |
| North Pacific | 250 | 9,599 | 0.001024 | 0.079485 | 77.6 |
| North Pacific | 1850 | 9,581 | 0.000350 | 0.003270 | 9.3 |
| Gulf Stream | 2.5 | 6,400 | 0.001643 | 0.430268 | 261.9 |
| Gulf Stream | 250 | 6,273 | 0.001516 | 0.246048 | 162.4 |
| Gulf Stream | 1850 | 6,215 | 0.000548 | 0.008827 | 16.1 |
| Agulhas | 2.5 | 6,400 | 0.001299 | 0.275050 | 211.8 |
| Agulhas | 250 | 6,398 | 0.001115 | 0.269966 | 242.2 |
| Agulhas | 1850 | 6,263 | 0.000360 | 0.008822 | 24.5 |

## Planned matched maps

After training, [the CPU renderer](render_maps.py) will compare physical fields and errors against OM4 on centered 64×64 native patches in the North Pacific, Gulf Stream and Agulhas at 2.5, 250 and 1,850 m. Separate temperature figure sets compare the two new models with original temperature-only/all-fields regressors and original classifiers. Surface u/v panels cover Gulf Stream and Agulhas; salinity panels cover the Black Sea at 1,850 m for target dates 2014-06-22 and 2014-10-05 (origins 4121 and 4142). These selected validation outliers are separate from the held-out July 2016 maps. Auxiliary-field comparisons use supervised all-fields models and matched persistence only.

The renderer checks coordinates, masks and targets before joining caches, requires every wet prediction, and displays native pixels without interpolation. Input-context boxes show 17-cell new-model and 33-cell original-model crops at the same scale. Area-weighted physical RMSE and short-wave power use matched cells; spectra use the same fully wet 64×64 or 32×32 tile for each comparison, planar detrending and a periodic Hann window. Power at wavelengths of 2–4 grid cells is a scale diagnostic, not proof of noise. Cached Samudra temperature is available as an additional metric/spectral benchmark; no cached Samudra u/v or salinity is used.

Original models differ in architecture, task, context and training/checkpoint budgets. Their maps are benchmarks and do not isolate the effect of the new target or initialization. OM4 is an ocean-model target, not observational truth. No new 25,000-update map or scientific comparison is complete at this submission stage.

## Benchmark identities

Origin denotes the initialization record index in the quarter-degree dataset. Origin 4273 initializes on 2016-07-16 and targets 2016-07-21. Display patches are centered native 64×64 cells within North Pacific 170–200°E/25–45°N, Gulf Stream 300–320°E/25–45°N, and Agulhas 40–60°E/50–30°S.

| Benchmark | Parameters | Crop | Selected weights |
|---|---:|---:|---|
| Original T regression | 207,974,401 | 33×33 | Final 25,000-update EMA |
| Original all-fields regression | 207,974,401 | 33×33 | Final 25,000-update EMA |
| Original T classifier | 210,067,456 | 33×33 | Likelihood-selected EMA at 27,624 total updates; 2,048 bins |
| Original all-fields classifier | 55,907,328 | 33×33 | Final 15,000-update EMA; 2,048 bins |

The [original regression/noise report](../pre-spike-noise/README.md) identifies its cached runs. The original T classifier checkpoint SHA-256 is `50ad50aaed2bd3305adeb519c7bfad17b34e57f657b3f81a2d5bb039afd425d5`; its cached inference archive SHA-256 is `1452f66cf69306297e2780895f233aa4be38faa93f7e753a2eb7c90dfbea0b9a`. Cached Samudra uses the October 8, 2026 quarter-degree two-input/one-output run 211241, epoch-62 model weights (no EMA), with true two-state history; checkpoint SHA-256 `1f62b6303519d54a2976d7e13deac4a3311512b7a090db0f61e3e288d76ba841`. Its cached targets and Samudra temperatures match the classifier cache exactly on the nine display patches.
