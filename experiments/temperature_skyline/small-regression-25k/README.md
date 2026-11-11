<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Small all-fields regression control: 25,000-update protocol

One fresh all-fields regression run has been submitted as job **22668910**, with completion callback 22668911. It passed first-step logging and reached 2,720 updates at the first check, with rolling GPU utilization 57.1%. The four-update qualification passed; no production endpoint or new map result is reported here. [Submission receipt](submission-receipt.json). This is an experiment record, not intended to merge.

This control matches the 17×17 input crop and width-512, four-block backbone of the [completed soft-prior classifiers](../soft-priors-25k/README.md). The earlier temperature-only/all-fields regressors had approximately 208 million parameters and 33×33 crops. Those larger models remain historical benchmarks; they cannot isolate the effect of categorical output in the smaller model. This run supplies the missing small all-fields regression comparison.

| Property | New all-fields regression | Completed all-fields classifier |
|---|---|---|
| Backbone and input crop | Width 512, four blocks, 17×17 cells | Same |
| Total parameters | 14,321,665 | 18,523,648 |
| Output-head parameters | 513 | 4,202,496 |
| Training objective | Normalized mean squared error | Two-hot cross entropy |
| Point readout | Continuous scalar | Fixed center of argmax category |
| Head initialization | Independent scalar-head initialization | Gaussian smoothing across category rows, sigma four bins |
| Probability output | None | 8,192-category distribution |
| Primary endpoint | Final update-25,000 EMA, pending | Final update-25,000 EMA, complete |

The head, objective and head initialization change together. Parameter counts therefore differ even though the remaining model matches. This comparison does not isolate a parameter-count effect from an objective or readout effect. The completed temperature-only classifier will appear as an additional task comparison; its auxiliary outputs are unsupervised and will not be treated as forecasts.

## Fixed training and evaluation setup

The [production configuration](all-fields-regression-25k.json) uses source `ea592e8a71ebf9b50285ef835d641aae1de4d3c9`, the same source as the completed classifiers. All models take continuous OM4 quarter-degree fields from two times five days apart and predict five days after the latest input. Targets comprise temperature, salinity and both velocity components at 19 depths, plus surface height: 77 channels. OM4 is ocean-model output, not observational truth.

Regression uses the existing full-grid normalized MSE objective with equal channel weights across the 77 channels. Its target objective is the sum of squared normalized errors over wet channel-cells divided by 77×H×W, with land contributing zero. Every channel has coefficient 1/77; no extra temperature or auxiliary-field multiplier is applied. Sampling weights correct the nonuniform depth/location proposals. Values are normalized with the same per-channel mean and standard deviation. Vocabulary bounds and the `bins` setting remain in the copied configuration for provenance but are unused by the regression head/loss; `categorical_target: hard` does not turn regression into classification.

The run starts at update zero with fresh optimizer and EMA state and seed 20261110. The learning rate warms up for 100 updates to 2×10⁻⁴, then decays by cosine to 2×10⁻⁵ at update 25,000. Effective batch is 256: eight logical draws of 32, grouped into physical batches of 64. All three runs use seed 20261110 and the same deterministic NumPy sampling/frame-stream recipe. Coverage sampling, date reuse, prefetch, validation-frame caching and EMA settings match the classifiers. The intended accepted training draws therefore match; the final sampler/frame states will be checked against the classifier checkpoint, while the full per-example stream is not logged. A completed run will consume 6.4 million accepted crop examples and 25,000 EMA updates. EMA is the exponential moving average of parameters, with decay min(0.999, (1+n)/(10+n)) at update n.

The primary comparison uses final update-25,000 EMA, with raw and EMA curves retained from initialization onward. Any run stopped before that budget is reported as incomplete; intermediate comparisons use equal update counts. Validation uses the same four dates and 16 sampled points per depth per date, or 64 date-cells per channel. Normalized MSE retains the full rectangular-grid denominator with land contributing zero and the sampling correction retained. RMSE is the square root of the mean normalized-channel MSE; it is not a physical-unit RMSE or an average of depth RMSEs. Persistence copies the latest input onto the same targets.

The [CPU curve renderer](render_curves.py) will compare regression, all-fields classifier and temperature-only classifier, retaining all 77 raw/EMA channel records in CSV. It plots all 19 depths of each volumetric field, the single surface-height channel aggregated across sampled locations/dates, and matched persistence. Unsupervised temperature-only auxiliary channels are retained only in CSV and excluded from scientific plot comparisons. Categorical negative log likelihood (NLL) is available only for classifiers; regression has no categorical probabilities or NLL. A classifier's hard-label NLL does not establish probability calibration. Step zero remains in CSV; logarithmic update axes begin at one. The renderer rejects changed validation samples, normalization, backbone configuration or undocumented production-config differences.

## Qualification and resource bounds

Qualification job **22668484** completed four optimizer and EMA updates, accepting 1,024 examples at effective batch 256. All 77 channels at updates 0, 1, 2 and 4 were finite for raw and EMA weights; validation origins, counts and persistence matched the completed classifier. [Qualification receipt](qualification-receipt.json), [validation and provenance data](qualification-data.json.gz).

Actual saved GPU initialization tensors outside the output head matched the completed classifier exactly. The CPU comparison also matched those tensors within the CPU environment. Local CPU and container initialization hashes differ; no cross-environment identity is claimed. The recorded CPU non-head tensor hash is `febf6602102b9dcea88747e5a0937396efe4dfaf309971f6025ae4db4d555e4f`. The full initial-parameter digest matched the saved regression step-zero snapshot. These checks establish the intended initial backbone match, not identical later trajectories. The classifier performance qualification already showed numerical trajectory differences under execution changes, and this remains a single-seed comparison on sparse validation samples.

Qualification consumed **58 allocated GPU-seconds**. Production has an eight-GPU-hour total cap; planned matched maps have a separate one-GPU-hour cap. Job 22668910 has a 2 h 05 min scheduler reservation to allow backfill, while the total allocation cap and resume/requeue behavior remain unchanged. Earlier pending job 22668714 was canceled before allocation and consumed zero GPU-seconds. No engineering fix was required for that scheduling change. Recovery is limited to two engineering fixes and one infrastructure retry, with allocated failures charged. No additional training arm is authorized by this protocol.
