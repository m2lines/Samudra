<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# The discrete pilot's 1,850 m failures and training exposure

The dominant failures involve warm deep-water cases, and several models predict categories that received **zero training targets at 1,850 m**. However, every validation truth bin was observed in training. The largest all-fields coarse/fine error has **131 exact truth-bin draws**, **877 draws within 0.1°C of truth**, and **801 draws in its true coarse category**. This supports investigating sparse conditional exposure and inappropriate category selection; it does not establish a causal rarity mechanism or a missing-truth-bin explanation.

We replayed all **512,000 accepted crops** and verified the final coverage RNG and frame-stream state against every model's checkpoint. The six arms used identical accepted locations. There were **26,855 draws at 1,850 m**, covering **26,546 unique date-cells**. We also reran the final EMA models on the original mixed-depth batches and reproduced all six saved 1,850 m MSE values, using the same BF16 inference and 64 validation points per model. This is a diagnostic of the [2k-update pilot](../discrete-pilot/README.md); no model was trained or changed.

Two validation cells explain most of the large differences. Both forecasts start on October 10, 2013 and target October 15. Their temperatures barely change over the five-day interval.

| Case | Location | Latest input | OM4 target | Training labels within ±0.1°C of target | Exact same-cell training draws |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 37.099°N, 19.875°E | 13.431963°C | 13.432016°C | 877 | 2 |
| B | 7.370°N, 119.375°E | 9.969985°C | 9.969925°C | 208 | 0 |

The training-label counts include accepted 1,850 m draws across all training dates and locations. Within the inclusive 9×9 neighborhood centered on each cell, A had 46 accepted draws and B had 66; these are spatial exposure counts, not matches to the complete input state. Only 24 of the 26,855 accepted deep targets exceeded 15°C; the same 24 exceeded 20°C.

| Model, final EMA | Prediction A | Prediction B | Largest case's share of depth error |
| --- | ---: | ---: | ---: |
| Temperature-only, flat 2,048 | 13.435°C | 13.435°C | B: 99.30% |
| All-fields, flat 2,048 | 13.435°C | 13.435°C | B: 99.24% |
| Temperature-only, flat 8,192 | 20.641°C | 13.447°C | A: 83.49% |
| All-fields, flat 8,192 | 9.472°C | 10.173°C | A: 99.13% |
| Temperature-only, coarse/fine joint MAP | 12.750°C | 13.621°C | B: 95.07% |
| All-fields, coarse/fine joint MAP | 21.256°C | 12.545°C | A: 91.83% |

Error shares use each case's contribution to the saved inverse-area-corrected full-grid temperature MSE at 1,850 m. The five largest cases account for 99.45–99.89% across these models. The two flat-2,048 models make exactly the same discrete predictions at A and B; B's predicted bin had 559 deep training targets, versus 33 in B's true bin. Their failure is consistent with confusing two represented warm regimes, rather than selecting an unseen output category.

Training exposure remains sparse under the actual objective. Coverage sampling deliberately revisits unusual examples, then uses importance corrections to preserve the original sampling objective. The following mass shares sum coverage importance times the original channel/spatial loss coefficient; they are not realized losses or gradient norms. Each share divides the coefficient mass for that truth bin by the total coefficient mass of all accepted temperature targets at 1,850 m (3,614.2067). Temperature has identical coefficients in both supervision arms, so these temperature-only denominators and shares are identical across the paired arms. Neighborhood-bin counts include the true bin: ±1 covers three bins and ±4 covers nine.

| Case | Vocabulary | True-bin accepted draws | True-bin share of deep loss-coefficient mass | ±1-bin draws | ±4-bin draws |
| --- | --- | ---: | ---: | ---: | ---: |
| A | 2,048 | 559 | 0.1396% | 863 | 878 |
| A | 8,192 | 131 | 0.03285% | 371 | 810 |
| B | 2,048 | 33 | 0.004593% | 186 | 208 |
| B | 8,192 | 26 | 0.003425% | 66 | 189 |

The 8,192-leaf all-fields coarse/fine model's twelve validation cases with only 1–4 true-bin training draws contribute just **0.0448%** of its 1,850 m error. Case A, with 131 true-bin draws, contributes 91.83%. Frequency alone therefore does not order the observed errors. Counts within a fixed physical ±0.1°C window help separate fine-bin fragmentation from lack of nearby target values; they still do not measure coverage of the full conditional input distribution.

For case A, the all-fields coarse/fine decoder makes a wrong coarse decision as well as a fine decision:

| Branch | Coarse ID | Coarse probability | Training targets in coarse category at 1,850 m | Selected scalar |
| --- | ---: | ---: | ---: | ---: |
| True category | 116 | 4.41% | 801 | Target 13.432°C |
| Joint-MAP leaf's category | 125 | 9.95% | 0 | 21.256°C |
| Greedy marginal-coarse category | 106 | 11.63% | 493 | 8.953°C |

Joint MAP chooses leaf 8,051 with probability **0.373%**; the true leaf 7,426 has probability **0.110%** and ranks 289th. Greedy decoding changes the coarse branch and reduces this cell's absolute error, but it remains wrong by 4.48°C. This is not merely failure to choose the correct subdivision within a correct coarse category. The joint-MAP leaf and coarse category are unobserved at this depth but have **133** and **3,992** supervised targets, respectively, across the shared all-fields head. The same leaf ID represents channel-specific normalized ranges, so shared-head counts do not mean the model saw 133 examples of 21.256°C at other channels. Temperature-only training also has globally shared categories across depths.

The count audit and model outputs are observational. They cannot distinguish insufficient training, inadequate conditioning, parameter sharing, or sampling effects. Neither case has a missing training truth bin under the audited vocabularies. These observations do not establish whether changing the number of bins would improve predictions. The coarse/fine model can choose an entirely unobserved depth-specific coarse range. No gradients, retraining intervention, or spatial-noise test was performed.

[All 384 validation predictions and exposure counts, compressed CSV](validation-cases.csv.gz) · [Five largest cases per model](top-cases.csv) · [Nearest training labels, compressed CSV](nearest-training-labels.csv.gz) · [Error concentration and frequency strata](summary.json) · [Replay and inference verification receipts](receipts.json.gz). Nearest-label rows select by target-temperature distance only; they are not nearest input-state analogues. Full accepted draws and categorical probability arrays are retained at `torch:/scratch/jr7309/skyline-discrete-rarity-20261110/`, under `replay/training-draws.npz` and `infer/validation-probabilities.npz`.

Replay used the unchanged coverage sampler with a tested crop-free center extractor. It recovered both the point RNG and frame-stream state, including the prefetched next date. Raw exposure counts retain duplicate accepted draws; channel loss coefficients exclude inactive deep SSH. Temperature-only mass calculations supervise only temperature; all-fields calculations include every active field. The saved source normalization, vocabulary bounds, and sampling configuration are included in the receipts. CPU tests compare the crop-free extractor against the original full-crop path, including accepted locations, targets, masks, weights, and RNG consumption.

CPU replay job `22590745` took 278 seconds on 16 CPUs. GPU inference job `22590747` took 28 seconds on one RTX PRO 6000 Blackwell GPU, matching the training GPU family. Both completed with exit 0:0, without retries: **0.0078 GPU-hours** and **1.24 allocated CPU-hours** for replay. The inference reconstruction matches the saved metrics at relative tolerance 1e-6 and absolute tolerance 1e-10. Source commit: `e8c70c30e70094f7a49bf3a50a919c54127ed40b`; source archive SHA-256: `915f3fddc12cf5836d072533870ffb12dd7a6216292e7b4c71088c9319402826`. All 19 focused CPU tests and repository commit hooks passed. This is an experiment/results record and is not intended to merge.
