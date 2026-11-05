<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Native v2 deep-OHC diagnosis

The native 64-channel v2 InstanceNorm model's deep-OHC (700–2000 m) validation RMSE mostly
fluctuates near 0.75 GJ/m² after update 3,000, reaches 0.7510 at 4,500, then rises
to 0.7805 at 5,000. This is a late rise following a plateau, not a sustained rise
throughout training. The matched small processor improves from 0.7405 at 3,000
to 0.6064 at 5,000. These are EMA validation scores from the
[native processor comparison](native-latent-pair-2026-10-30.md).

OBS interior training loss continues falling in both models:

| Update window | v2 InstanceNorm | Small processor |
|---|---:|---:|
| 2,501–3,000 | 0.16394 | 0.16583 |
| 3,501–4,000 | 0.12809 | 0.13167 |
| 4,501–5,000 | 0.07958 | 0.08546 |

These are means over OBS updates only, using the current training weights;
the validation scores above use EMA weights. The loss is the diffusion-denoising
objective for monthly temperature and salinity, not decoded OHC RMSE. The
training-loss decrease and validation-OHC rise are consistent with overfitting,
but comparing different objectives does not establish its cause.

## Pending diagnostic decodes

Beta job **223638** evaluates saved checkpoints without training. It requests
one four-GPU node, standard QoS, and two hours. Code commit: `3cb135d8b`.
Root: `beta:/projects/ny/lz1955/multiscale/jrusak/runs/native-ohc-diagnosis-20261105`.

The fixed cohorts are nine training months (November 2011–July 2012) and the
original nine validation months (November 2013–July 2014). Matching calendar
months reduces seasonal confounding; nine training months do not represent the
full training distribution. Each decode uses the original eight-member mean,
32 sampling steps and fixed noise seeds. Validation metrics must reproduce the
original saved scores within relative tolerance 1e-6 and absolute tolerance
1e-8 before the new artifacts are accepted.

The v2 checkpoints are 2,500, 3,000, 4,000, 4,500 and 5,000, each with raw and
EMA weights. The small-model checkpoints are 3,000, 4,500 and 5,000 with EMA.
The outputs include surface fields, monthly temperature/salinity and 0–700 m and 700–2000 m OHC, with targets and masks. Analysis will separate spatial mean bias from
pattern error, identify regional and depth contributions, compare training and
validation decoded error, and check whether the rise is shared across months.
