<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Does Samudra 2 forecast better when land is absent from every layer?

**This is an experiment and results record. It is not intended to merge.** The library change it tests,
`land: mask | omit` ([#1001](https://github.com/m2lines/Samudra/issues/1001)), is in its own feature PR.

## Question

Samudra's U-Net computes over every grid cell, land included, and zeroes land only in its output. So land values
mix into coastal ocean cells through every convolution, normalization, pool and upsample. With `land: omit`, land is
absent from every layer instead (partial convolutions, ocean-only batch norm, ocean-only pooling and upsampling; see
`docs/models/samudra.md`). A small 2° prototype in #1001 forecast slightly better at 5 days and drifted far less over
8 years. Does that hold for the full Samudra 2 at 1°?

## Design

Two runs from scratch, identical except for `model.land`:

| | `mask` (control: standard Samudra 2) | `omit` |
| --- | --- | --- |
| Model | Samudra 2 (`samudra_om4_v2` preset): widths 280/380/480/520, ConvNeXt ×2, batch norm, fp32 | same, `land: omit` |
| Data | 1° OM4 v2026-09, 2 inputs → 2 five-day outputs per call; train 1975–2013-09, val 2013-10–2014-10 | same |
| Recipe | dynamic variance-weighted MSE (cap 20), 4 calls, 70 epochs, cosine lr 6e-4, batch 32 on 8 × RTX 6000, seed 15, EMA 0.999 | same |
| Evaluation | final EMA checkpoint; rollout 2014-10-10 → 2022-12-24; `samudra.metrics` vs OM4 and OISST, DUACS, Argo/IAP | same |

The configs live in `src/samudra/configs/experiments/land_omit_onedeg/`, because the Torch code layer packages only
`src/`. `train_mask.yaml` and `train_omit.yaml` resolve to configs that differ only in `model.land` and the run name.

The control is retrained rather than taken from the paper checkpoint. The paper model was trained on `om4_onedeg_v3`
with older code, so comparing against it would confound the flag with data and code changes.

## Caveats

- One seed per arm. A difference smaller than the seed-to-seed spread would not be meaningful, and we don't measure
  that spread here.
- The coastal band is scored separately in the report, since that is where the two models compute differently.

## Running

```bash
# On the Torch login node, after pushing this branch and building its code layer:
CODE_COMMIT=<full sha> bash experiments/land_omit_onedeg/submit.sh
```

Job IDs are recorded in `/scratch/am16581/agent-logs/2026-11-15-land-omit-onedeg-jobs.tsv`.
