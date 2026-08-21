<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# One-degree spatial-grid Perceiver

This preset is the full single-scale experiment selected by the Perceiver
encoder/decoder meta-analysis. It uses one-degree OM4 only. The encoder groups
6 x 10 cells and retains four coordinate-tied Perceiver outputs, producing a
60 x 72 processor grid. The decoder uses direct cross-attention, zero anonymous
context, one-ring overlap assembly, smooth processor conditioning, and physical
residual prediction.

The train preset assumes two GPUs: per-rank batch 2 and eight accumulation steps
produce effective global batch 32. Change accumulation if the world size changes
so optimizer exposure remains comparable. The experiment data preset uses raw
`hfds` forcing (it does not compute `hfds` anomalies). Set `experiment.data_root`
to the one-degree directory that contains `OM4.zarr`, `OM4_means.zarr`, and
`OM4_stds.zarr`. Pass the validation-selected checkpoint explicitly to evaluation:

```bash
samudra train perceiver_spatial_grid_1deg/train.yaml --experiment.data_root=/path/to/om4_onedeg_v3
samudra eval perceiver_spatial_grid_1deg/eval.yaml --experiment.data_root=/path/to/om4_onedeg_v3 --ckpt_path=/path/to/ckpt.pt
samudra viz perceiver_spatial_grid_1deg/viz.yaml --data_root=/path/to/om4_onedeg_v3
```
