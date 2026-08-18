<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver seam-removal search on 2-degree OM4

This experiment compares hard decoder tiling, shared global context, and
overlap-add decoding while holding the selected Perceiver architecture and
learning rate fixed. See
[`docs/experiments/perceiver_2deg_seam_removal_search.md`](../../docs/experiments/perceiver_2deg_seam_removal_search.md)
for the pre-registered hypotheses, design, queries, and eventual results.

It reuses the validated Torch controller wrapper from
`experiments/perceiver_search_2deg/container_python.sh`, the staged public OM4
2-degree data, W&B grouping, and public Parquet artifact publication.
