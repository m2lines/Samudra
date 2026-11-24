<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Train

## Rollout curriculum

Training backpropagates through a rollout of `steps` model calls. A list of lengths with matching `step_transition`
epochs trains in stages, for example short rollouts first and longer ones later, as GraphCast and Keisler (2022) do.
Stage `i` runs from epoch `step_transition[i - 1]` up to, but not including, epoch `step_transition[i]`.

Memory grows with rollout length times batch size. To keep the effective batch constant as rollouts lengthen, give
each stage its own `step_batch_size` and `step_gradient_accumulation_steps`. Their first entries must equal
`batch_size` and `gradient_accumulation_steps`:

```yaml
batch_size: 4
gradient_accumulation_steps: 1
steps:                            [1, 2, 4, 8, 16, 32]
step_transition:                     [30, 45, 55, 62, 67]
step_batch_size:                  [4, 4, 4, 4,  2,  1]
step_gradient_accumulation_steps: [1, 1, 1, 1,  2,  4]
```

Validation runs one step per sample and keeps `batch_size` in every stage. The `dynamic` loss updates its weights
once per micro-batch, so a smaller micro-batch makes it adapt faster and with more noise, even at a constant
effective batch.

::: samudra.train
