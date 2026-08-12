<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Perceiver v2: direct output-query decoder

## Motivation

The decoder root-cause experiments found that SamudraMulti's full PerceiverIO
decoder introduces a second learned latent bank after the encoder and spatial
processor. This makes output routing unnecessarily indirect and gives each
cross-attention head a value path narrower than the 77 prognostic channels in
the default configuration.

The first v2 intervention keeps the original Perceiver latent-array encoder,
but applies the Perceiver IO output-query stage directly to the processor
tokens. It also retains the output query through a residual connection. This
matters in small/local contexts: attention over one context token is otherwise
independent of the query because the softmax has length one.

The old full `perceiver_io` decoder remains available as an ablation. The
SamudraMulti OM4 preset selects `direct_cross_attention`, uses 128-dimensional
output queries, and transports 128 values per output cross-attention operation
(two heads of width 64).

The Perceiver blocks are owned by Samudra and use
`torch.nn.functional.scaled_dot_product_attention`. With `auto`, PyTorch
selects FlashAttention, memory-efficient attention, or its math implementation
for the current device, dtype, and tensor shapes. The former
`perceiver-pytorch`, `flash-perceiver`, and external `flash-attn` dependencies
are not required. The `naive` and `flash` configuration values remain as
backward-compatible aliases that force PyTorch's math and FlashAttention
backends respectively; new configurations should normally use `auto`.

## Laptop probe

`docs/experiments/spikes/probe_perceiver_om4_patches.py` compares both decoders with the same
original-Perceiver encoder, random seed, optimization schedule, and real OM4
samples. The diagnostic uses 3x5 patches containing all 77 prognostic channels
from the public 2-degree source. Eight timestamps are used for fitting and four
later timestamps are held out. These experiments are an architectural gate,
not a production forecast benchmark.

On Apple MPS, with 32 patches per timestamp and 500 optimizer updates:

| Task | Decoder | Parameters | Held-out masked MSE | Predicted/target RMS |
| --- | --- | ---: | ---: | ---: |
| Reconstruction | Full PerceiverIO | 3,424,329 | 0.3923 | 0.7930 |
| Reconstruction | Direct output queries | 2,697,545 | **0.3660** | 0.7954 |
| One-step forecast | Full PerceiverIO | 3,424,329 | 0.4420 | 0.7645 |
| One-step forecast | Direct output queries | 2,697,545 | **0.4256** | 0.7984 |

The direct decoder improves reconstruction MSE by 6.7% and the preliminary
one-step MSE by 3.7%, with 21% fewer parameters. This is positive evidence for
removing the second latent bottleneck, but the sample is too small to establish
forecast skill or statistical significance.

## Next gates

1. Repeat the comparison across several seeds and more held-out timestamps.
2. Run a full SamudraMulti single-scale 2-degree training experiment, including
   boundary forcings and autoregressive validation.
3. Ablate query width, transported value width, context radius, and patch size.
4. Only after single-scale rollout stability is established, test shared
   weights across 2-, 1-, and 0.5-degree sources.
5. Profile hierarchical/local input routing before attempting LLC-scale grids;
   the original Perceiver removes the quadratic input bottleneck, but a global
   cross-attention pass still scales linearly with the number of grid points.
