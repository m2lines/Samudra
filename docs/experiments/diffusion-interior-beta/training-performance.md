<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Latent diffusion training performance investigation

Work in progress, October 1, 2026. These are disposable timing experiments using
an existing checkpoint, not resumed scientific training. The scientific runs
remain stopped pending review of loss-domain and deterministic-control matching.

## First hardware and layout measurements

One fixed, real observation-training month with seven forecast leads; two members,
16 sampling steps, the unchanged fair-CRPS objective, bf16, backward pass, clipping
and AdamW updates. Same checkpoint, sample and explicit noise seeds on both GPUs.
Two warmup updates precede three measured updates. Optimizer state is fresh in
each variant; these short trajectories measure runtime, not fine-tuning quality.
Data are preloaded. Replay, validation, checkpoint saves and loader startup are
excluded; this is not end-to-end campaign throughput.

| GPU | Original layout, median seconds/update | Channels-last | Result |
| --- | ---: | ---: | --- |
| L40S | 35.073 | 38.763 | Channels-last slower |
| H200 NVL | 24.269 | 26.826 | Channels-last slower |

The first H200 comparison is **1.445× faster**, a 30.8% reduction in update time.
Channels-last adds about 10.5% on both devices. These are single-node placements,
not a distribution over host contention or GPU variants; subsequent paired
measurements will retain their own baseline.

Producer `75242f068`; jobs **24535708** (L40S) and **24535709** (H200), both completed.
The two allocations used 664 and 551 seconds, respectively (0.3375 GPU-hours total).
Profiler time is excluded from the timing medians. Raw operator summaries,
per-update timings and checkpoint/sample fingerprints are retained under
`runs/latent-performance-v1/job-JOBID` on Engaging and in local collected outputs.

## Bottleneck and follow-up tests

On H200, the separate profiled baseline update attributes 47.4% of self GPU time
to `aten::native_group_norm`. Convolution forward and backward each account for
about 10%. Operator and kernel rows overlap: their percentages must not be added.
Compilation tests target GroupNorm modules alone and then the decoder; jobs
**24536571** (H200) and **24536572** (L40S), producer `70f610d89`. Each retains an
in-job original baseline and records first-update gradient samples. No speedup
or numerical equivalence is claimed until their outputs are checked.

An additional opt-in change omits initial physical readouts when only forecast
losses are needed. Those fields never enter latent recurrence. It consumes the
same unused noise draw to preserve future samples. Seven tests pass, including
bitwise forecast, parameter-gradient and generator-state equivalence on small
stochastic and deterministic models. It is not enabled in existing scientific
runs; its full-size runtime benefit remains to be measured.

## Research basis

PyTorch documents [channels-last convolution layouts](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html),
[compilation for diffusion](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/),
and the [recomputation tradeoff in activation checkpointing](https://pytorch.org/blog/activation-checkpointing-techniques/).
These identify candidates, not expected speedups for this workload. In particular,
the channels-last hypothesis failed our initial measurement. Reducing sampling
steps or changing the learning objective would require separate scientific
validation and will not be presented as an equivalent implementation speedup.
