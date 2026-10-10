<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# CUDA investigation record

The successful four-run comparison is in [RESULTS.md](../RESULTS.md).
The original checkpoint was used only for diagnosis; final reported runs started fresh.

| Job | Outcome |
|---|---|
| 19457054/56/59/61 | All original runs failed in epoch 2 on gr103, rank 0 CUDA/cuDNN errors. |
| 19460455 | Canceled during initialization after landing on gr103 again. |
| 19461486 | gr101: container could not resolve user identity; no training. |
| 19511233 | gr106: resume failed in 56s before first step. |
| 19512489 | Synchronous CUDA reporting identified skip-module Conv2d forward (44s). |
| 19513737 | Disabling cuDNN benchmarking still failed there (43s). |
| 19515467 | Captured rank-0 input/weights before first failing 1×1 Conv2d (40s). |
| 19518164 | Isolated one-GPU suite passed all five forward/backward cases (21s). |
| 19518558 | Rust host prefetch passed 105 batches then failed during skip-convolution recomputation (3m27s). |
| 19520341 | Native backend for 1×1 convolutions: completed epochs 2–10 and their validations, reached epoch 11, then intended 30-minute timeout. |

Captured input: float32 `[2,160,180,360]`, strides `[10368000,64800,360,1]`;
weights `[280,160,1,1]`, same padding, groups 1, bf16 autocast.
`diagnostics/failing-conv.pt` (~80 MiB) remains at the remote campaign root.
[Isolated results](conv-suite-results.json) show finite inputs/weights and successful
original, random, channels-last, FP32, and cuDNN-disabled cases. This does not
reproduce the failure outside training or establish the library root cause.

The successful wrapper preserves Conv2d parameter shapes and operations but
changes its 1×1 backend. It applies at import so spawned evaluation workers
also use it; the main guard prevents recursive training. Its exact executed
bytes are archived in `sd_pointwise_native.executed.py.txt`; the `.py` version
only adds formatting and a type-checker annotation. Other diagnostic scripts are
formatted copies for reproducibility. No core library change or general-purpose
workaround PR is claimed here.

Runtime root: `/scratch/jr7309/runs/current-sd-seeds16-17`.
Scoring initially failed before Python in job 19527118 because the CPU module
load failed. The successful CPU launcher uses the already-installed runtime.
The scheduler chooses CPU partitions automatically; explicitly requesting
`cpu_short` was rejected for the scoring resource request.
