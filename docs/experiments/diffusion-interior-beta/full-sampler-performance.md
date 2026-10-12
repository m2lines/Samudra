<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Full-sampler performance on Beta, October 12

**Distributing the same effective batch of eight across eight GPUs reduced measured
observation computation from 312.49 to 38.90 seconds, approximately 8×.**
This is the strongest measured opportunity; it preserves the full sampler and loss.

This profiles the working full-sampler observation objective, not the unsuccessful
one-step surrogate. Compilation and BF16 were already enabled and are confirmed
in the runtime records. Neither removing a sampler synchronization nor switching
the whole model to channels-last produced a useful speedup. Larger decoder batches
are promising, but require preserving each example's masks, monthly weights and
random draws; they are not a drop-in training batch-size setting.

## Matched observation computation

All measurements load the same residual replay-bound parent: 8,128 OM4 and 8,128
observation updates. They use the same eight training months, corruption/noise
seeds, full-latitude masks, normalization and initial weights. Each example uses
two members and 32 Heun intervals (63 decoder evaluations per readout), with the
existing forecast, completion and monthly interior reconstruction losses. Each
configuration has one warm-up and two measured passes. Weights never change.

| Configuration | Seconds per effective batch of eight | Relative speed |
|---|---:|---:|
| One GPU, current compiled BF16 | 312.49 | 1.00× |
| One GPU, no sampler scalar synchronization | 310.96 | 1.005× |
| One GPU, channels-last | 370.52 | 0.84× |
| Eight GPUs, same effective batch | 38.90 | 8.03× |

The two warmed eight-GPU passes took 39.49 and 38.32 s. The slight apparent
superlinear speedup is within placement/timing variability, not evidence of
better-than-linear scaling. Rank-zero reduction time includes waiting for longer
examples on other ranks; it is not a pure communication benchmark.

These are forward/backward and gradient-reduction/clipping measurements, excluding
optimizer updates, OM4/replay work, checkpoint writing and recurring data loading.
Eight GPU ranks each receive one of the same eight examples; their gradients are
summed after the existing division by eight. This is a distributed computation
prototype, not a completed production DDP training runner. The single-GPU variants
ran concurrently on separate devices of the first host; the eight-rank phase used
both hosts. Two repetitions are sufficient to reject a large gain from the tested
switches, not to establish a sub-percent synchronization improvement.

The baseline's measured compute divides into **166.54 s forecast/completion
(53.3%)** and **145.94 s monthly reconstruction (46.7%)**. Its peak allocated tensor
memory is **42.77 GiB**. Initial loading/normalization of the eight months took
5.41 s and is outside the measured passes; this does not establish production I/O
throughput for a changing cohort. Norm/clipping bookkeeping is about 9 ms.

## Compilation, precision and kernels

The baseline compiled two decoder graphs, capturing 356 calls, without recorded
graph breaks. Graph counts stayed fixed across the measured passes. The initializer
and processor were not separately compiled. BF16 autocast covers the neural
network; parameters and selected sampler/loss arithmetic remain FP32. This is
mixed precision, not conversion of every tensor to 16 bits.

A separate representative-example profile captured 1,474,231 CUDA events.
Convolution forward/backward, layout conversions and padding are prominent.
NCHW/NHWC conversion kernels account for 3.78 s of device activity in that capture,
but the measured channels-last configuration was slower. Operator and kernel rows
in the [operator table](full-sampler-performance-assets/operators.txt) overlap;
do not add them or interpret profiler CPU/CUDA totals as unprofiled utilization.
The large trace is retained remotely rather than committed to Git.

The synchronization experiment substitutes a known Python loop-index condition
for a CUDA-scalar endpoint test; Heun arithmetic and evaluation counts stay the
same. CPU output/gradient checks were bit-exact with and without anchors and
activation checkpointing. Separate GPU processes are not bitwise equivalent:
relative loss differences versus baseline are 9.15 ppm for synchronization removal
and 24.29 ppm for channels-last. Sampled-gradient relative L2 differences are
2.26% and 3.57%, with cosines 0.999768 and 0.999426. These do not justify calling
the GPU computations identical or claiming a scientific quality improvement.
The eight-GPU reduction changes the loss by 17.25 ppm
and the sampled gradient by 1.91% relative L2
(cosine 0.999825). Objective and seeds are matched, but
BF16/backend/reduction order prevent a bitwise comparison; validate an actual
short training continuation when integrating distributed optimizer updates.

## Decoder batch size

| Decoder batch | Forward/backward time | Examples/s | Throughput versus batch one | Peak allocated GiB |
|---|---:|---:|---:|---:|
| 1 | 10.35 ms | 96.59 | 1.00× | 1.78 |
| 2 | 23.82 ms | 83.96 | 0.87× | 2.96 |
| 4 | 27.37 ms | 146.14 | 1.51× | 5.36 |
| 8 | 51.11 ms | 156.52 | 1.62× | 10.40 |

This isolates one compiled decoder forward/backward call, without activation
checkpointing, using a real observation latent/mask and synthetic correlated noisy
physical inputs. It does not include a full diffusion sampler or retained rollout
graphs. Identical examples are repeated with a per-example mean loss; output and
gradient comparisons against the eager serial reference are exported in the CSV.
Those comparisons include compiled-versus-eager differences, not just batching.

Batch four is a more promising starting point than simply doubling the batch.
The next implementation candidate is batching independent readouts (members and/or
leads) while preserving the current sampling draws and per-example losses. Its
whole-objective speed and memory must be measured before adopting it. The isolated
batch-eight memory figure does not imply a full batch-eight sampler fits.

## Eight-GPU execution and provenance

Beta test job **212394** allocated two hosts, four GPUs each, for at most one hour
(maximum eight allocated GPU-hours). Slurm labels these GPUs `b200`; PyTorch reports
**NVIDIA GB200**. The original NCCL bootstrap auto-selected `ibp3s0` and failed with
“No route to host.” A retry inside the same allocation explicitly used `bond0`;
all eight ranks connected, including cross-host `P2P/MNNVL`. The existing stalled
launch remained allocated while the retry ran, so startup cost and any residual
process overhead must not be hidden in an idealized scaling claim.

Runtime: PyTorch `2.12.0a0+0291f960b6.nv26.04.48445190`, CUDA 13.2; the same pinned
PhysicsNeMo container as the preceding pilot. Runtime archive
`diffusion-profile-2deb0da809cb`, based on Git `b4bbb3e6f`. The retained launcher adds
`NCCL_SOCKET_IFNAME=bond0` after the observed recovery; the archived initial launcher
did not set it. All profiling measurements completed. The allocation was then explicitly
cancelled to release the abandoned original NCCL processes, so Slurm records
`CANCELLED` despite complete benchmark receipts. Actual allocation was 2,581 s ×
eight GPUs = **5.74 GPU-hours**; monitoring is retired. No scientific checkpoint
was written or modified.

Recommended next step: integrate data parallelism into the real mixed trainer,
with correct per-rank OM4 replay ownership and one optimizer step per effective
batch. Keep compilation, BF16, sampling steps and effective batch unchanged.
Then test batching four independent decoder readouts. Do not extrapolate this
observation-only result directly into an eightfold end-to-end campaign speedup.

Raw receipts, gradients, logs, allocation and trace:
`beta:/projects/ny/lz1955/multiscale/jrusak/profiles/diffusion-full-sampler-2deb0da809cb/results/212394/`.

- [Observation timings and numerical comparisons](full-sampler-performance-assets/full-observation.csv)
- [Decoder batch timings and numerical comparisons](full-sampler-performance-assets/decoder-batch.csv)
- [Machine-readable summary](full-sampler-performance-assets/summary.json)
- [Observation profiling code](../../../scripts/profile_diffusion_full_sampler.py)
- [Decoder profiling code](../../../scripts/profile_diffusion_decoder_batch.py)
- [Beta launcher](../../../scripts/slurm_diffusion_profile_beta.sbatch)
