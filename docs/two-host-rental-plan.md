# Two-host rental plan

Rental: node3000 + node3301 (16 H200), 2026-10-16 10:00 to 10-20 00:00.

One global DDP group of rank-local 3x3 blocks (`blend_scope=rank`, block windows), validation offloaded. No parameter averaging: the logs show the stalls are I/O on peer ranks, not communication or a persistent slow GPU.

## Evidence from the logs

- Worst rental run (24926987): median step 7.49 s, mean 21.3 s. Slow steps were 64% of wall clock, and 14.5 h of that was rank 0 with its data ready, waiting on a peer.
- `alloc retries: 0` on every production step. The allocator is not the cause.
- 7-rank medians (7.46-7.57 s) equal the 1-GPU 7.5 s, so there is no persistent compute straggler. GPU 1 was excluded together with a buffer change, so "GPU 1 is bad" is unproven.
- Each step is one 1-hour forward/backward. Lead only changes when a row reseeds, which costs an extra read.
- One rank peaked at 186 GB of host RAM at startup, mostly two float32 copies of all 144 block masks in the loss normalization. That is now chunked.

## What was built

| Piece | File | What it does |
| --- | --- | --- |
| Read simulator (T0) | `scripts/_bench/replay_read.py`, `JOBS/other/speedtests/read_sim.sh` | CPU-only. N srun ranks do the trainer's per-step block reads at the training cadence, held in lockstep by a gloo barrier |
| Per-node launcher | `JOBS/torchrun_node.sh` | One torchrun per node (c10d rendezvous). Per-node store (`STORES`) and GPU pinning (`NODE_GPUS`); forwards USR1/TERM |
| Speed test | `JOBS/other/speedtests/speed_test.sh` | 2 nodes x 2 H200: GPU health, six timing cells (baseline, compile, compile + bf16 labels on scratch/disk/split), stop/resume drill |
| Report | `JOBS/other/speedtests/perf_sweep_report.py` | Step/wait/load mean and median, startup minutes, per-rank compute/tail/data and peak RSS, cross-rank compute spread |
| GPU health | `scripts/gpu_health.py` | nvidia-smi state plus a bf16 matmul and dilated-conv burn-in per GPU; flags GPUs >10% below the node median |
| Rental launcher | `JOBS/train_llc_1face.sh` | 2 nodes x 8, health check first, LR 0.0006 ramped from 0.6x over 500 steps after resume, same run dir |
| Trainer | `train.py`, `config.py`, `utils/` | Stop agreed across ranks; shared logs from rank 0 only; `rank_timing`, `compile_model`, `restart_lr_warmup_on_resume`; chunked loss norms |

## Test order (each must pass before the next is submitted)

1. T0: `sbatch JOBS/other/speedtests/read_sim.sh` (CPU queue).
2. Pre-rental suite: `sbatch JOBS/other/speedtests/speed_test.sh` (2 nodes x 2 GPUs, 20 + 180 steps per cell).
3. Rental: `sbatch JOBS/train_llc_1face.sh`. Keep `STORES` on scratch (disk can't feed 8 ranks per node), and set `NODE_GPUS` if the health check flags a GPU.

`cudnn.benchmark` is already on for every run (`utils/distributed.py:set_seed`).

## Compute and memory levers

T1-T3 (2026-10-11) never waited on data: step 7.8-8.4 s = compute 6.4-6.8 s + a 1.2-1.7 s tail. Compute is now the bottleneck.

Where GPU time goes (nsys, w672, 09-26 profile):

| Kernel family | Share |
| --- | ---: |
| Convolutions | 41% |
| GroupNorm, run in fp32 under autocast | 21% |
| Elementwise and casts | 22% |
| NCCL | 9% |

- The model is 52.6 TFLOP forward per 752² tile. A step reaches only ~24% of H200 bf16 peak.
- The full-resolution decoder block (`layers.16`, 672→1344→1344 channels) is 54% of FLOPs and most activation memory. A narrower top level would roughly halve both, but it means a new model trained from scratch.
- Replay lead and buffer size cost no GPU memory. A coarse forcing UNet run under `no_grad` before the fine forward adds only its weights. Inject its fields at a deep UNet level, not as full-resolution input channels.

Kept flags; the rental launcher defaults both on. Results: `docs/speedtest-findings.md`.

| Cell | Flag | What it changes |
| --- | --- | --- |
| compile | `--compile_model` | `nn.Module.compile()`; checkpoint keys unchanged; dynamo cache limit raised |
| bf16-labels | `--replay.resident_dtype bfloat16` | Resident labels in bf16 (~2 GB), lossless with bf16 transport |

- Compile needs Python headers on the GPU nodes: `CPATH=.../anaconda3/2023.07/include/python3.11`, set in both launchers.
- If compute gets faster, loads matter again: scratch (3.1-3.6 s) still hides under ~4.5 s of compute, disk (4-5.7 s) may not.
