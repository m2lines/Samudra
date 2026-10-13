# Speed test findings

Job logs are `logs/1-face/10-16-26/speedtests/<job>.out`. Per-cell logs are `logs/speed_test/speed-test-<job>-<cell>.out` and `logs/read_sim/<job>/`. The tooling is in `JOBS/other/speedtests/`.

## 10-13-26

Pre-rental suite on 2 nodes x 2 H200 (node2433 + node3400), 20 + 180 steps per cell (job 25686318). Logs: `logs/1face-speed-25686318.out`, cells `logs/speed_test/speed-test-25686318-*.out`.

- **Baseline:** 8.62 s median / 10.42 s mean step, 476 retries, worst 42.9 s; it degrades over a long run.
- **Compile:** 5.48 / 5.55 s, 2 retries, but free GPU memory dips to 0.0-0.1 GB (1.3 GB headroom).
- **Compile + bf16 labels (chosen default):** 5.52 / 5.53 s, 0 retries, steady 2.9 GB free (4.1 GB headroom).
- **Disk, compile + bf16:** 5.52 s median but 5.95 s mean (worst 33.9 s) from disk-read hiccups (4.7 s mean load).
- **Split, compile / + bf16:** 5.78 / 5.63 s mean; node3400 loads about 1 s slower than node2433 (4.8 vs 3.7 s), still hidden.
- **GPU health:** all 4 GPUs within 1% (655-664 TFLOP/s bf16 matmul).
- **Stop/resume drill:** passed; all 4 ranks stopped at one step, `ckpt_emergency.pt` was written, and the resume trained 20 steps.

## 10-12-26

Three flags were removed after the 2xH200 A/B (job 25663405):

- **channels_last:** made compile 47% slower (5.39 to 7.95 s/step).
- **Fused Adam:** no gain (8.36 vs 8.79 s/step baseline).
- **bf16 gradient all-reduce:** no gain on one node (8.27 vs 8.79 s/step).

## All speed tests by date

Before September, almost every run was data-bound: about 0.3-0.6 s/iter of compute, with the rest spent waiting on data.

- **02-26-26, first DDP tests.** 1 GPU: 7.4-8.0 s/iter, of which 7.4 s is data wait (0.31 s compute). 3-GPU runs hung on NCCL or crashed in the pin thread. Logs: `logs/old/old-2/DDP_tests/`.
- **02-27-26, 3 GPUs with 4 workers per rank.** No faster than 1 GPU (21.8-27.0 s/iter). Logs: `logs/old/old-2/DDP_tests/*3gpu_test*-9795047.out`.
- **02-28-26, worker/pin matrix (3 GPUs).** 24 workers instead of 12 cut 22.3 to 14.4 s/iter. pin_memory didn't help (16.6 vs 14.4 s at 24 workers). Logs: `logs/old/old-2/DDP_tests/worker_pin_matrix/`.
- **03-04-26, 2 GPUs with all fields.** 26.7 s/iter against a 0.97 s median step (23.5 s data wait). Logs: `logs/old/old-2/DDP_tests/*2gpu_speed_test*-9996343.out`.
- **04-02/04-05-26, h200-tests (temporal stride 24).** 43.7-48.3 s/iter, of which 42-48 s is data wait. Logs: `logs/old/old-2/h200-tests/`.
- **05-04-26, cache worker/pin sweep.** 6 workers without pin: 92.6 s/iter; with pin: about 45 s/iter (0.57 s compute). Logs: `logs/old/cache/old/cache_speed_tests/2026-05-4-*`.
- **05-05-26, prefetch_factor=4.** First full epoch at 11.7 s/iter, down from 45-93 s; two crashes and a scratch-copy test (56 MB/s) are in the same dir. Logs: `logs/old/cache/old/cache_speed_tests/*-13383960.out`.
- **06-25/06-28-26, replay dataloader profiling.** Replay averaged 4.3-27.6 s/iter, dominated by data wait, with one prefetch thread. Logs: `logs/nsys/logs/*PROFILE*`.
- **06-29-26, 6 replay prefetch threads.** 0.49 s/iter mean with 0.12 s data wait, down from 27.6 s. Logs: `logs/nsys/logs/*PROFILE-4-16815459.out`.
- **06-30-26, batch 2 x grad-accum 2, 4 workers.** 0.65 s/iter with 0.002 s data wait, so fully compute-bound. Logs: `logs/nsys/logs/*PROFILE-5-float16-16872980.out`.
- **08-12-26, 4-tile validation speed.** The validation fix took autoregressive validation from 5.3-7.9 to 0.66-0.81 s/step (training 0.65-0.84 s/iter). Logs: `logs/old/4-tiles/speed-profiles-etc/`, `logs/old/4-tiles/tests/*VAL-SPEED-TEST-20314527.out`.
- **08-21-26, 4x1104 tiles.** Inconclusive: crashed with a CUDA invalid-configuration error after 16 min of setup. Logs: `logs/old/4-tiles/1104-tiles/`.
- **09-11-26, 1-tile loader comparison.** The zarr cache was fastest (0.96 s/iter) against the Rust raw-LLC loader (1.62-1.94 s/iter). Logs: `logs/nsys/logs/*1-tile-*-nsys*`.
- **09-11-26, 4 grouped 752 tiles from cache.** 3.83 s/iter mean with 0.18 s data wait, so compute-bound. Logs: `logs/nsys/logs/*4-752-tile-cache-nsys-22497307.out`.
- **09-26-26, Nsight width sweep (4xH200, face topology).**
  - Median step: w448 6.6 s, w576 8.0 s, w672 9.5 s with 2 tiles per chunk, 12.3 s with 3.
  - GPU time: 41% convolutions, 21% fp32 GroupNorm, 22% elementwise.
  - Logs: `logs/1-face/10-02-26/pre/nsight/`. Profiles: `logs/nsys/width-sweep-24068216-*.nsys-rep`.
- **09-28-26, code-lever sweep (4xH200).**
  - Code fixes t1-t4 plus `checkpointing=simple` cut w672 with 2 tiles per chunk from 10.3 to 6.6 s/step.
  - channels_last was 20% slower. One tile per chunk was slower. Three tiles per chunk thrashed (74 allocator retries).
  - Logs: `logs/1-face/10-02-26/pre/sweeps/code-lever-sweep/`.
- **09-30-26, storage sweep (4xH200).**
  - Disk and flash ran the same step at w448 (11.78 vs 11.70 s).
  - Every rank-local cell crashed at that point.
  - Logs: `logs/1-face/10-02-26/pre/sweeps/perf-sweep-24494672-*.out`.
- **09-30-26, face read, disk vs flash (CPU, 4 ranks x 8 threads).**
  - One full-face frame: 2.68 s disk vs 1.08 s flash (median).
  - Logs: `logs/1-face/10-02-26/pre/sweeps/2026-09-30:llc:face-read-disk-vs-flash-24495555.out` and `face-read-*.json`.
- **10-02-26, rental preflights (node3400).**
  - 8 GPUs, buffer 5: erratic (median 11.7 s, mean 29.2 s, worst 185 s), with rank 0's own reads spiking.
  - 7 GPUs, buffer 4: steady at 7.45 s with 2 tiles per chunk. One tile per chunk took 8.4 s; three took 16.8 s with 108 retries.
  - Logs: `logs/1-face/10-02-26/pre/preflight/`.
- **10-08-26, old vs chunk-aligned block windows (1xH200).**
  - Block windows loaded 1.36x faster: scratch 4.75 to 3.48 s, disk 5.55 to 4.04 s.
  - Step time stayed about 7.5 s, because the step was compute-bound.
  - Logs: `1face-speed-1h200-25302522.out`. Jobs 25235242 and 25278135 failed on setup and an orphaned worker.
- **10-09-26, read simulator on CPU nodes.**
  - One rank: 2.6 s scratch vs 3.7 s disk per block.
  - Eight ranks per CPU node choked at 0.54 GB/s on the node's NIC, so these numbers are not representative.
  - Logs: `1face-read-sim-25433349.out`.
- **10-11-26, read simulator on H200 nodes (2 nodes x 8 ranks, preempted after 4 of 7 cells).**
  - Scratch: 3.4 s mean, no stalls.
  - Disk: 8.0 s mean, simulated step 8.42 s.
  - Split: disk host 6.5 s, scratch host 3.1 s. Disk cannot feed 8 ranks per node under a compiled ~5.4 s step.
  - Logs: `1face-read-sim-25544834.out`.
- **10-11-26, T1/T2/T3 (2 GPUs on 1 node; 1+1 nodes; 2+2 nodes).**
  - Step 7.8-8.4 s, with zero data wait.
  - Compute spread across ranks at most 2.4%. Peak host RAM 94-99 GB per rank.
  - Compile cells failed because the GPU nodes have no `Python.h`.
  - Logs: `1face-speed-25544962.out`, `25544968`, `25544976`.
- **10-12-26, A/B (2xH200).**
  - Compile cut the step from 8.79 to 5.39 s, with 0 retries.
  - bf16 labels alone: 7.03 s.
  - all-disk matched all (7.00 vs 7.08 s).
  - Logs: `1face-speed-25663405.out`. Job 25617804 failed because node4100-c1 has no `/usr/bin/python3.11`.
