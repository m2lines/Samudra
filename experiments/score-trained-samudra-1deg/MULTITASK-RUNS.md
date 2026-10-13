<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Structured multitask execution record

The [approved F/M comparison](MULTITASK.md) uses the user-confirmed 50/25/12.5/12.5 mixture.
This is an experiment/results record, not intended to merge.

## October 12: Torch qualification

Qualification source `79cb5003903bb6b800bae16d562de8dbd484948a`, immutable snapshot under
`/scratch/jr7309/runs/score-multitask-1deg-20261012/sources/`.
The original H100 submission used source `480bcc7df886a808ab324c3410a055f4a8fa73b1`, adding a GPU-visibility
guard to qualification. Replacement RTX production uses `a4b90dfddcb11f45be0c4ab378915c916c0fd9aa`,
the same source as the successful RTX retry, with the opt-in native 1×1 backend described below.

| Stage | Torch job | Resources | Last verified state |
| --- | --- | --- | --- |
| All-task qualification | 19648755 | 1 H100, 6 CPUs, 64 GiB RAM | Completed, exit 0; 2:01 on gh001 |
| Completion notification | 19648789 | CPU-only, 1 CPU, 512 MiB | Completed; callback delivered |
| F: forecast-only | 19649331 | 2 H100s, 16 CPUs, 192 GiB RAM, 36-hour ceiling | Cancelled while pending; replaced by RTX pair |
| M: structured multitask | 19649333 | Same as F | Cancelled while pending; replaced by RTX pair |
| Original production notifications | 19649371 / 19649375 | CPU-only, 1 CPU, 512 MiB each | Cancelled before replacing H100 jobs |
| RTX qualification | 19656293 | 2 RTX GPUs, 16 CPUs, 192 GiB RAM | Failed after 59 seconds on gr103; CUDA illegal memory access |
| RTX qualification notification | 19656296 | CPU-only, 1 CPU, 512 MiB | Completed; quiet monitor handed result back |
| RTX native-1×1 retry | 19656804 | Same gr103 node, 2 RTX GPUs, 16 CPUs, 192 GiB RAM | Completed, exit 0; 1:52 on gr103 |
| RTX retry notification | 19656825 | CPU-only, 1 CPU, 512 MiB | Completed; terminal result verified |
| F: RTX forecast-only | 19657151 | 2 RTX GPUs, 16 CPUs, 192 GiB RAM, 36-hour ceiling | Failed near epoch 6 end; epoch 5 checkpoint retained |
| M: RTX structured multitask | 19657152 | Same as RTX F | Failed before training on gr102; initial NCCL barrier timed out |
| M: unchanged startup retry | 19657895 | Same as RTX F | Running on gr106; epoch 6 progress verified |
| F: checkpoint recovery | 19664764 | Same as RTX F | Submitted; resumes epoch 5 checkpoint |
| F recovery notification | 19664888 | CPU-only, 1 CPU, 512 MiB | Verified `afterany:19664764` |
| M retry notification | 19657907 | CPU-only, 1 CPU, 512 MiB | Verified `afterany:19657895` |
| RTX production notifications | 19657177 / 19657179 | CPU-only, 1 CPU, 512 MiB each | Verified `afterany` dependencies on RTX F / M |

Campaign root: `/scratch/jr7309/runs/score-multitask-1deg-20261012`.
Original inputs: `configs/{qualification,train}-{forecast,multitask}.yaml`.
Current production inputs: `configs/train-{forecast,multitask}-rtx.yaml`.
Current production outputs: `outputs/train-{forecast,multitask}-rtx/`.
Logs: `logs/<job>.{out,err}`; allocation records: `allocations/`.

The qualification explicitly backpropagated all four tasks on real data, ran three optimizer updates,
and passed one-step/completion and 360-day ensemble validation. [Raw qualification results](results/multitask-qualification-19648755.json)
record **42.42 GB peak allocated GPU memory** and 18.74 seconds for the six training minibatches, including
initial summary overhead. Job cost was 0.0336 allocated H100 GPU-hours. The largest recorded host RSS was
11.29 GiB. These short-run timings do not establish full epoch throughput. The randomly sampled updates
did not happen to choose a gap task, but the explicit pre-update backward probe verified it separately.
The published JSON retains every metric and represents its verified Git fingerprint as a commit URL.
The qualification-only checkpoint does not count as a training epoch or initialize production weights.
Production retains batch 4/rank, accumulation 4, effective batch 32 and 70 epochs. Both arms use online W&B.

The qualification callback `f4d1a330-e42b-4131-88f4-6bfc25456181` delivered and was retired after verification.
Authentication passed on the login and compute nodes. Callback credentials stay in the private remote
directory and are not committed. Production receives separate callbacks and hourly reminders.

F callback: `ccd71a7c-9edf-4134-86e9-e3e9ccd08321`; M callback: `6e8c72d6-35e1-4d8c-a7ed-75ae5b950b43`.
Both authenticated successfully from Torch; both registrations were retired when the pending H100 jobs
were replaced. Original production routing was verified:
account `torch_pr_347_lzanna`, partition `h100`, constraint `h100`, two GPUs per job,
`Comment=preemption=yes;requeue=true`, `Requeue=1`. Neither run is claimed to have started merely from
submission or an advisory start estimate. Neither H100 job reached W&B or training startup.

At the 15:33 check both H100 jobs were still pending priority and had consumed no GPU time. A fresh
two-GPU RTX dry run estimated an immediate start on gr103 through `rtx6000_lzanna`; H200 capacity was
estimated several days away. RTX qualification 19656293 therefore checked the same model/container with
two ranks before moving production. It used the previously unused `qualification-forecast` configuration
and output directory; the explicit backward probes and completion validation still exercise all four tasks.
The production H100 pair remained queued through qualification, then was replaced after the successful
retry described below. An advisory dry-run start time is not evidence that an allocation has begun.

Following the updated async guidance, the new RTX callback `3936168a-78b7-4f95-8636-1d4deec41f97` uses
quiet polling by `/root/rtx_qualification_monitor`, with hourly reminders and durable parent fallback.
Its terminal event was handled and the registration retired. The earlier H100 callbacks were subsequently
retired when the pending pair was replaced.

The RTX run initialized both DDP ranks on NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, then failed
in the first explicit backward probe, before optimizer updates or validation. The asynchronous trace points
to skip-module Conv2d during activation-checkpoint recomputation; this does not establish the earliest
failing kernel or library root cause. Its allocated cost was 0.0328 RTX GPU-hours. The qualification used
effective batch 16 (4/rank × accumulation 2 × two ranks), while both production configs were rechecked
to retain effective batch 32 (accumulation 4).

A comparable [PR #915 diagnostic](https://github.com/m2lines/Samudra/pull/915#issuecomment-6072906494)
motivates an opt-in `native_pointwise` runtime option. It routes only 1×1 Conv2d calls through native
PyTorch with cuDNN disabled in that operation, preserving parameter/checkpoint shapes and all other
convolutions. Training and held-out forecast evaluation read the same config flag. Tests verify backend
scope, unchanged state/parameter identity, restored global flags, and output/gradient agreement on CPU.
The same-node RTX retry passed with this mitigation; it is not a claimed CUDA root-cause fix.

Retry 19656804 used immutable source `a4b90dfddcb11f45be0c4ab378915c916c0fd9aa`, the same gr103 node,
container, two-rank allocation and qualification recipe. Its configuration is
`configs/qualification-forecast-native1x1.yaml`, with `native_pointwise: true` and a distinct output name.
[Raw RTX qualification results](results/multitask-qualification-19656804.json) record all four finite backward
probes, three optimizer updates, and finite completion plus 360-day validation. Both ranks and two RTX PRO
6000 Blackwell GPUs were verified in logs. Peak allocated memory was 42.55 GB; elapsed time was 1:52,
exit 0, with no stderr, costing 0.0622 RTX GPU-hours. A short qualification does not establish 70-epoch
stability. Callback `5ec6985f-dc5b-464c-b5ef-b2207e04ee6d` delivered and was retired.

Both H100 jobs were freshly confirmed pending before cancellation. Their CPU notifications were cancelled
first, then the pending workloads; neither used GPU time. A full RTX dry-run request was accepted before
replacement. Production F/M jobs 19657151/19657152 use the exact successful RTX source and immutable
`configs/train-{forecast,multitask}-rtx.yaml`, with output directories `outputs/train-{forecast,multitask}-rtx/`.
Both enable `native_pointwise`, use batch 4/rank × accumulation 4 × two ranks = effective batch 32,
and preserve all other approved training settings, including 70 epochs, online W&B and annual validation.
Slurm read-back confirmed `rtx6000_lzanna`, two RTX GPUs, 16 CPUs, 192 GiB, `Requeue=1` and
`Comment=preemption=yes;requeue=true`. The five-minute submission delay allowed callback installation.

F callback: `3ddc6737-a24d-4751-8344-b7c8cb335704`, monitor `/root/rtx_qualification_monitor`.
M callback: `ca26508f-2b5d-4351-bc1f-bac024efe117`, monitor `/root/multitask_rtx_monitor`.
Both login authentication checks passed, and CPU `afterany` dependencies were verified. Quiet monitors
check startup once, then hourly. Subsequent startup findings and the replacement M callback are below. The within-wave hardware/backend recipe is matched;
comparison to historical C must disclose the backend, hardware and rank-count differences.


F startup was verified on gr103: two RTX PRO 6000 GPUs, world size 2, native pointwise backend and
effective batch 32. [Online W&B run 3uv0p502](https://wandb.ai/ocean_emulators/default/runs/3uv0p502)
is active; epoch 1 reached minibatch 62/352 with finite losses, about 3.30 seconds/batch and negligible
loader wait. This establishes startup, not full-epoch performance or scientific results.

M 19657152 failed on gr102 after 11:29, exit 1, costing 0.3828 allocated RTX GPU-hours. Both ranks
hit a 600-second timeout at the initial NCCL barrier (`init_distributed_mode`, first one-element
ALLREDUCE), before W&B/model/data/training initialization. This is distinct from the convolution error.
No checkpoints or metrics were produced; the empty checkpoint directory, config and startup/error logs
are preserved under `outputs/failed-19657152-train-multitask-rtx/`.

Torch rejected a dry-run request with `--exclude=gr102` and asked that the issue be reported to HPC;
no exclusion or equivalent placement workaround was used, and no external message was sent. An
unchanged retry 19657895 tests whether startup failure was transient; scheduler placement may again
choose gr102. Config, source, model settings and communication settings are unchanged. The original
M callback was retired. Replacement callback `43b3cabc-9d6b-4b4a-90fa-e9dac5d57c62` authenticated,
and CPU notification 19657907 has a verified `afterany` dependency. `/root/multitask_rtx_monitor`
checks bring-up after ten minutes, then hourly; another initial-barrier failure requires diagnosis before
further resubmission. F continues under its existing monitor and callback.

Torch held job 19648662 could not be released (`scontrol release` returned an unspecified error twice).
It and dependent notification 19648699 were cancelled before allocation; their callback was revoked.
The replacement uses a two-minute delayed start to attach monitoring. Beta held qualification 212521
was also cancelled without allocation after the user requested available Torch capacity.

## Environment and data verification

Container: `/scratch/jr7309/.apptainer-images/sd-seeds-base-0b5320248.sif`.
Its `uv.lock` and `pyproject.toml` match the source byte-for-byte; both are compared again at launch.
The installed x86 Rust extension imports successfully. PyTorch reports
`2.12.0a0+0291f960b6.nv26.04.48445190`. Qualification confirmed CUDA and real Rust-loader operation.
The production harness additionally forwards Slurm's GPU visibility through the clean container environment
and asserts that the visible GPU count equals the requested process count; training/model code is unchanged.

Torch data: `/scratch/jr7309/data/om4-v2026-09/om4_onedeg`.
Beta reference: `/projects/ny/lz1955/multiscale/jrusak/data/om4/v2026-09/om4_onedeg`.
[Fingerprint utility](fingerprint_data.py) checks metadata, static arrays including masks/coordinates,
all normalization chunks, and first/middle/last time chunks of each ocean array. Both hosts returned
identical hashes: 504 files / 67,652,886 bytes for `OM4.zarr`, and 303 files / 66,587 bytes for each
normalization store. This sampled comparison is not a full reread of every ocean field. The machine-readable
record is `data-sample-fingerprint.json` in the Torch campaign root. Training additionally saves the
resolved normalization, masks, coordinates, variable order, time splits and upstream attributes.

Local validation: 22 model/numerical tests passed, including initial C parity, mask leakage, requested-only
gradients, checkpointed backward equivalence and the opt-in 1×1 backend. Scoped pre-commit and shell syntax checks passed.
`evaluate_score.py --arm F|M` loads the new conditioning architecture for the same held-out forecast
protocol; final completion and continuous-rollout evaluation remain to be qualified before use.

## October 12, 18:07 Eastern: storage recovery

M retry 19657895 started successfully on gr106 with unchanged settings, two RTX GPUs and world size 2.
[W&B bfjwav3j](https://wandb.ai/ocean_emulators/default/runs/bfjwav3j) is online; finite training progressed
through epoch 5 into epoch 6. Passing startup on another allocation does not prove a gr102 hardware fault.

F 19657151 stopped near epoch 6 minibatch 330/352, after 1:58:53 (3.9628 allocated RTX GPU-hours),
with exit 1 and no Python/CUDA traceback. The latest completed checkpoints/annual validation are epoch 5.
Its CPU notification 19657177 also failed (`0:53`) without a Slurm output file; the hourly reminder
caught the terminal state. Slurm did not identify a root cancellation or provide GPU utilization evidence.

A direct scratch write failed with `Disk quota exceeded`; `myquota` reported the 5 TB scratch quota
at 99.97–100%, despite free capacity on the shared filesystem. This is an independently verified storage
failure, not conclusive proof of F's exit cause. `apptainer cache clean --force` removed 53.09 GiB of
regenerable download-cache content under `/scratch/jr7309/apptainer-cache`. The actual running SIF,
datasets and all checkpoints were retained. Writes initially still failed, then succeeded about 30 seconds
later; M's log writes also resumed, with finite epoch-6 progress. The quota report still showed 100% at
18:07, so available quota headroom is not yet verified and remains a monitoring concern.

F recovery 19664764 uses the unchanged immutable source/config and existing output directory. The
`preemptible` path automatically loads `saved_nets/ckpt.pt`; its ZIP structure is readable. Bring-up must
confirm epoch 6, model/EMA/optimizer/scheduler restoration and W&B run 3uv0p502 resumption. The
interrupted portion of epoch 6 will be repeated; no bitwise equivalence to uninterrupted training is claimed.
Callback `306f6f54-abc5-496a-9816-b3f2f0608220` authenticated, and CPU notification 19664888 has a
verified `afterany` dependency. `/root/rtx_qualification_monitor` checks startup after ten minutes, then
hourly, including storage and checkpoint advancement. The old F registration is retired; M's callback
and monitor remain active. This recovery changes no scientific settings or experimental scope.

## October 13, 15:15 Eastern: multitask training complete

M job 19657895 completed all 70 epochs with exit 0 after 22:56:44 on gr106 (45.8911 allocated RTX
GPU-hours). CPU notification 19657907 also completed; its delivered registration was retired.
Epoch-70 latest, numbered, EMA and best-validation checkpoint files were saved at 15:14. Terminal
[W&B bfjwav3j](https://wandb.ai/ocean_emulators/default/runs/bfjwav3j) sync confirmed five files; checkpoint
contents will be read back during evaluation preparation. The exposure log contains all 70 epochs and
197,120 contexts: 97,684 forecast (49.56%), 49,824 surface (25.28%), 24,476 variable (12.42%) and
25,136 gap (12.75%), consistent with the approved random mixture.

Final training-time annual validation is finite: normalized 360-day empirical CRPS 0.1632009 and
ensemble-mean RMSE 0.3535590. The selection score is 0.13045144, with best selection at epoch 70.
These are validation results, not the held-out paired comparison. F recovery 19664764 remains running
in epoch 70, minibatch 81/352 at 15:15; its completion callback and hourly fallback remain active.
Both arms progressed and saved checkpoints after the quota incident. The last successful quota check
at 14:48 showed 92.75% usage. No scientific comparison conclusion is drawn before F and paired
held-out evaluation are complete.
