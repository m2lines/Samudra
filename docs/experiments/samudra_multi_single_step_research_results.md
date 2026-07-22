<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# SamudraMulti single-step research results

## Scope

This note records execution of the single-step research
[plan](samudra_multi_single_step_research_plan.md), starting from the completed
plain-MSE [baseline](samudra_multi_single_step_mse.md). All promotion metrics are
unweighted normalized one-step MSE. The physical patch extent remains 3 degrees by
5 degrees. There is no residual prediction, autoregressive training, multi-step
target, or long-rollout evaluation in this work.

## A2 tiny fixed-set identity diagnostics

The first identity screen trained the unchanged SamudraMulti architecture to
reconstruct four fixed validation samples for 30 epochs, or 120 optimizer updates,
independently at each resolution. All runs used plain MSE, batch size one, one GPU,
four CPUs, seed 15, and immutable container
`26.05-b557702ae572882068626f29c8d01991ca992eac`.

| Resolution | Slurm | W&B | Allocation | Elapsed | Apptainer MaxRSS | Status |
|---|---:|---|---|---:|---:|---|
| 1 degree | `14321190` | [jacnt8bt](https://wandb.ai/ocean_emulators/default/runs/jacnt8bt) | 1 GPU, 4 CPU, 64 GiB | 21:19 | 6.5 GiB | Completed |
| 1/2 degree | `14321489` | [ta0qvznk](https://wandb.ai/ocean_emulators/default/runs/ta0qvznk) | 1 GPU, 4 CPU, 32 GiB | 5:15 | 8.7 GiB | Completed |
| 1/4 degree | `14321638` | [bj186l8z](https://wandb.ai/ocean_emulators/default/runs/bj186l8z) | 1 GPU, 4 CPU, 32 GiB | 6:01 | 16.7 GiB | Completed |

The 1-degree allocation includes the one-time OCI-to-SIF conversion. That build
peaked near the 64-GiB request, while its actual Apptainer training step used only
6.5 GiB host RSS. With the image cached, both higher-resolution diagnostics fit in
32 GiB. The final metrics are:

| Resolution | Best epoch | Mean MSE | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | Mean high-k power ratio | Mean seam ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 degree | 30 | 0.401007 | 0.114511 | 0.402367 | 0.538925 | 0.565896 | 0.121514 | 0.170374 | 1.18738 |
| 1/2 degree | 30 | 0.387499 | 0.107667 | 0.382336 | 0.527219 | 0.549712 | 0.097076 | 0.161547 | 1.23328 |
| 1/4 degree | 30 | 0.376430 | 0.095853 | 0.365348 | 0.519257 | 0.541126 | 0.106599 | 0.190746 | 1.37313 |

The ordinary MSE does not collapse as resolution increases, but none of the four-
sample tasks approaches identity. All three resolutions retain only 16 to 19% of
mean target high-wavenumber power. The aggregate patch-seam error ratio grows with
resolution, reaching 1.37 at 1/4 degree; the 1/4-degree variable ratios reach 1.77
for temperature and 2.23 for SSH. This is direct evidence that similar scalar MSEs
can hide smoothing and increasingly structured patch artifacts.

This screen is intentionally too small to make the final representation decision.
The next A2 stage repeats the diagnostic on 32 fixed samples before deciding which
bottleneck change, if any, is justified.

### Artifact checksums

| Resolution | Artifact | SHA-256 |
|---|---|---|
| 1 degree | resolved `config.yaml` | `ad351af16b6981872243c8635a3941a0f412af9203080a7dd002420fb7ae8c92` |
| 1 degree | `identity_metrics.json` | `85190121616495027aa442e38b8654d0d0fda0d3ea1a8ac005ed7a8f893c57cf` |
| 1 degree | `identity_spectra.pt` | `f1b036427d757777a48a42153afb2e6c0820771d7811ac1dfdb9a330a6da8f96` |
| 1 degree | `saved_nets/ckpt.pt` | `f9e48b8d21aa7b0ef79cbc626dd0bbc6ab0649da070422c897d4f3270b3eb6c9` |
| 1/2 degree | resolved `config.yaml` | `fa0bef490b3d090002b3b8739f63fed7e0dace4e125840e20a4a2eebfe5d18e7` |
| 1/2 degree | `identity_metrics.json` | `e7f56b724db4ea7ea187e57b4d38172e82c4033b14a20aa47172034b90db8f42` |
| 1/2 degree | `identity_spectra.pt` | `29098c434f9b8ad51878f43974c487f997bd2a02a45b2834cd236d9e5b3c636a` |
| 1/2 degree | `saved_nets/ckpt.pt` | `fda1054d1931f6196a2f77c32627efee336813a1d45f80bfe07abbf02c80fadf` |
| 1/4 degree | resolved `config.yaml` | `407c11f5e1933c1b0b55092cf83496dcf4c4f2f72ae28c2aa4819f5c3ad96c7b` |
| 1/4 degree | `identity_metrics.json` | `8017a3a0c79cfd41937befd86a8e8cf739dd4e30692a1fde5581a76a0f313ee1` |
| 1/4 degree | `identity_spectra.pt` | `ebd6bcdb4eca1ab239bb1210333f309eef9ed06a2c6150e14a5c3fe5c7ee8923` |
| 1/4 degree | `saved_nets/ckpt.pt` | `9dee0f3d249de7a8733ae62cf91c3dd98530e953ff568b9b4bdc6c44b761f1bf` |

The run directories are under `/scratch/jr7309/runs/` with names matching the W&B
runs above. Recreate the summary table from copied metric files with
`scripts/summarize_identity_runs.py`.

## A3 stratified proxy dataset

The training data interface now selects arbitrary valid forecast windows without
slicing away the adjacent input/target timestamps each window needs. Selection is
deterministic and round-robins across `(decade, season)` strata before sorting the
chosen indices for efficient reads. The pinned proxy selection uses 512 of 2,829
valid one-step windows, seed `20260719`, and spans 1975 through 2013. Its 20 strata
contain either 25 or 26 examples each: all four seasons in the 1970s, 1980s, and
1990s receive 26, while all four seasons in the 2000s and 2010s receive 25. The
validation interval remains fixed at 2013-10-05 through 2014-10-05.

`tests/test_stratified_samples.py` verifies deterministic balance, different-seed
behavior, immutability, and contiguous window semantics. The screening configs are
`train_1deg_mse_stratified_proxy.yaml` and
`train_1deg_mse_stratified_updates_proxy.yaml`; the latter is the isolated
optimizer-update scheduler candidate.

The two-seed scheduler screen was submitted as a serial dependency chain on the
validated `c79c302f` image. Each run uses one RTX6000 GPU, four CPUs, 40 GiB of host
memory, plain MSE, one-step targets, batch size two with 16-step gradient
accumulation (effective global batch 32), five-window decoder chunks, legacy `all`
checkpointing, and no `wandb.watch` model logging.

| Schedule | Seed | Slurm | State at launch |
|---|---:|---:|---|
| Epoch-based cosine | 15 | `14333937` | Running |
| Epoch-based cosine | 16 | `14333938` | After `14333937` |
| Update-based cosine | 15 | `14333939` | After `14333938` |
| Update-based cosine | 16 | `14333940` | After `14333939` |
| v2 epoch-based control | 15 | `14334736` | After `14333940` |
| v2 epoch-based control | 16 | `14334740` | After `14334736` |

This screen will select the scheduling control before normalization, receptive-field,
or representation ablations. Three-seed finalist calibration remains pending that
funnel; no candidate will be promoted to full data unless its proxy all-channel MSE
is at most `0.08575`.

The first stratified run exposed an A1 diagnostic bug rather than a distribution
shift. Its legacy `val/mean/loss` was `0.87351` at epoch 3, but the independently
recomputed
`val/resolution/180x360/unweighted_normalized_mse/mean/loss` was `0.44290`, nearly
identical to the original contiguous proxy's `0.44238` at the same epoch; training
MSEs were likewise `0.44748` and `0.44904`. The overall and resolution-specific
aggregators had retained aliases to the first batch's loss tensors, so their later
in-place additions double-counted every batch after the first in the overall value.

`TrainAggregator` now clones its initial accumulation state, and a regression test
checks that two batches produce identical overall and scale-specific means. The run
summarizer prefers the explicit unweighted diagnostic when available and falls back
to legacy plain-MSE keys for older controls. Consequently the in-flight `c79c302f`
runs remain valid for screening through their independently recomputed unweighted
keys; their corrupted legacy top-level value is not used for selection or gating.
The matched stratified v2 controls remain necessary to demonstrate ranking
consistency, and the final decision will use completed two-seed results.

Through epoch 6, the corrected stratified control also reproduces the original
contiguous SamudraMulti curve closely. The mean absolute difference across the six
paired validation epochs is `0.00141`, with a maximum of `0.00490`; epochs 1--4
differ by at most `0.00052`. This supplies an initial representative-control
calibration for the arbitrary-index interface. The completed two-seed result below
quantifies terminal seed variance; the matched v2 ordering is still required before
choosing the proxy for the B funnel.

The same run confirms the selected A5 path at normal proxy fidelity. Its latest
completed epoch used about `283` training seconds for 512 samples, or `1.81`
samples/second on one GPU. The historical four-GPU proxy took about `120` seconds
per epoch (`4.27` samples/second total, approximately `1.07` per GPU). Thus the
portable path is about `1.7x` more throughput-efficient per GPU while reproducing
the control curve and reducing the allocation from four GPUs to one.

Both epoch-scheduled controls completed all 12 epochs with exit code zero. Seed 15
used `1:05:20` of training time and `1:06:00` of Slurm allocation time; seed 16
used `1:17:42` and `1:18:15`, respectively. Each requested one GPU, four CPUs, and
40 GiB. Apptainer MaxRSS was 13.34 GiB for seed 15 and 13.39 GiB for seed 16;
their GPU peaks were about 25.9 GiB. Their validation-selected epoch 12 results
are:

| Run | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|---:|
| Original contiguous control | 0.385084 | 0.098353 | 0.353942 | 0.536955 | 0.566338 | 0.083592 |
| [Stratified seed 15](https://wandb.ai/ocean_emulators/default/runs/ec0f03n4) | 0.384732 | 0.098102 | 0.351683 | 0.537368 | 0.566746 | 0.081088 |
| [Stratified seed 16](https://wandb.ai/ocean_emulators/default/runs/g4m85ppt) | 0.383026 | 0.090220 | 0.355029 | 0.536380 | 0.566804 | 0.071868 |

The seed-15 stratified result is `0.000351` lower overall than the original
contiguous control and reproduces every variable group closely. Across the two
stratified seeds, overall MSE is `0.383879` mean, `0.001206` sample standard
deviation, and `0.001706` range. This validates the arbitrary-index proxy for this
representative control and quantifies a small screening-scale seed effect. The
runs' experiment logs report aliased legacy `val/mean/loss` values near twice the
correct result; those values are invalid and retained only as evidence of the
diagnosed aggregation bug. Selection uses the independently recomputed unweighted
results above.

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| Resolved `config.yaml` | `84ad2325ed71fb3495d8fe5d20e4824509969033b5613384010a9c803e2558b9` | 2,562 |
| `saved_nets/best_validation_ckpt.pt` | `94840349b7540b087cce298c8b26f14bea0a9fb11e748b0737df05bb1df08a98` | 1,215,668,023 |
| `saved_nets/ckpt.pt` | `01d34b68a45023ee6187bd1834b8bd6d3b64786ec6633af101e576bd92257735` | 1,215,668,023 |

Seed 16 is pinned separately by:

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| Resolved `config.yaml` | `3fe3ee2a11ec50e789059771bf07993539e462a0c1c68a681601540964da6b34` | 2,562 |
| `saved_nets/best_validation_ckpt.pt` | `3fcf577a88c973b371d24f24ea0e94ca3ee2e223ee5dde534e87f399afe236ea` | 1,215,668,023 |
| `saved_nets/ckpt.pt` | `2fb528c7ee5b844f92b4f97f66b8d230ebd5e47064869804f67fab8f3a8be291` | 1,215,668,023 |

### Optimizer-update scheduling control

The update-scheduled controls also completed all 12 epochs with exit code zero.
They made the same 192 optimizer updates as the epoch controls but evaluated the
cosine schedule against the planned full-data budget of 6,160 updates. Their final
learning rate was therefore about `0.000599`, preserving sample/update semantics
instead of compressing a full training schedule into the small proxy.

| Run | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|---:|
| [Update schedule, seed 15](https://wandb.ai/ocean_emulators/default/runs/zy4e7qcc) | 0.385737 | 0.099912 | 0.355364 | 0.536428 | 0.566951 | 0.080649 |
| [Update schedule, seed 16](https://wandb.ai/ocean_emulators/default/runs/5ol8hchc) | 0.377733 | 0.089240 | 0.335379 | 0.536166 | 0.566817 | 0.073219 |

Their overall MSE is `0.381735` mean, `0.005660` sample standard deviation, and
`0.008004` range. The mean is `0.002144` lower than the epoch-scheduled mean, while
the observed two-seed variance is larger. This is quality-neutral at screening
fidelity and retains the desired update/sample-based semantics, so the B funnel
uses the update schedule. The v2 controls below still determine whether the proxy
preserves the expected model ranking.

| Seed | Slurm elapsed | Apptainer MaxRSS | Resolved config SHA-256 | Best-checkpoint SHA-256 | Latest-checkpoint SHA-256 |
|---:|---:|---:|---|---|---|
| 15 | 1:07:03 | 12.39 GiB | `bc971b8ed87529b660cb8cca23512b7db6a88f11ebbb99f7812ae542f7a987de` | `1c3bdddd7d1c1d588c5084d9c6beae3ced271018fa3429222c437e2d0907cc21` | `d7d95ae9f315ea81d3ff7f1c9ab257b2deb61d95209da41d240d8024dae4c934` |
| 16 | 1:04:55 | 11.81 GiB | `276d3f88487edcb34ee9e03f7bd41745f2da6b77018efaf74c8978368696fb17` | `4b29231dfcf85338b193486825f2e0d0592e23846b543bca55692b6c427c9e5f` | `0e70bbff67c5620c7275b0d355fd86fbeca883dfed17adad15fde00afc5f2519` |

Each resolved config is 2,577 bytes and each checkpoint is 1,215,668,023
bytes. Both runs used one GPU, four CPUs, and a 40-GiB host-memory request.

### Matched v2 ranking calibration

The matched stratified v2 controls completed cleanly and reproduce the original v2
proxy result of `0.04287`. Their validation-selected epoch 12 results are:

| Run | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|---:|
| [v2 seed 15](https://wandb.ai/ocean_emulators/default/runs/c9417mhl) | 0.042446 | 0.006899 | 0.006900 | 0.045959 | 0.063859 | 0.007910 |
| [v2 seed 16](https://wandb.ai/ocean_emulators/default/runs/eimcfxwz) | 0.042334 | 0.006652 | 0.006760 | 0.045951 | 0.063713 | 0.007754 |

Overall MSE is `0.042390` mean, `0.000079` sample standard deviation, and
`0.000112` range. The selected SamudraMulti update-scheduled control is 9.00 times
worse by the two-seed means, so the proxy preserves both the expected ordering and
the approximate historical gap. A3 is therefore calibrated for the isolated B
funnel.

| Seed | Slurm elapsed | Apptainer MaxRSS | Resolved config SHA-256 | Best-checkpoint SHA-256 | Latest-checkpoint SHA-256 |
|---:|---:|---:|---|---|---|
| 15 | 0:53:23 | 11.99 GiB | `6cc3e24935658bd491f53f2134fb643d0819f2c35866d795a1646d3a987e59f2` | `7d3c992722a54d83e371491acdb7983b1cd54ad7f15d7e7286a6332ec9cae755` | `2b61d9c8a6b5707d503a0e06b6e42bd825f523f1db7af730443e7876b3887caf` |
| 16 | 0:50:40 | 11.96 GiB | `f913a56ade4bde1117c1b54c56b28a38a543f9773a2f8602664657ae330ff38e` | `b86780a51a0971e018d0f0445e3dee9483eaa73227ddf844d1a54731517e10bc` | `9d831d7800dbac7b1d002b966eecd325f8e903272b8a8c82a36b14016bd6bbca` |

Each resolved config is 2,344 bytes and each checkpoint is 1,344,153,659
bytes. Both runs used one GPU, four CPUs, and a 40-GiB host-memory request.

## A5 decoder, checkpoint, and logging microbenchmarks

The A5 screen isolates three avoidable costs on the same four-sample, 30-epoch
1-degree identity task. Steady-state time is the mean over epochs 2 through 30;
epoch 1 is excluded because it includes compilation and warm-up. Each cached-image
run used one RTX6000 GPU, four CPUs, and 32 GiB unless noted. The immutable image is
`26.05-c79c302fdaf627008041f09d68de8894e167c394`.

| Change from the original path | Slurm | W&B | Mean seconds/epoch | Speedup | Final MSE | Apptainer MaxRSS |
|---|---:|---|---:|---:|---:|---:|
| None: one decoder window/call, checkpoint `all`, W&B watch `all` | `14322261` | [yo9gfyiq](https://wandb.ai/ocean_emulators/default/runs/yo9gfyiq) | 6.232 | 1.00x | 0.397866 | 6.4 GiB |
| Disable W&B model watching | `14322981` | [kyr7ksyv](https://wandb.ai/ocean_emulators/default/runs/kyr7ksyv) | 6.058 | 1.03x | 0.403311 | 6.0 GiB |
| Batch five decoder windows/call; watch off | `14322986` | [8jzfpul9](https://wandb.ai/ocean_emulators/default/runs/8jzfpul9) | 3.243 | 1.92x | 0.404222 | 5.7 GiB |
| Batch all decoder windows/call; watch off | `14322987` | [95wgs76b](https://wandb.ai/ocean_emulators/default/runs/95wgs76b) | 2.604 | 2.39x | 0.401563 | 5.6 GiB |
| Checkpoint `simple`; one window/call; watch off | `14322991` | [ljnr7le0](https://wandb.ai/ocean_emulators/default/runs/ljnr7le0) | 1.927 | 3.23x | 0.398786 | 6.4 GiB |
| No checkpointing; one window/call; watch off | `14322994` | [cgnwj21e](https://wandb.ai/ocean_emulators/default/runs/cgnwj21e) | 1.874 | 3.33x | 0.404904 | 5.6 GiB |
| Batch all windows; checkpoint `simple`; watch off | `14323333` | [c1ckf0rw](https://wandb.ai/ocean_emulators/default/runs/c1ckf0rw) | 1.861 | 3.35x | 0.400319 | 5.5 GiB |
| Batch all windows; no checkpointing; watch off | `14323670` | [vtp3n31g](https://wandb.ai/ocean_emulators/default/runs/vtp3n31g) | 1.648 | 3.78x | 0.403360 | 5.6 GiB |

The largest single saving comes from removing the outer `checkpointing: all`
wrapper. The underlying processor keeps its intended block-level checkpointing in
`simple` mode. Once that nested wrapper is gone, batching all 30 decoder windows
adds only about 3% on this tiny task; it remains worthwhile because the decoder
outputs are exactly equivalent and larger forecast batches amortize launches more
effectively. Disabling `wandb.watch` saves a smaller but repeatable 3%.
Removing the remaining processor checkpointing saves another 11% relative to the
fully batched `simple` path, subject to the real-forecast GPU-memory check below.

`tests/test_decoder.py` compares sequential and fully batched decoder outputs and
passes exactly. The focused decoder/config suite reports 23 passing tests, and the
stratified-sampling, scheduler, and spatial-diagnostic suites report 15 passing
tests. A broader local run reached 135 passes before being stopped; its 12 setup
errors were all caused by the development environment lacking the optional
`flash_perceiver` package, not by these changes. The x86 image CI, including GPU
tests, passed at the pinned commit.

The 0.398 to 0.405 final-MSE spread is small relative to the identity failure and
reflects flash-kernel/training-order nondeterminism across independent runs. There
is no systematic quality penalty associated with the faster paths.

The simple-checkpoint combination is pinned by:

| Artifact | SHA-256 |
|---|---|
| Resolved `config.yaml` | `1d296e9db550da1226bcc0783c24958d7819a91861a42fdd801749958801f874` |
| `identity_metrics.json` | `7145838d1d9168de4c5c899b8b1754762780d1b4f8f1137bb8560ef6992cec90` |
| `saved_nets/ckpt.pt` | `79ad419e3126f950610f24a4c84936b6dc9880de1277c4fe1ee67756cfe585f6` |

The fully uncheckpointed comparison is pinned by:

| Artifact | SHA-256 |
|---|---|
| Resolved `config.yaml` | `106a2254041333b029ca9e343baec4e4e1d6e47fd77e9ca47cd7febe1fa0dad2` |
| `identity_metrics.json` | `ccb712a5f5275598dab792318503ac2dc353d6dd5630d56e95b9ec5251ebd856` |
| `saved_nets/ckpt.pt` | `4ea4a019cddbeafe4f3d6de5f75ce21c8acb039bb75766a64208670a3f4217a3` |

### Normal forecast-path benchmark

The identity task understates activation memory, so two one-epoch forecast runs
used the actual stratified 512-window proxy, plain MSE, one step, batch size two,
16-step gradient accumulation, and one GPU. Both therefore processed 512 samples
in 256 microbatches and made 16 optimizer updates, preserving effective global
batch 32. Both disabled `wandb.watch` and used `checkpointing: simple`.

| Decoder chunk | Slurm | W&B | Allocation | Train seconds | Mean seconds/microbatch | Mean samples/s | Mean data wait | Peak GPU | Apptainer MaxRSS | One-step MSE |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| All 30 windows | `14323887` | [9c39cn6s](https://wandb.ai/ocean_emulators/default/runs/9c39cn6s) | 1 GPU, 4 CPU, 32 GiB | 209.34 | 0.818 | 2.736 | 0.275 s | 74.49 GiB | 31.96 GiB | 0.51819 |
| Five windows | `14324068` | [qkk1e7rl](https://wandb.ai/ocean_emulators/default/runs/qkk1e7rl) | 1 GPU, 4 CPU, 40 GiB | 173.60 | 0.678 | 3.334 | 0.119 s | 74.49 GiB | 17.50 GiB | 0.51819 |
| Five windows, first selective design | `14324576` | [dyx8cje0](https://wandb.ai/ocean_emulators/default/runs/dyx8cje0) | 1 GPU, 4 CPU, 64 GiB | 213.51 | 0.834 | 2.673 | 0.098 s | 74.57 GiB | 22.91 GiB | 0.51820 |
| Five windows, legacy `all` | `14333437` | [o6dmocpm](https://wandb.ai/ocean_emulators/default/runs/o6dmocpm) | 1 GPU, 4 CPU, 32 GiB | 302.03 | 1.179 | 1.811 | 0.002 s | 25.86 GiB | 31.96 GiB | 0.51819 |

Chunking does not reduce GPU activation memory under `simple`: all decoder calls
remain in the autograd graph until backward, so both paths peak at about 74.5 GiB.
The five-window run's apparent throughput advantage is mostly a warm-cache effect;
subtracting loader wait leaves about 0.54 to 0.56 seconds of compute per microbatch
in both runs. Data wait is now measurable, at 18 to 34% of observed iteration time,
but it is not evidence for repacking the Zarr store: the second run's lower wait and
the successful four-CPU allocation show that caching/prefetch overlap matters
first. No data-layout change is justified by this profile.

The 74.5-GiB peak makes both `simple` variants non-portable to 48-GiB RTX6000 nodes
and rules out a full forecast trial with checkpointing disabled. The first
`selective` implementation in `d9002299` confirmed that cheap-layer processor
checkpointing was insufficient: its peak remained 74.57 GiB. Commit `fb0c173c`
corrects the design to checkpoint individual processor layers plus the encoder and
decoder without a redundant outer processor wrapper. Its focused formatting,
typing, schema, and 25-test suite passes, but a pinned image could not be built
because GitHub's EC2 runner service repeatedly failed before repository code ran.
It remains an unpromoted optimization, not part of the baseline.

The robust selection is therefore five-window decoder chunks, legacy
`checkpointing: all`, and no `wandb.watch` on validated image `c79c302f`. It peaks
at 25.86 GiB per GPU, fits one GPU/four CPUs/32 GiB, and preserves the exact
one-epoch validation result. At 1.179 seconds per microbatch, it is 1.62 times
faster than the original 1.909-second forecast path. Its near-zero data wait on
`gr102` also confirms that no data-layout change is warranted. A3 will use a
40-GiB host request for margin while retaining only four CPUs.

The normal-path artifacts are pinned by:

| Run | Artifact | SHA-256 |
|---|---|---|
| All windows | Resolved `config.yaml` | `f89eee3e955f6be6f0b860b2d9c08ad0360acca51239a8f958717c52d450425a` |
| All windows | `saved_nets/ckpt.pt` | `d6834b15171468df4d637911423d2fdf9c8be627efc480ead341cc078649b44b` |
| Five windows | Resolved `config.yaml` | `bf8f6959d50ce3bc4cd9d304cbf6e5a9d7aa3e148152c9804d85801c3e8be04c` |
| Five windows | `saved_nets/ckpt.pt` | `266d407e5b64a0be781e5586655c5e14b6c7244f8512a841759d5fa5d263ca08` |
| First selective design | Resolved `config.yaml` | `bd473e5b6819b7f1ba90f429367577e2012e1d591a1b383097856381eaab31c3` |
| First selective design | `saved_nets/ckpt.pt` | `1a15347cd7ecef15e1b78e7bc872a79f599c06e265774d3562109705c85c98fa` |
| Selected legacy `all` | Resolved `config.yaml` | `cf5bb33c84a08b05dd850125635f5804daf36961410a101796b2b27f35f16848` |
| Selected legacy `all` | `saved_nets/ckpt.pt` | `0b274c79e3b7deeabb1cbd1ba1bdd11f87bd9eed4dbd76d139fc1b00099fbb3c` |

## A2 32-sample identity diagnostics

The higher-fidelity identity test used 32 fixed samples for 20 epochs, or 640
optimizer updates, with the same seed, model, loss, and container. All three cached-
image runs fit in the reduced one-GPU, four-CPU, 32-GiB allocation:

| Resolution | Slurm | W&B | Elapsed | Apptainer MaxRSS | Status |
|---|---:|---|---:|---:|---|
| 1 degree | `14321970` | [1ufunem3](https://wandb.ai/ocean_emulators/default/runs/1ufunem3) | 20:19 | 6.4 GiB | Completed |
| 1/2 degree | `14322251` | [14qlcx2z](https://wandb.ai/ocean_emulators/default/runs/14qlcx2z) | 20:59 | 9.4 GiB | Completed |
| 1/4 degree | `14322252` | [qmnbg2q0](https://wandb.ai/ocean_emulators/default/runs/qmnbg2q0) | 23:39 | 22.4 GiB | Completed |

| Resolution | Mean MSE | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | Mean high-k power ratio | Mean seam ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 degree | 0.324229 | 0.057409 | 0.176728 | 0.509057 | 0.566564 | 0.060717 | 0.259720 | 1.15737 |
| 1/2 degree | 0.315166 | 0.053396 | 0.169834 | 0.497860 | 0.552674 | 0.045485 | 0.269957 | 1.19612 |
| 1/4 degree | 0.317481 | 0.056496 | 0.178394 | 0.504786 | 0.543059 | 0.059539 | 0.264092 | 1.38213 |

The larger test reaches the same conclusion more clearly. Thermohaline fields and
SSH learn part of the identity map, while velocity barely improves. Final
high-wavenumber power ratios by variable are:

| Resolution | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|
| 1 degree | 0.643921 | 0.351492 | 0.010405 | 0.000381 | 0.880622 |
| 1/2 degree | 0.660956 | 0.339476 | 0.035433 | 0.000873 | 1.088690 |
| 1/4 degree | 0.630756 | 0.294928 | 0.089749 | 0.002675 | 0.991035 |

Thus the model reconstructs only 1.0 to 9.0% of zonal-velocity and 0.04 to 0.27%
of meridional-velocity high-wavenumber power. The 1/4-degree seam ratios are 1.66
for temperature and salinity and 2.35 for SSH. The close all-channel MSE values do
not imply resolution-invariant fidelity; they conceal a severe velocity-information
bottleneck and increasingly structured fine-grid seams.

This evidence justifies a representation-capacity ablation after the isolated
optimization, normalization, and receptive-field controls. It does not justify
adding higher-resolution forecast training: the 1-degree forecast-quality gate is
still closed.

### 32-sample artifact checksums

| Resolution | Artifact | SHA-256 |
|---|---|---|
| 1 degree | resolved `config.yaml` | `55c1168b6fafadf44f787f8b78a84d6e8d1b63be0abb669c67e5bdd90ae50ebe` |
| 1 degree | `identity_metrics.json` | `a41938f5160f4c757218604f4b376ea401b17b25fbfab5603e7a700daa4d9c8c` |
| 1 degree | `identity_spectra.pt` | `03565c8a9bbd93b5224bedba263bc5a7b6b67fe52df986d8ad965ad4caa4e99f` |
| 1 degree | `saved_nets/ckpt.pt` | `b47e84c0b93dc196ef776c6ab5180e08be08fc7fc83073a17f5051ea17fd2da3` |
| 1/2 degree | resolved `config.yaml` | `a8875c17aacbaaceaa3ebea7bb967764c65faed03b6d95703613aa90ed78fb92` |
| 1/2 degree | `identity_metrics.json` | `c264d5d4523f9310a036ac203861fe6425b2c1445f12cc75095943b62b8ac1a0` |
| 1/2 degree | `identity_spectra.pt` | `1cfdc362b38d602811105bf1056cd408169c81ce58881d38e4801ba1306ea0fc` |
| 1/2 degree | `saved_nets/ckpt.pt` | `d0bfcf7460e8c13c18b29d7dd02772a6bdaa056bd6dec1b9f8f2e6e27fbfa013` |
| 1/4 degree | resolved `config.yaml` | `fdcea235d6d080cdb0854df455f0812f6938c0f46e01d3f8c07379b8e911e457` |
| 1/4 degree | `identity_metrics.json` | `a91dc665db5f2332a947c54c5d9ede579204e63323e116497acde52bd9e80c86` |
| 1/4 degree | `identity_spectra.pt` | `3c3f478fb95d191a7090f96bc520840c9a9ffa66f195f9b6db8c43ced0d3c1f3` |
| 1/4 degree | `saved_nets/ckpt.pt` | `0c92547f36bdfe5cefa24017dba53ed346f5a0630671ef37b42b85d0438b533c` |

## Torch container bring-up findings

The initial bring-up exposed three image-staging constraints before any model batch
ran: 32 GiB was insufficient for the one-time Docker-to-SIF conversion, root-backed
`/tmp` lacked enough disk space, and shared scratch does not support the xattr
operations Apptainer needs. Torch provides a large per-job XFS directory through
`SLURM_TMPDIR`; using it allowed the pinned image to build successfully. The training
harness now prefers that path automatically, validates `REPO_DIR`, and creates
configured cache/temp directories. The Torch skill and repository guide record the
same behavior.

## B1 forcing-input decision

No `hfds_anomalies` ablation is scheduled in this funnel. The field is derived rather
than stored in the physical Zarr boundary planes, and `RustDataBackend` explicitly
rejects derived boundary variables before opening the dataset. This behavior is
covered by `test_rust_loading_rejects_derived_boundary_variables_before_open` and is
also recorded in `docs/rust-data-loader-plan.md`. Implementing derived-field support
would be a separate loader change; until that exists, all matched comparisons retain
the physical `tauuo`, `tauvo`, and `hfds` inputs selected by `tau_hfds`.

## B4 normalization and receptive-field screen

The first B quality screen changes only the processor normalization from the
selected InstanceNorm control. LayerNorm, GroupNorm, and BatchNorm overrides were
validated through `TrainConfig`; all retain plain MSE, one-step targets, absolute
prediction, the fixed 3-degree by 5-degree physical patch, update-based scheduling,
effective global batch 32, and the selected A5 execution path. Seed 15 runs in a
serial one-GPU chain:

| Normalization | Slurm | W&B | State | All-channel MSE |
|---|---:|---|---|---:|
| InstanceNorm control | `14333939` | [zy4e7qcc](https://wandb.ai/ocean_emulators/default/runs/zy4e7qcc) | Completed | 0.385737 |
| Channel LayerNorm | `14376739` | [6j9pmvkz](https://wandb.ai/ocean_emulators/default/runs/6j9pmvkz) | Completed | 0.388022 |
| GroupNorm | `14376740` | [rdxqukj0](https://wandb.ai/ocean_emulators/default/runs/rdxqukj0) | Completed | 0.382284 |
| BatchNorm | `14376741` | [2de0ng9c](https://wandb.ai/ocean_emulators/default/runs/2de0ng9c) | Completed | 0.381857 |

Each job requests one RTX6000 GPU, four CPUs, and 40 GiB for at most two hours on
the immutable `c79c302f` image. Results will be compared with the two-seed
InstanceNorm control above before changing dilation or representation capacity.

LayerNorm completed all 12 epochs with exit code zero in `1:03:45`. Its explicit
unweighted one-degree MSE was `0.388022`: temperature `0.099937`, salinity
`0.360853`, zonal velocity `0.539586`, meridional velocity `0.566932`, and SSH
`0.086209`. This is `0.002285` (0.59%) worse than the paired seed-15 InstanceNorm
control, so LayerNorm does not displace the control. Peak GPU allocation was about
26.5 GiB and the Apptainer step MaxRSS was 12.38 GiB.

| LayerNorm artifact | SHA-256 | Bytes |
|---|---|---:|
| Resolved `config.yaml` | `0f4f11eff302eaa829ff3e00fb2022f6cdc4d23f48ec00bf72fca70754231e83` | 2,577 |
| `saved_nets/best_validation_ckpt.pt` | `5189272c505888d83618bf1d041258eef0736429f22c40a64e2cf616ae50849f` | 1,216,083,779 |
| `saved_nets/ckpt.pt` | `76680a179d9c06fd820ddec72c068ee1dc11c4e6f91273ce8051e9e96087dcf4` | 1,216,083,779 |
| `saved_nets/ema_ckpt.pt` | `62d5fb7d8518c5ac1c52b12d5258dd8315cb7e295a852a1df67dd5de472ac8b9` | 912,077,015 |

GroupNorm and BatchNorm also completed all 12 epochs with exit code zero in
`1:03:10` and `1:03:38`, respectively. Their terminal explicit unweighted metrics
are:

| Normalization | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | All |
|---|---:|---:|---:|---:|---:|---:|
| InstanceNorm control | 0.099912 | 0.355364 | 0.536428 | 0.566951 | 0.080649 | 0.385737 |
| Channel LayerNorm | 0.099937 | 0.360853 | 0.539586 | 0.566932 | 0.086209 | 0.388022 |
| GroupNorm | 0.097157 | 0.344761 | 0.535734 | 0.567041 | 0.069605 | 0.382284 |
| BatchNorm | 0.097506 | 0.342987 | 0.535487 | 0.566356 | 0.081974 | 0.381857 |

BatchNorm is the best paired seed-15 result, improving all-channel MSE by
`0.003880` (1.01%) relative to InstanceNorm. GroupNorm is only `0.000427` worse
overall and gives the strongest SSH result. These differences are smaller than the
two-seed update-control range (`0.008004`), so this is a screening choice rather
than evidence that BatchNorm is universally superior. The isolated receptive-field
ablation therefore carries BatchNorm forward while keeping every other control
fixed; finalist replication will determine whether the combined candidate is
stable.

| Normalization | Resolved config SHA-256 | Best-checkpoint SHA-256 | Latest-checkpoint SHA-256 | EMA-checkpoint SHA-256 |
|---|---|---|---|---|
| GroupNorm | `55fe113b2dfddc8fd76788888b00f240c1b6adfbe1821c894d4502e84c0be152` | `ce3cc873ba4be0dc9f8bab01b78fa70b11da9af528ec4c7a304f1b2f8985cb1f` | `9fb07753780404aca4c75c5959cf71b400be885b1efecb940268b27dc2209bca` | `1669810f57f817c163e45a126b0c7110442ebb0620248e4035308669aabe9ee7` |
| BatchNorm | `1674e600beb74540ee64601428cbc818a7d73fd920d4eb6b19486d313e71c9f0` | `0ec49e7b051d446086d232b97320e12a3082a5344937f03b2d3427241c651805` | `696d573238e85f789321c4ebb204f4c7980b3a3b40a924e181e3792fedb9ce22` | `8be003497e46a586194475287d1aa393dd0a10ff8f0fff9ad7525c97730347f4` |

GroupNorm's resolved config is 2,577 bytes, its best/latest checkpoints are
1,216,079,747 bytes, and its EMA checkpoint is 912,073,879 bytes. BatchNorm's
corresponding sizes are 2,577, 1,216,186,305, and 912,180,331 bytes. Their
Apptainer-step MaxRSS values were 12.29 and 12.41 GiB; both peaked near 26.5 GiB
of GPU memory.

### Receptive-field ablation

The isolated receptive-field candidate changes the selected BatchNorm processor
dilation from `[1,1,1]` to `[1,2,4]`. Config validation confirms that Slurm job
`14380979` ([j112axvq](https://wandb.ai/ocean_emulators/default/runs/j112axvq))
otherwise retains the same seed-15 stratified proxy, plain MSE,
one-step absolute prediction, effective global batch 32, update-based scheduler,
five-window decoder batching, fixed 3-degree by 5-degree patch extent, and
immutable `c79c302f` container. It requests one RTX6000 GPU, four CPUs, and 40 GiB
for at most two hours. The proxy gate remains `0.08575`; no full-data or
higher-resolution forecast run is authorized by this launch.

User-authorized parallel capacity expanded this into a two-seed 2-by-2 factorial
screen while preserving one GPU per run and effective global batch 32. Width 128,
dilation `[1,1,1]`, seed 15 is the completed BatchNorm control (`14376741`), and
width 128, dilation `[1,2,4]`, seed 15 is `14380979`. The remaining cells are:

| Embedding width | Dilation | Seed | Slurm | W&B |
|---:|---|---:|---:|---|
| 128 | `[1,1,1]` | 16 | `14381097` | [tq2ijc44](https://wandb.ai/ocean_emulators/default/runs/tq2ijc44) |
| 128 | `[1,2,4]` | 16 | `14381098` | [7exmk0d0](https://wandb.ai/ocean_emulators/default/runs/7exmk0d0) |
| 256 | `[1,1,1]` | 15 | `14381093` | [8hld5gtb](https://wandb.ai/ocean_emulators/default/runs/8hld5gtb) |
| 256 | `[1,1,1]` | 16 | `14381096` | [bgmibtof](https://wandb.ai/ocean_emulators/default/runs/bgmibtof) |
| 256 | `[1,2,4]` | 15 | `14381099` | [9lhfpsma](https://wandb.ai/ocean_emulators/default/runs/9lhfpsma) |
| 256 | `[1,2,4]` | 16 | `14381095` | [7l389bk1](https://wandb.ai/ocean_emulators/default/runs/7l389bk1) |

All six new jobs use the same one-GPU, four-CPU, 40-GiB allocation and immutable
container. This layout separates seed variance, receptive field, capacity, and
their interaction without introducing a multi-GPU numerical or BatchNorm change.

All seven jobs completed with exit code zero in 64 to 67 minutes. Terminal
explicit unweighted one-degree metrics are:

| Width | Dilation | Seed | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | All |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 128 | `[1,1,1]` | 15 | 0.097506 | 0.342987 | 0.535487 | 0.566356 | 0.081974 | 0.381857 |
| 128 | `[1,1,1]` | 16 | 0.099323 | 0.356212 | 0.538805 | 0.566752 | 0.081361 | 0.386325 |
| 128 | `[1,2,4]` | 15 | 0.096645 | 0.342503 | 0.535654 | 0.566566 | 0.076713 | 0.381491 |
| 128 | `[1,2,4]` | 16 | 0.096682 | 0.365005 | 0.533023 | 0.567011 | 0.082802 | 0.386429 |
| 256 | `[1,1,1]` | 15 | 0.095604 | 0.344894 | 0.537880 | 0.566657 | 0.078913 | 0.381871 |
| 256 | `[1,1,1]` | 16 | 0.098108 | 0.360732 | 0.539096 | 0.567330 | 0.079449 | 0.386918 |
| 256 | `[1,2,4]` | 15 | 0.094384 | 0.346495 | 0.537243 | 0.566462 | 0.080906 | 0.381802 |
| 256 | `[1,2,4]` | 16 | 0.096359 | 0.369233 | 0.538737 | 0.567056 | 0.075166 | 0.388449 |

The factorial two-seed summary is:

| Width | Dilation | Mean MSE | Seed range | Change from width-128, `[1,1,1]` |
|---:|---|---:|---:|---:|
| 128 | `[1,1,1]` | 0.384091 | 0.004468 | -- |
| 128 | `[1,2,4]` | 0.383960 | 0.004939 | -0.000131 |
| 256 | `[1,1,1]` | 0.384394 | 0.005046 | +0.000303 |
| 256 | `[1,2,4]` | 0.385125 | 0.006647 | +0.001034 |

Restoring multiscale dilation is therefore neutral (a 0.034% mean improvement at
width 128), while doubling the patch embedding is neutral to worse. Both effects
are much smaller than seed variation. The BatchNorm width-128, `[1,1,1]` mean is
also `0.002356` worse than the selected two-seed InstanceNorm update control
(`0.381735`), reversing the apparent seed-15 BatchNorm win. No normalization,
receptive-field, or scalar embedding-width candidate is a finalist, and the proxy
gate remains closed by more than a factor of four. The next quality experiment
must change how spatial information crosses the encoder/decoder bottleneck rather
than tune these controls further.

| Width/dilation/seed | Config SHA-256 | Best checkpoint SHA-256 | Latest checkpoint SHA-256 | EMA checkpoint SHA-256 |
|---|---|---|---|---|
| 128/124/15 | `85a9a317e871fa2ce8fac66a2c83a60306ace2e12ad481b0c5f53f33ccb4c41d` | `1077e56fa93165f25192097edf2a079f78e21ae274dbb59db28fdc932e6adc5e` | `59ceb5202613d0e875ce87a01cacb290626da567d9833747da9e9a69396c029b` | `2590682f270b9d532476ff277d1aa85d30c84d6591fff00d328a7d11cedcab10` |
| 128/111/16 | `d10f9f887e0aa2f2e30f15497dfce78e90218e0660ab5fa657877c328365dd05` | `7a15afdf1acbaa33cd2dc367b6154db5a1363dcb667c9b6f08e574169e9129ce` | `b3ba227809877c5dae9f5ff4182196cb594aaf4db7c2a35cf7b31455fff9572e` | `3ecc881a00022727ad0f06cd236f7abd28b688abbe51efda5e8fd843493ec938` |
| 128/124/16 | `840f8bd046ab153186b02cb063aa77ded507a839748392c4f7cc1fb82517e72a` | `8fbd7a521c80e9a59d6d28fad05546d3fa39d12728cbebca46af647d62cd24d9` | `77e06ff4292223c8fd0fb32f39a226cf7f07c4f397b99c43cfd781dbac257ba2` | `fc64e4153d2524b5f483a18765e0f797ab9dff2742017a86ee0c5eee4293e32b` |
| 256/111/15 | `67fcd84e2f8ea7f98fe19dff33b45c8d3c09a0322bd3ed8c297fdea1248a61af` | `148ad642e8303017f0a39f8e574d86ecf327d5a2d2af67dbd034886dc8a1116a` | `6e2d40756c7c95a6a164cb1e0f8906040b59156cdedd50d4e7a0d80facc409f8` | `aa105d78d67db2aa59fd3298695b4c1897d3714c7cec1623c257b82a489ab920` |
| 256/111/16 | `f290456658aae48b860ae25fe58918851b36d62f88939f4e61742c1f3f37f43e` | `9adddd7f99cf0c2f5783605ed7aafe1f61a41bb13f6d5520333a2b3dc2cd1d71` | `716d77f1318e14bed4491ff06c44941a984861921a0a1db71d14a1b7ff3129f7` | `484fc1d4bb0adfa75ffa189b51a65e47dce6206edf1d507727806d24f7bb1eb0` |
| 256/124/15 | `809c0132ffae5ac481b72eb15bfafe5d7ee516cf9431d1defae821a98afaf513` | `f23f06d3a5c9f83ae122985845c17a5163f42f6215068dac6af964f59c252162` | `c45bf8c201ce0ab20a4e70463b72ed020a9a3db1befc3b7492b8afc7866515a6` | `84b3462ea4637878d20c4d2cbde8e4919ac5698da89132273aaec1c2f100e415` |
| 256/124/16 | `76c77536c807bfea57b008df1e1b6d19108f2488d4b60dbe9fc91486b53f17b7` | `450448227a99c961a1e24d3fb27d7e78a8da93d711f926ed2cdc4d23404268d3` | `6ec8b964b25de92b32ee686c8c6dc3dbc62ef42e2ecd1c611ea1341d4e8f3da0` | `0f3138afb504dee5c3cbdbce29cdb1d3c34669d2110d897d021f2da830f51aee` |

Width-128 checkpoints are 1,216,186,305 bytes (EMA 912,180,331), while
width-256 checkpoints are 1,262,727,233 bytes (EMA 947,087,019). All GPU peaks
were about 26.5 GiB. Apptainer MaxRSS ranged from 12.39 to 12.76 GiB on `gr101`;
the two dilation jobs placed on `gr102` used 25.40 to 25.75 GiB, still within the
40-GiB request.

## B2 learned fine-scale decoder queries

> Historical note: the `use_fine_scale_queries` feature evaluated in this section
> was subsequently removed because it bypassed the latent processor route. These
> results remain useful evidence about the original representation bottleneck but
> do not describe an available model option.

The identity evidence and the failed scalar-width ablation justify changing how
full-resolution information crosses the patch bottleneck. The smallest isolated
candidate projects the raw prognostic and boundary inputs at every grid cell through
a zero-initialized learned 1-by-1 projection and adds that representation to the
decoder's pixel queries. The processor and patch-token path are unchanged. This is
an encoder-to-decoder feature connection, not residual-field prediction: the model
still predicts the complete absolute target, `pred_residuals` remains `false`, and
no input field is added to the output.

The implementation is opt-in through `model.use_fine_scale_queries`, requires the
input and output grids to match, and preserves the original decoder exactly at
initialization. Tests cover global and windowed decoders, initial equivalence,
nonzero learning signal from the zero initialization, input-shape validation, and
the existing window-vectorization equivalence. The focused decoder, model, and
configuration suite passes 29 tests; mypy and Ruff pass for all changed Python
files. CLI validation of the pinned proxy confirms fine-scale queries are enabled
while retaining plain MSE, one step, absolute prediction, and the 3-degree by
5-degree patch extent.

The first screen will use the selected two-seed InstanceNorm/update-schedule control
and change only `model.use_fine_scale_queries`. Independent one-GPU jobs preserve
effective global batch 32 and normalization behavior while using the authorized
parallel GPU capacity. A one-degree 32-sample identity diagnostic will run alongside
the two forecast proxies to measure whether the path improves MSE, high-wavenumber
power, and patch seams. The proxy promotion threshold remains `0.08575`.

### Fine-query identity result

The 32-sample one-degree identity diagnostic completed 20 epochs and 640 optimizer
updates on seed 15. It changed only the fine-scale-query option relative to the
previous 32-sample one-degree identity control. The result strongly validates the
new information path:

| Candidate | Mean MSE | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | Mean high-k power ratio | Mean seam ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Original patch bottleneck | 0.324229 | 0.057409 | 0.176728 | 0.509057 | 0.566564 | 0.060717 | 0.259720 | 1.15737 |
| Fine-scale decoder queries | 0.028478 | 0.014306 | 0.047032 | 0.023945 | 0.028633 | 0.013981 | 0.862189 | 1.04379 |

The fine-query path reduces aggregate reconstruction error by `11.39x`. The most
important change is that both velocity components now learn the identity map rather
than remaining near their untrained errors. Mean retained high-wavenumber power
rises from 26% to 86%, and the mean patch-seam ratio moves close to one. Salinity
is the weakest remaining identity variable, retaining 66% of high-wavenumber power,
but it is no longer evidence of a catastrophic shared bottleneck.

The first allocation, Slurm `14384170`, requested 40 GiB and failed during the
one-time OCI-to-SIF conversion before Python started. The cached-image retry,
Slurm `14384736`, used one GPU, four CPUs, and a 96-GiB host-memory request and
completed successfully. Its allocation elapsed time was `38:31`, including image
conversion; the training application took `20:50`, peaked at 6.48 GiB host RSS,
and the identity loop peaked at about 26.5 GiB GPU memory. The final epoch took
45.1 seconds for 32 samples. The immutable container is
`26.05-70baf6f216ad9cefc36ff7d9bfa0825b02e262f2`, built from commit
`70baf6f216ad9cefc36ff7d9bfa0825b02e262f2`. The run is
[0p33su9x](https://wandb.ai/ocean_emulators/default/runs/0p33su9x).

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| Resolved `config.yaml` | `d59db742762f89dc7304abe17e79da4ad532933e416552808a8a1f822cd69265` | 2,542 |
| `identity_metrics.json` | `79064637dab5054a94bc80ae09bb253a6599bfe9af667ef164ec4f67bdc24b6c` | 432,618 |
| `identity_spectra.pt` | `87beaaca42e40d439eb668defcddfe28544ceb3aac6642c3ecbf3e3e41dbbb2c` | 114,477 |
| `saved_nets/ckpt.pt` | `cbdb9e425f021cbcde012f75cd5aa6cac82a44e948ad4950eafdb41a53edb89c` | 1,215,833,281 |

### Fine-query proxy screen

The paired forecast screen completed on separate one-GPU allocations. Both jobs
used the stratified 512-window update-scheduled proxy, plain
normalized MSE, one-step absolute prediction, batch size two with 16-step gradient
accumulation, and effective global batch 32.

| Seed | Slurm | W&B | Allocation |
|---:|---:|---|---|
| 15 | `14385012` | [wki0obs5](https://wandb.ai/ocean_emulators/default/runs/wki0obs5) | 1 GPU, 4 CPU, 40 GiB |
| 16 | `14385013` | [vnx6on61](https://wandb.ai/ocean_emulators/default/runs/vnx6on61) | 1 GPU, 4 CPU, 40 GiB |

Both jobs completed normally at epoch 12. The fine-query representation materially
improves the forecast objective relative to the selected InstanceNorm/update control,
but it does not pass the `0.08575` promotion gate:

| Seed | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | Persistence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.153675 | 0.055032 | 0.199370 | 0.133110 | 0.171916 | 0.046324 | 0.084565 |
| 16 | 0.151690 | 0.050937 | 0.207194 | 0.131776 | 0.168149 | 0.032902 | 0.084565 |
| Mean | **0.152682** | 0.052985 | 0.203282 | 0.132443 | 0.170032 | 0.039613 | **0.084565** |

The two-seed mean is 60.0% lower than the selected scalar-token control mean of
`0.381735`, confirming that the decoder-side information path is useful. It remains
`1.78x` the proxy gate and `3.60x` the v2 proxy mean of `0.042390`, however. It is
also `1.81x` the one-step persistence error: the absolute-field forecast has not yet
learned the target increment well enough to beat copying the input. Salinity is now
the largest variable error, followed by meridional velocity. No third seed or
full-data run is promoted from this candidate.

The nearest terminal spatial diagnostics are consistent across seeds. High-wavenumber
power ratios for seed 15/16 are temperature `0.755/0.721`, salinity `0.249/0.258`,
zonal velocity `0.555/0.574`, meridional velocity `0.388/0.402`, and SSH
`1.711/1.435`. Thus the learned query path restores considerable temperature and
velocity structure but still strongly smooths salinity and subsurface velocity.
Variable patch-seam ratios stay near one (`0.978` to `1.077`), so explicit seam
amplification is not the primary remaining failure. Each run retains 782 validation
images covering full-field/error maps and spectra in its linked W&B record.

Seed 15 used 1:45:33 of application time and seed 16 used 1:46:03. Both processed
6,144 samples and 192 optimizer updates; terminal throughput was about 1.05 samples
per second, peak allocated GPU memory was about 26.5 GiB, and application MaxRSS was
13.17/13.16 GiB. The immutable container commit is
`70baf6f216ad9cefc36ff7d9bfa0825b02e262f2`.

| Seed | Artifact | SHA-256 | Bytes |
|---:|---|---|---:|
| 15 | Resolved `config.yaml` | `774fc8d4e92130f398d0f0bd1266e84aacb31841d6a9ecf070922a396f96257a` | 2,596 |
| 15 | `saved_nets/best_validation_ckpt.pt` | `764281c86056f2abc4949862a2d4944070b36f90b58f8dd67a2dd295a051c0d0` | 1,215,833,537 |
| 16 | Resolved `config.yaml` | `ae266541a3c503ac23fc305a9b6d952a2e551f8e8f504b17fcbb1c301ca6b308` | 2,596 |
| 16 | `saved_nets/best_validation_ckpt.pt` | `66924ea72341af7ac7349cef45780e6f2cc613d65311049ece9082569e5b8014` | 1,215,833,537 |

The terminal comparison used the validation-selected epoch from the explicit
`unweighted_normalized_mse` family, never the dashboard alias. At that same epoch
the record includes every variable and depth, the persistence MSE (equivalently,
the normalized target-increment MSE), processed samples, optimizer updates,
throughput, and peak memory. Image-validation epochs also provide mean error and
full-field maps, zonal spectra, high-wavenumber power ratios, and patch-seam jump
ratios. The nearest image-validation epoch at or before the selected checkpoint is
used when the selected epoch itself is not an image epoch.

### Contingency: multiple spatial encoder tokens

The paired fine-query screen has closed the gate, so this is the next isolated
representation experiment. It keeps the
outer 3-degree by 5-degree physical patch grid at 60 by 72 but replaces the single
mean-pooled patch vector with spatially queried intra-patch tokens. At one degree,
a 3-by-5 query grid naturally corresponds to the 15 native cells in each physical
patch. Packing 16 channels from each query would give the processor 240 explicitly
ordered input channels while leaving its spatial grid, receptive field, decoder
windowing, target, and loss unchanged. A PerceiverIO-style encoder can produce all
15 query outputs from one shared latent computation; it avoids running 15 separate
encoders. The first isolation adds this encoder representation to the
fine-query candidate and changes no other training control.

The implementation is opt-in through `model.encoder.spatial_query_shape`, with
`spatial_query_channels` controlling the emitted channels per query and
`queries_dim` controlling the coordinate-conditioned query embedding. The planned
one-degree isolation uses shape `[3, 5]`, 16 channels per query, and query dimension
64, producing 240 explicitly ordered processor input channels. Input pixels retain
their existing intra-patch Fourier coordinates; the 15 outputs share one PerceiverIO
latent computation and are packed in row-major order. The processor is built from
the encoder's actual output width, while the original scalar encoder remains the
default. The real naive PerceiverIO forward/backward path and focused encoder,
decoder, model, and configuration suite pass 38 tests; the repository-wide Ruff,
mypy, schema, YAML, secrets, and license checks pass.

### Spatial-token identity result

The 32-sample one-degree identity diagnostic completed 20 epochs and 640 optimizer
updates on seed 15. Relative to the fine-query identity control, it changed only the
encoder representation from one 128-channel mean-pooled patch vector to 15 ordered
queries with 16 channels each. It retained fine decoder queries, absolute-field
prediction, plain MSE, and the 3-degree by 5-degree physical patch extent.

| Candidate | Mean MSE | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|---:|
| Fine decoder queries | 0.028478 | 0.014306 | 0.047032 | 0.023945 | 0.028633 | 0.013981 |
| Fine queries + 3x5 encoder tokens | **0.026923** | **0.013103** | **0.041771** | **0.022449** | 0.028945 | **0.010139** |

The spatial-token encoder lowers aggregate reconstruction MSE by 5.46%. The largest
relative gains are SSH (27.5%) and salinity (11.2%); meridional velocity is 1.1%
worse. Variable high-wavenumber power ratios remain effectively unchanged: the
fine-query/spatial-token values are temperature `0.883/0.876`, salinity
`0.663/0.670`, zonal velocity `0.860/0.858`, meridional velocity `0.863/0.873`,
and SSH `1.043/1.034`. Variable seam ratios remain near one (`0.998` to `1.054`).
This is a small but broad reconstruction improvement, not evidence by itself that
the forecast gate will pass.

The immutable x86 image was published successfully by
[GitHub Actions](https://github.com/m2lines/Samudra/actions/runs/29768664769) from
commit `5399a4ee805428c355f4c82330fa5aeb06864741`. Slurm `14397453` completed
normally in `37:40`, including the one-time SIF conversion; the application step
took `20:26` and peaked at 6.51 GiB MaxRSS. Steady identity epochs fell from about
45 seconds to 38 seconds and throughput rose from `0.7100` to `0.8429` samples per
second. The W&B run is
[ou5hjipa](https://wandb.ai/ocean_emulators/default/runs/ou5hjipa).

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| Resolved `config.yaml` | `3b325d02c949c847cc6fc471faf5c4a13d2f70d85fbf85bbd15a6b0216e9d36d` | 2,649 |
| `identity_metrics.json` | `2a421ae2fa44ab316a068aeab2a415fbab96b7fd5278f3f8e33c48ecac010b94` | 432,584 |
| `identity_spectra.pt` | `7564f0bf9d3bfb96d149e1f4799883ee1610cf85a28d7733cee7167cb138c782` | 114,477 |
| `saved_nets/ckpt.pt` | `c6499119c17b94ab03b1a3008588347fb1c73fb86c8ed6eb25ff3bb1dd8974b3` | 1,248,600,239 |

The paired forecast screen uses one GPU per seed with batch size two and 16-step
gradient accumulation, preserving effective global batch 32. Slurm `14399789`
(seed 15) and `14399790` (seed 16) use the same immutable image and began normally;
their two-seed mean will be compared directly with the `0.08575` promotion gate.

### Spatial-token proxy result

Both spatial-token proxy jobs completed normally at epoch 12. The exact terminal
unweighted metrics are:

| Seed | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | Persistence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.151654 | 0.051359 | 0.195287 | 0.135030 | 0.169040 | 0.035632 | 0.084565 |
| 16 | 0.150076 | 0.051941 | 0.199199 | 0.127790 | 0.169143 | 0.038953 | 0.084565 |
| Mean | **0.150865** | 0.051650 | 0.197243 | 0.131410 | 0.169092 | 0.037292 | **0.084565** |

The 15-token encoder improves the paired fine-query mean by only 1.19%, from
`0.152682` to `0.150865`. It remains `1.76x` the `0.08575` gate, `3.56x` the v2
proxy mean, and `1.78x` persistence. Therefore this representation is not promoted
to a third seed or full-data training. Salinity remains the largest error.

High-wavenumber power ratios for seed 15/16 are temperature `0.761/0.737`,
salinity `0.251/0.278`, zonal velocity `0.571/0.565`, meridional velocity
`0.417/0.397`, and SSH `1.515/1.462`. Variable patch-seam ratios remain between
`0.977` and `1.082`, again excluding seam amplification as the main error. Each
run processed 6,144 samples and 192 optimizer updates. Terminal throughput was
`1.402/1.181` samples per second and peak allocated GPU memory in the training log
was about 25.1 GiB.

Seed 15 completed in `1:42:26` with 13.08 GiB application MaxRSS; seed 16 completed
in `1:47:57` with 12.66 GiB MaxRSS. The W&B runs are
[pb5xt0pf](https://wandb.ai/ocean_emulators/default/runs/pb5xt0pf) and
[gox02otd](https://wandb.ai/ocean_emulators/default/runs/gox02otd). Resolved
configs and validation-selected checkpoints are preserved under the corresponding
Slurm output directories. Both runs use immutable container commit
`5399a4ee805428c355f4c82330fa5aeb06864741`.

### One-cell representation controls

The next controlled comparison separates two bottlenecks that the 3-degree by
5-degree experiments conflate. Both new arms use a 1-degree by 1-degree physical
patch extent and therefore preserve the native 180 by 360 grid throughout the
processor:

1. **One-cell Perceiver:** retain the Perceiver encoder and PerceiverIO decoder,
   but give each encoder patch and decoder target exactly one native grid cell.
   This removes within-patch spatial compression while retaining the learned
   latent representation machinery.
2. **One-cell direct:** replace the encoder and decoder Perceivers with learned
   1-by-1 channel projections. This removes both within-patch compression and
   the Perceiver bottleneck while leaving the processor unchanged.

The primary patch-size control is the completed 3-degree by 5-degree scalar
Perceiver baseline: like both one-cell arms, it has 128 processor input channels and
no fine-scale decoder query path. The running 3-degree by 5-degree spatial-token
arm is retained as a stronger contemporaneous benchmark, not treated as a
single-variable control because it has 240 processor input channels and fine-scale
queries. All arms retain one-degree data, one-step absolute-field prediction, plain
normalized MSE, effective global batch 32, the fixed proxy samples and seeds, and
the same unweighted validation diagnostics. Identity and throughput screens precede
paired proxy training because the full-resolution processor is expected to cost
materially more memory and compute. This user-authorized diagnostic is the explicit
exception to the promoted setup's fixed 3-degree by 5-degree physical patch rule.

Aurora's scale encoding ordinarily estimates patch area from the minimum and
maximum pixel centers in each patch. Those values coincide for a one-pixel patch,
so the first focused test exposed a zero-area assertion. The shared encoder and
decoder encoding path now infers latitude and longitude cell edges from adjacent
centers for exactly this case. The direct heads are opt-in through
`model.encoder.direct_projection` and `model.decoder.direct_projection`, reject
spatially compressive patch sizes, and do not add a skip or residual path. The
focused representation/configuration suite passes 45 tests; repository-wide Ruff,
formatting, mypy, schema, YAML, secret, and license checks pass. The broader CPU
suite passes 404 tests with 10 expected failures and 2 skips; six unrelated
SamudraMulti data tests cannot construct the existing `auto` Perceiver locally
because CUDA is visible but the optional `flash_perceiver` package is absent.

### One-cell fit and throughput probes

The immutable x86 image for these controls was published by
[GitHub Actions](https://github.com/m2lines/Samudra/actions/runs/29775735655) from
commit `0417c48a09795cfbbf298fdee78d5e1c3e971082`. Initial Slurm jobs `14411601`
and `14411602` were killed while converting the new image under a 40 GiB host-memory
limit; neither entered model code. Job `14415794` materialized the shared SIF with a
temporary 120 GiB request, after which all model probes returned to 40 GiB host
memory.

The direct 1-by-1 projection control fits on one GPU. With four identity samples,
the checkpoint-all probe (`14416901`) took 47.58 seconds, or `0.0841` samples per
second, and reached MSE `0.352240` after four updates. The matched selective-
checkpoint probe (`14421234`) took 52.40 seconds, or `0.0763` samples per second,
and reached MSE `0.352141`. Their W&B runs are
[qk7cgmrc](https://wandb.ai/ocean_emulators/default/runs/qk7cgmrc) and
[wqw35fze](https://wandb.ai/ocean_emulators/default/runs/wqw35fze). The one-cell
seam ratio is undefined because every grid edge is a patch edge; the resulting NaN
is expected and the full-field/spectral diagnostics remain applicable.

Keeping 256 Perceiver latents per one-pixel patch is not trainable even on the
96 GiB RTX6000. Decoder window batches of 16 (`14415794`), 4 (`14416900`), and 1
(`14420121`) all fail during the first backward pass. The latter two reach 92.87 GiB
of allocated GPU memory. Selective checkpointing (`14421233`) exposes the root
cause more directly: the encoder creates 256 latents for each of 64,800 input
pixels, or 16.6 million latent tokens, and its recomputation requests another
47.46 GiB after 49.35 GiB is already allocated. The meaningful Perceiver control
therefore uses one latent for its one input token and one output query; this keeps
the Perceiver computation present without manufacturing 256 redundant latents per
cell. A four-sample fit probe is the next gate before any longer identity or proxy
run.

The one-latent flash implementation (`14422071`) avoids the memory failure but
aborts on an illegal-memory-access kernel edge case for length-one attention. The
same architecture with standard attention (`14423956`) completes normally. Its
four-sample train/eval loop takes 35.89 seconds (`0.1115` samples per second) and
ends at MSE `0.550016` after four updates; the W&B run is
[ziilmmu3](https://wandb.ai/ocean_emulators/default/runs/ziilmmu3). This is faster
than the direct selective-checkpoint smoke, but the initial MSEs are not comparable
after four updates because the heads initialize differently. Slurm `14424555`
(one-latent Perceiver) and `14424556` (direct projections) therefore run the same
32 fixed samples for 10 epochs, evaluating every two epochs on separate one-GPU
allocations. This is the first quality-bearing comparison between the one-cell
heads.

Both 10-epoch jobs completed all 320 updates. The direct projections learn the
fixed identity task; the one-latent Perceiver does not:

| Head | Mean MSE | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|---:|
| One-latent Perceiver | 0.285467 | 0.043439 | 0.074915 | 0.477047 | 0.558473 | 0.051854 |
| Direct 1-by-1 projections | **0.005257** | **0.003655** | **0.002985** | **0.006883** | **0.007747** | **0.005073** |

The direct head is 54.3 times better in aggregate and its improvement covers every
variable. Its final high-wavenumber power ratios are temperature `0.881`, salinity
`0.946`, zonal velocity `0.912`, meridional velocity `0.978`, and SSH `0.874`.
The Perceiver retains only `0.026/0.001` for zonal/meridional velocity despite
reasonable thermohaline ratios (`0.804/0.793`). Steady final-epoch throughput is
`1.718` samples per second for direct versus `0.945` for Perceiver. The W&B runs
are [n9a3gne6](https://wandb.ai/ocean_emulators/default/runs/n9a3gne6) and
[mb5j6ft0](https://wandb.ai/ocean_emulators/default/runs/mb5j6ft0).

The direct control is therefore promoted to paired stratified proxy runs with
batch size one and 32-step gradient accumulation, retaining effective global batch
32. Slurm `14425434` (seed 15) and `14425437` (seed 16) use one GPU each. Slurm
`14425438` separately repeats the Perceiver identity control with four latents so
the representation conclusion does not depend on the length-one latent case.

The four-latent control completes at MSE `0.285778`, effectively identical to the
one-latent result. Its variable MSEs are temperature `0.042419`, salinity
`0.074535`, zonal velocity `0.475744`, meridional velocity `0.561658`, and SSH
`0.049393`. Velocity high-wavenumber ratios remain near zero (`0.0292/0.0008`).
Increasing the one-cell Perceiver to four latents therefore adds cost without
closing the representation gap; no Perceiver proxy is promoted. Its W&B run is
[yf157jdj](https://wandb.ai/ocean_emulators/default/runs/yf157jdj).

### Direct one-cell proxy result

All three direct proxy jobs completed normally at epoch 12 and decisively passed
the promotion gate:

| Seed | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH | Persistence |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.042535 | 0.006683 | 0.006366 | 0.047173 | 0.065870 | 0.007645 | 0.084426 |
| 16 | 0.040106 | 0.005877 | 0.005669 | 0.044818 | 0.061858 | 0.006353 | 0.084426 |
| 17 | 0.041192 | 0.006107 | 0.006081 | 0.045879 | 0.063761 | 0.006718 | 0.084426 |
| Mean | **0.041278** | 0.006223 | 0.006039 | 0.045957 | 0.063830 | 0.006905 | **0.084426** |

The three-seed mean is 51.1% below persistence, 51.9% below the `0.08575` gate,
72.6% below the spatial-token mean, and 2.6% better than the matched v2 proxy mean
of `0.042390`. Temperature, salinity, and SSH are all around `0.006`; velocity is
still the hardest group but is no longer catastrophically dominant.

Mean high-wavenumber power ratios are temperature `0.861`, salinity `0.900`, zonal
velocity `0.599`, meridional velocity `0.677`, and SSH `0.901`. One-cell patch-seam
ratios are undefined by construction. Each run processed 6,144 samples and 192
optimizer updates. Terminal throughput was between `2.23` and `3.00` samples per
second, peak allocated GPU memory was only about 3.7 GiB, and application MaxRSS
was about 11.2 GiB. The jobs completed in 54--57 minutes. Their W&B runs are
[qapetrfv](https://wandb.ai/ocean_emulators/default/runs/qapetrfv) and
[gfqs4knn](https://wandb.ai/ocean_emulators/default/runs/gfqs4knn), and
[407lfafs](https://wandb.ai/ocean_emulators/default/runs/407lfafs).

The saved `config.yaml` SHA-256 values for seeds 15/16/17 are respectively
`cbffda34ed8f244e2d2327b5e025d3464e2bc3139fab7dc64ef0979b313383cd`,
`c0897d590a5eafcf46e2f472cf51e61c2f3e0e7752163179b2b0c14076d41142`, and
`5cebcf04a35ab42e4e01894e4a8f73b1db5b6b02d876387413650adbde1f0f27`.
The corresponding 1,188,313,039-byte best-checkpoint hashes are
`25b02da5c3b7e0f35fdbb9aa0f21f33b769e86262da8c8e2cc1679d8bc5cb4d1`,
`e630cacc84e5a59de3f3c35cb58b94152af4fb9773ead7cc08dd27a67b02b58d`, and
`5ad1c7f16776aebfda1835f3af394b2635944454d0eb3a78d8a3505b90a40436`.

The third fixed seed confirms that the direct control is a genuine finalist.
Full-data Slurm job `14440510` therefore promotes the pinned configuration on two
GPUs with batch size two per rank, eight-step gradient accumulation, effective
global batch 32, and the fixed 6,160-update cosine schedule. It retains
absolute-field, single-step normalized MSE and disables parameter watching. Initial
eight- and four-GPU allocations `14437365` and `14437494` never started model code:
the former was cancelled after a July 22 resource estimate, and the latter exposed
the account's lower GPU QoS cap. The two-GPU job started immediately without
changing any scientific control. The immutable image is the same `0417c48a` image
used by the proxy jobs; its CLI overrides are semantically equivalent to
`train_1deg_1cell_direct_mse_updates.yaml` except for the resource-equivalent
accumulation count, and the resolved run config is the authoritative record.

### Full-data direct one-cell result

Slurm `14440510` completed normally with exit code zero in `08:19:45` (application
training time `08:19:11`). The validation-selected checkpoint is epoch 53:

| Metric | Direct one-cell | Quoted v2 reference | v2 / direct |
|---|---:|---:|---:|
| All channels | **0.015976** | 0.023600 | 1.48x |
| Temperature | 0.000549 | 0.001490 | 2.71x |
| Salinity | 0.000573 | 0.001380 | 2.41x |
| Zonal velocity | 0.017191 | 0.036100 | 2.10x |
| Meridional velocity | 0.024280 | 0.056500 | 2.33x |
| SSH | 0.001253 | 0.002390 | 1.91x |

These are unweighted normalized one-step MSEs. The direct model is 78.7% below the
`0.075` completion threshold, 68.0% below the stronger `0.05` target, and 32.3%
below the quoted v2 aggregate reference. This is not a rigorously matched parity
claim, but it establishes that the full-resolution processor can match or exceed
the practical v2 target when the patch representation bottleneck is removed.
The linked W&B run is
[uwzdfu8s](https://wandb.ai/ocean_emulators/default/runs/uwzdfu8s).

The persistence MSE is `0.08456`, equivalently the normalized target-increment MSE.
The selected forecast is 81.1% lower. At terminal epoch 70 the model remains close
to the best checkpoint at aggregate MSE `0.01631`; its variable MSEs are temperature
`0.00052`, salinity `0.00055`, zonal velocity `0.01734`, meridional velocity
`0.02455`, and SSH `0.00121`. Terminal high-wavenumber power ratios are temperature
`0.98301`, salinity `0.99389`, zonal velocity `0.83560`, meridional velocity
`0.88508`, and SSH `0.97269`. Thus the direct model largely preserves thermohaline
and SSH variance and retains much more velocity energy than the patch-token
candidates. One-cell seam ratios are undefined by construction rather than evidence
of a seam artifact. The run retains 2,737 W&B media files with mean-error/full-field
maps, snapshots, spectra, and the underlying channel diagnostics.

Terminal depth MSEs for levels 0 through 18 are respectively `0.02989`, `0.02486`,
`0.01643`, `0.01045`, `0.00710`, `0.00548`, `0.00463`, `0.00448`, `0.00440`,
`0.00434`, `0.00435`, `0.00496`, `0.00619`, `0.00819`, `0.01155`, `0.01626`,
`0.01954`, `0.01114`, and `0.00261`. The remaining error is concentrated near the
surface and at levels 15--16, consistent with velocity remaining the hardest
variable rather than a new deep thermohaline failure.

The run processed 197,680 samples and made 6,230 optimizer updates. The latter is
70 more than the nominal 6,160-update cosine target: 706 microbatches per epoch
leave a two-microbatch final accumulation cycle, and the trainer correctly performs
one final partial optimizer step each epoch. The learning rate therefore reached
its minimum near the end and moved only negligibly past it; this accounting mismatch
does not explain the quality gain, but must be retained when reproducing the exact
schedule. Terminal global throughput was `9.307` samples per second. Per-rank peak
allocated GPU memory reached about 17.3 GiB during bring-up; application MaxRSS was
51,854,124 KiB for the two-rank step, while the terminal per-rank CPU peak was about
4.57 GiB.

The resolved 2,717-byte `config.yaml` has SHA-256
`6682d6e78677df00e39658c8a41f8f950002a15f2fbe91cf88c529c91f8d7ddd`.
The 1,188,313,039-byte epoch-53 `best_validation_ckpt.pt` has SHA-256
`b28761c2f73e4135a21ac503171bff8d4135994de62ad0ffd29f5b55b1850aff`.
The code image remains commit `0417c48a09795cfbbf298fdee78d5e1c3e971082`; the
dedicated checked-in configuration and proxy decision record are on commit
`059fe1ce`, with the resource-adjusted launch record on `56ce0007`.

### Pure-autoencoder localization

The one-cell result above still passed data through the full processor, so a
32-sample pure-autoencoder factorial bypassed the processor and separated the two
heads. Every arm used seed 15, plain MSE, 10 epochs, 320 updates, and the same fixed
one-degree samples. The diagnostic controls were published in immutable image
`307f9b134536531593fa8739f5b7ba22e71a2f74` by
[GitHub Actions](https://github.com/m2lines/Samudra/actions/runs/29832577268).

| Encoder / decoder | Slurm | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|---:|---:|
| Direct / direct | `14481633` | **0.012083** | 0.009244 | 0.008433 | 0.014648 | 0.015996 | 0.019846 |
| Perceiver / direct | `14482666` | **0.025431** | 0.013312 | 0.016189 | 0.034166 | 0.037758 | 0.019835 |
| Direct / Perceiver | `14482667` | 0.279303 | 0.040842 | 0.073957 | 0.470026 | 0.544039 | 0.048317 |
| Perceiver / Perceiver | `14482668` | 0.285353 | 0.043859 | 0.074375 | 0.475765 | 0.558917 | 0.049720 |

The W&B runs are [dqwsd76y](https://wandb.ai/ocean_emulators/default/runs/dqwsd76y),
[ddya628l](https://wandb.ai/ocean_emulators/default/runs/ddya628l),
[mfwis08s](https://wandb.ai/ocean_emulators/default/runs/mfwis08s), and
[xb4o9xjb](https://wandb.ai/ocean_emulators/default/runs/xb4o9xjb), in table order.
Changing only the encoder adds about `0.0133` MSE, whereas changing only the decoder
adds about `0.2672`. The full Perceiver result is also effectively identical to the
earlier processor-present one-cell identity MSE of `0.285467`. The processor is
therefore neither the source of the failure nor able to repair the decoder.

The decoder geometry explains the result. Its nominal one-latent configuration has
one learned latent per 12-by-12 output window, not one latent per output query. The
window presents 196 context cells to that latent and then asks 144 coordinate
queries to recover different output cells. With only one key, the output-query
cross-attention softmax is exactly one for every query, so the query coordinates
cannot select different values. More unanchored latents leave a difficult learned
spatial-assignment problem rather than providing an identity-aligned route.

Targeted decoder-only controls support this diagnosis:

| Decoder control, direct encoder | All MSE |
|---|---:|
| 12-by-12 window, 1 latent | 0.279303 |
| 12-by-12 window, 16 latents | 0.280127 |
| 12-by-12 window, 64 latents | 0.267841 |
| 12-by-12 window, 256 latents | 0.278018 |
| 12-by-12 window, 64 latents, no input-context norm | 0.276040 |
| One output cell, 3-by-3 context, 1 latent | 0.133080 |
| One output cell, one input cell, 1 latent | 0.040990 |
| One output cell, one input cell, 64 latents, no context norm | 0.038147 |
| One output cell, one input cell, width 128 | 0.033983 |
| One output cell, one input cell, width 128, no context norm | **0.030861** |
| 12-by-12 window, 256 width-128 latents | 0.265364 |
| 12-by-12 window, 256 latents, 30 epochs | 0.248054 |

Thus latent count, width, input normalization, and three times as much training are
secondary. Aligning one input cell with one output query closes most of the gap,
while adding neighboring context without an explicit spatial route makes the task
worse. The 30-epoch control does improve slowly but remains more than 20 times the
direct/direct result, excluding ordinary undertraining as the main explanation.

To preserve variable input/output grids without this second latent bottleneck, the
decoder now offers an opt-in resolution-flexible projection: bilinear resampling of
the canonical processor grid followed by a learned 1-by-1 channel projection. On a
matching grid it skips interpolation and is exactly the direct projection path.
Slurm `14486427` tested it with the Perceiver encoder and processor bypassed; its
MSE is `0.025430571`, bit-for-reported-precision identical to the Perceiver/direct
control, with identical variable metrics. The W&B run is
[yj1m6z35](https://wandb.ai/ocean_emulators/default/runs/yj1m6z35). The implementation
was published in image `6736cf22f989be81f94d30ca4cb9bac0a2aaeb07` by
[GitHub Actions](https://github.com/m2lines/Samudra/actions/runs/29837326578).

### Perceiver-encoder/resampling-decoder proxy

The candidate then completed the calibrated one-degree single-step proxy on seeds
15 and 16. It retained one Perceiver latent per one-degree input cell, the full
processor, plain normalized MSE, the fixed 512 stratified samples, effective global
batch 32, and the 6,160-update schedule semantics. Only the original Perceiver
decoder was replaced by the resampling projection.

| Seed | Slurm | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | `14489084` | 0.049906 | 0.009647 | 0.009004 | 0.058899 | 0.078911 | 0.008486 |
| 16 | `14489085` | 0.053405 | 0.010690 | 0.010807 | 0.062794 | 0.083468 | 0.010035 |
| Mean |  | **0.051655** | 0.010169 | 0.009905 | 0.060846 | 0.081190 | 0.009260 |

The two W&B runs are [dn9ka17p](https://wandb.ai/ocean_emulators/default/runs/dn9ka17p)
and [958qmvlf](https://wandb.ai/ocean_emulators/default/runs/958qmvlf). Both jobs
completed normally in `1:09:29` and `1:15:09`. Peak allocated GPU memory was about
4.4 GiB during training and 2.9 GiB during validation.

The candidate mean is 86.5% below the original update-scheduled SamudraMulti proxy
mean of `0.381735`, 39.8% below the `0.08575` promotion gate, and 38.8% below the
`0.084426` persistence baseline. It is 25.1% worse than the direct-head proxy mean
and 21.9% worse than the matched v2 proxy mean, which is a reasonably comparable
screening result rather than parity. Mean high-wavenumber power ratios are
temperature `0.869`, salinity `0.892`, zonal velocity `0.543`, meridional velocity
`0.631`, and SSH `0.908`. The remaining gap is concentrated in velocity, but the
catastrophic decoder failure is gone.

The revised representation decision is specific. The Perceiver encoder can handle
the one-degree identity task reasonably well (`0.025431` versus `0.012083` for the
direct encoder), while the existing Perceiver decoder's unanchored window-level
latent routing causes nearly the entire surprising gap. A one-cell direct
projection already produces a strong full-data forecast, and the new resampling
projection preserves output-resolution and output-channel flexibility without that
routing problem. Its successful proxy closes the diagnostic loop: the next model
work should test interpolation behavior at a different output grid and then, if
that remains sound, promote this exact candidate to full-data one-degree training
before revisiting multi-resolution training. No residual prediction, dynamic loss,
spectral loss, or autoregressive objective was needed for this conclusion.
