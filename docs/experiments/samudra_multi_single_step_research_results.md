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
calibration for the arbitrary-index interface. Terminal results and the matched v2
ordering are still required before choosing the proxy for the B funnel.

The same run confirms the selected A5 path at normal proxy fidelity. Its latest
completed epoch used about `283` training seconds for 512 samples, or `1.81`
samples/second on one GPU. The historical four-GPU proxy took about `120` seconds
per epoch (`4.27` samples/second total, approximately `1.07` per GPU). Thus the
portable path is about `1.7x` more throughput-efficient per GPU while reproducing
the control curve and reducing the allocation from four GPUs to one.

The epoch-scheduled seed-15 control completed all 12 epochs in `1:05:20` of training
time and `1:06:00` of Slurm allocation time with exit code zero. The one-GPU,
four-CPU, 40-GiB request used 13.34 GiB Apptainer MaxRSS; W&B recorded 5.25 GiB
process CPU peak and 25.87 GiB GPU peak. Its validation-selected epoch 12 is:

| Run | All | Temperature | Salinity | Zonal velocity | Meridional velocity | SSH |
|---|---:|---:|---:|---:|---:|---:|
| Original contiguous control | 0.385084 | 0.098353 | 0.353942 | 0.536955 | 0.566338 | 0.083592 |
| [Stratified seed 15](https://wandb.ai/ocean_emulators/default/runs/ec0f03n4) | 0.384732 | 0.098102 | 0.351683 | 0.537368 | 0.566746 | 0.081088 |

The stratified result is `0.000351` lower overall and reproduces every variable
group closely. It therefore validates the arbitrary-index proxy for this
representative control. The run's experiment log reports `0.759` for the aliased
legacy `val/mean/loss`; that value is invalid and is retained only as evidence of
the diagnosed aggregation bug. Selection uses the independently recomputed
unweighted result above.

| Artifact | SHA-256 | Bytes |
|---|---|---:|
| Resolved `config.yaml` | `84ad2325ed71fb3495d8fe5d20e4824509969033b5613384010a9c803e2558b9` | 2,562 |
| `saved_nets/best_validation_ckpt.pt` | `94840349b7540b087cce298c8b26f14bea0a9fb11e748b0737df05bb1df08a98` | 1,215,668,023 |
| `saved_nets/ckpt.pt` | `01d34b68a45023ee6187bd1834b8bd6d3b64786ec6633af101e576bd92257735` | 1,215,668,023 |

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
