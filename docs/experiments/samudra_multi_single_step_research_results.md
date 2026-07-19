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
