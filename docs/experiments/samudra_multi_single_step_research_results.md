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

## Torch container bring-up findings

The initial bring-up exposed three image-staging constraints before any model batch
ran: 32 GiB was insufficient for the one-time Docker-to-SIF conversion, root-backed
`/tmp` lacked enough disk space, and shared scratch does not support the xattr
operations Apptainer needs. Torch provides a large per-job XFS directory through
`SLURM_TMPDIR`; using it allowed the pinned image to build successfully. The training
harness now prefers that path automatically, validates `REPO_DIR`, and creates
configured cache/temp directories. The Torch skill and repository guide record the
same behavior.
