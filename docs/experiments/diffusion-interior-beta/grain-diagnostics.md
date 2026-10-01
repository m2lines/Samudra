<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Grain diagnostics and a globally supervised next run

The authorized diagnostic array is **24582219**, evaluator `72a9c05ee`, two
single-L40S jobs. No new training was submitted. Results and the resulting run
recommendation will be recorded below after both checkpoint evaluations finish.

## Tests

Use seed-1729 long-trained OM4 weights before observation adaptation and the
stopped, validation-selected weights after adaptation. These are the checkpoints
used in the endpoint report, not the fresh 2k noise pilot. Each checkpoint keeps
its original normalization, forcing and inference mask.

1. Cache the deterministic latent states at day 30 and day 365. Decode eight
   members at 16, 32, 64 and 128 Heun steps with exactly the same initial Gaussian
   draws and unchanged noise endpoints. Compare means, individual fields, spectra,
   RMSE, fair CRPS and spread. Repeat on OM4-initialized January 2015/2018/2021
   cases before and after adaptation, and observation-initialized cases after
   adaptation. Latents match across step counts within a checkpoint; different
   checkpoints have their own learned latents. This evaluates the full learned
   system, not an isolated decoder-weight swap.
2. For the same native day-30 latent, corrupt the actual two clean OM4 target
   fields at sigma 0.002, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 and 80. Predict the
   clean fields in one denoiser call. Use four independent paired noise draws at
   every sigma and checkpoint. Targets enter only the corrupted denoiser input;
   the initializer and processor never see the verifying interior.

Report SST, SSH and T/S at 550 m. For the second test, subtract each cell's mean
error across noise draws before measuring noise response. Noise-to-error gain
is `cov(error, injected noise) / var(injected noise)`; normalize by sigma to
express the fraction of injected noise retained. Also compute this for neighboring
spatial differences, using wet–wet pairs and no latitude wrapping. An identity
predictor has gain and correlation 1; a predictor insensitive to injected noise
has gain 0. This diagnostic is not itself a proper skill score: removing all
noise dependence could also erase legitimate stochastic variability.

Near-zero-sigma gain near 1 alone is not evidence of a broken denoiser. The
absolute corruption is tiny, and the trained posterior mean can legitimately
trust nearly clean inputs. Interpret gain together with sigma, physical error,
sampled member roughness and solver convergence. Denoising known corrupted
fields tests an on-distribution corruption path; generated intermediate states
may lie elsewhere. Neither test uniquely identifies a U-Net architectural limit.

Use global finite wet support for primary metrics, plus the previous ±60° domain
as a labeled sensitivity. Spectra use the same fully wet Pacific patch, averaging
powers across dates rather than fields. Maps use shared limits and 2×2 pixels per
native cell. This three-date, one-seed diagnostic is not the 96-origin benchmark.

## Data contract for the next run

The comparator is the [global physical-only report](https://github.com/m2lines/Samudra/blob/codex/d-observation-pilot/docs/experiments/observation-d/global-physical-day30-2026-09-30.md),
training producer `79025a163817a577ab81af95b401f5cb0563cd12`. Its mixed endpoint
has 8k OM4 and 8k observation updates; scratch endpoints have 8k/16k observation
updates. Architecture and parameter counts need not match, but the observation
data and operators must.

- Use the exact prepared observation samples, grid, masks, train/validation/test
  origins, existing normalization and calendar weights from that report: 243
  training, nine validation and 96 test months. Verify source manifest, grid,
  statistics and sample hashes against the comparator before launching; matching
  counts or product names alone is insufficient.
- Match OISST SST, DUACS SSH, IAP interior T/S, ERA5 forcing, depth representation,
  19-bin surface/forcing history, five-day forecast targets and calendar-weighted
  monthly interior targets. Do not invent instantaneous IAP targets or daily
  losses from these five-day samples.
- Remove the ±60° cutoff from observation forecast, reconstruction, completion,
  validation selection and evaluation weights. Retain cosine-latitude weighting,
  model wet masks and finite observation support. Global does not mean filling
  missing polar observations with targets. OM4 denoising already uses global wet
  support; existing normalization constants remain unchanged.
- Match the newer zero **normalized** placeholders plus explicit validity flags
  for missing surface inputs, with observed-only surface anchoring and learned
  missing-cell completion. The current diffusion loader instead fills input gaps
  with training climatology. Change both training and annual inference paths.
- Match source/task conditioning and target support, and preserve OM4 physical
  velocity supervision during mixed training. ERA5 contains no supplied future
  ocean SST/SSH. Keep native OM4 forcing as used in the comparator; no forcing
  or higher-resolution data change belongs in this wave.
- Score ensemble means with the comparator's global integrated-plus-spectral
  metrics and its frozen reference, and report CRPS/spread/member spectra
  separately. A CRPS training loss is not numerically comparable with its MSE
  training loss. Include inferred persistence and training climatology.
- Retain the full 96 monthly test origins, January 2015/2018/2021 annual cases,
  monthly OHC, global SST/SSH means, learned-interior initializer diagnostics,
  and independent-member temporal diagnostics. Diffusion readouts are still
  independently sampled along a deterministic latent trajectory.

Engaging currently has the 350-file observation bundle (21,888,642,966 bytes)
with those split counts. Its manifest/grid/statistics hashes are recorded in the
staged data receipt. A live read-only Torch check confirmed that its `SHA256SUMS`, `grid.npz` and
`statistics.npz` hashes exactly match the Engaging receipt. Full sample payload
revalidation remains a launch qualification; no new observation preparation or
transfer is needed.

## Implementation and validation

[GPU diagnostic](../../../src/samudra/experiments/diffusion_grain_diagnostics.py),
[Slurm job](../../../scripts/slurm_diffusion_grain.sbatch),
[analysis and figures](../../../scripts/report_diffusion_grain.py).
A CPU test verifies identical initial noise and final RNG state for all four
solver step counts. A synthetic identity-response fixture verifies gain and
correlation equal 1. These fixtures are not published scientific results.
