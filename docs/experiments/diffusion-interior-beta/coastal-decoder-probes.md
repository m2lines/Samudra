<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Coastal, conditioning and latent-amplitude diagnostics

**Completed:** [results, matched member maps and numerical exports](coastal-decoder-results.md).
Beta job 212173 completed all 84 cases without training in 0.06778 allocated
B200 GPU-hours. The original protocol below is retained for provenance.

User-authorized follow-up to the completed [continuation campaign](interventions-v2-results.md).
Question: why do individual diffusion draws remain grainy, why is annual error
concentrated near land, and would normalization of the persistent latent state
help? These are small frozen-checkpoint diagnostics, not a new training wave.
Initial allocation: one B200 for at most 90 minutes; total diagnostic/recovery
ceiling eight allocated GPU-hours. No training or new data is required.

Use the saved control and replay-bound 256-update checkpoints, the same 2015
OM4 surface initialization and forcings, and physical days 30 and 365. Preserve
32 denoising steps, 0.5 correlated-noise variance fraction, matched random draws,
and the original masks. Report SST/SSH errors by distance to land and individual
maps; two draws are a mechanistic probe, not a calibration benchmark.

1. Denoise the same true field corrupted at sigma 0.05, 0.3, 1 and 5 with the
   corresponding latent, zero latent, or origin latent. Compare denoised error
   with the input error. Counterfactual conditioning is not a valid forecast.
2. Cross the two decoders with the two saved latent trajectories under identical
   sampling seeds. Both started at the same parent, but their latent coordinate
   systems can change during fitting; crossing does not eliminate that confound.
3. At day 365, extend each noisy channel into land from its nearest wet cell
   only before the denoiser stem. Preserve the original EDM skip, sampler,
   output mask and all weights. This inference-only distribution change can
   expose sensitivity to boundary handling, but does not validate a trained
   masked-convolution replacement.
4. Compare channel-RMS limiting only before readout with limiting after every
   five-day recurrent update. Use twice the encoded-origin channel RMS, with
   the same 0.1 scale floor as the soft training penalty. Do not overwrite
   baseline states. This is a controlled amplitude probe, not GroupNorm training;
   changing normalization can discard meaningful state information.

The processor already has GroupNorm inside its residual blocks; its persistent
state update is an unnormalized residual addition. Decoded noisy fields are not
fed back into the processor. OM4 replay trains one-step denoising of noised true
fields conditioned on late latents; it does not backpropagate through a complete
noise-to-field sampler at those late leads. Observation supervision uses two
sampled members and fair CRPS, including wet-neighbor increments, with interior
fields time-averaged before scoring. Neither training loss scores an eight-member
mean. The eight-member mean is used for the published point-error evaluations.

CPU diagnostics already checked that coast-proximate error is disproportionate
in the single native 2015 case, including below 60 degrees absolute latitude.
Mask association does not establish causation: dynamics, data and geometry also
vary near coasts. Nearest-wet indices and distance use periodic longitude;
distance is in grid cells rather than physical kilometers.
