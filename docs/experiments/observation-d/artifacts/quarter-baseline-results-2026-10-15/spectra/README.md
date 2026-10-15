<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Regional native-grid spectral diagnostics

Five fixed sources: validation-selected native-quarter U-global, native-quarter observation-only, legacy 1° U-global seed 1729, date-matched quarter-degree OM4, and quarter-degree observations. The forecast slices come from already completed continuous annual rollouts; no model inference, checkpoint selection, or training was performed for this analysis.

The six boxes and `spectrum` / `geostrophic` functions come unchanged from `scripts/global_physical_spectra.py`. Radial bin count is `max(3, min(height, width)//4)`; the minimum-three guard differs from the paper for short legacy boxes. Each snapshot is plane-detrended on its wet support, Hann-windowed, Fourier-transformed and azimuthally averaged. The ordinate is angular-wavenumber times PSD, with units °C² km for SST, m² km for SSH/ADT, and m² s⁻² km for geostrophic KE. The KE curve is half the sum of the two geostrophic velocity-component powers, not the power of speed squared and not temporal-anomaly EKE. The equatorial ±5° strip is excluded from geostrophic diagnostics, leaving Niño 3.4 unavailable. The Tropical Pacific's Cartesian distance conversion is approximate.

`*-snapshot-power-*` averages the three independently computed origin powers; `*-mean-field-*` instead transforms their average field. The latter follows the field-first averaging order in Samudra 2 Appendix A.5, but this three-origin, fixed-lead diagnostic does not reproduce the paper's multi-year cohort. There are no sampling confidence intervals. The JSON retains all individual-origin powers.

All fields keep their native resolution; legacy curves stop at their own Nyquist limit. Common support is a conservative geographic mask: every fine-grid cell mapping to a legacy cell must be valid for all three origins, both leads, and all five sources. The resulting coarse-cell intersection is mapped back to native grids without resampling values. SST, SSH and geostrophic-derivative support are computed separately. Masks are fixed across day 30/day 365. Fractions below count native grid cells, not area-weighted ocean coverage. Masking/taper correction produces a diagnostic pseudo-spectrum; it does not undo mask-induced spectral leakage. Day30/day365 axes are identical within each variable, region and aggregation order.

Exact target-bin arrays are verified across all five sources. Each January-start day30 slice is Jan26–Jan31 with Jan28 12:00 midpoint; day365 is Dec27–Jan1 with Dec29 12:00 midpoint, in 2015/2018/2021. OM4 uses exact matching nominal calendar year/month/day midpoint labels, rather than approximate nearest dates. Its compact store lacks explicit time bounds: five-day source time-mean metadata, midpoint convention and adjacent five-day cadence establish the interpreted bins. Matching uses the nominal Julian-calendar labels, not an absolute Julian-to-Gregorian epoch shift; these source details are preserved in extraction provenance.

Numerical outputs, per-origin powers, source hashes, target bins and support fractions are in `spectra.json.gz`; source extraction receipts and plotting config are preserved as compressed provenance. Decompression restores the original bytes. Inputs remain under `/tmp/quarter-spectra-inputs` on the analysis host; extraction receipts provide durable cluster source paths.

Reproduction (once the verified inputs have been restored to their configured paths):

```bash
MPLCONFIGDIR=/tmp/quarter-spectra-matplotlib .venv/bin/python scripts/quarter_regional_spectra.py \
  --config /tmp/quarter-spectra-inputs/plot-config.json \
  --output /tmp/quarter-spectra-reproduction
```

Ten focused tests cover conservative coastal support, invariant masks across leads, date mismatch rejection, native Nyquist limits, averaging order, velocity-component normalization, spatial plane removal and unavailable equatorial plotting. Figures were inspected after generation.

## Support and inputs

The four extratropical boxes and Niño3.4 have complete shared SST/SSH coverage. Niño3.4 has no allowed geostrophic support. In the Tropical Pacific, shared coverage is 88.89% of the box for SST, 88.49% for SSH and 72.50% for geostrophic KE (including the excluded equatorial strip). These masks retain approximately 95.3–97.6% of each source's otherwise-valid cells. Exact source-specific counts/fractions are in `support-fractions.csv`; the same mask is used at both leads.

`band-power-ratios.csv` gives an optional descriptive summary for the four fully wet extratropical regions: sum the mean-snapshot `k×PSD` samples with wavelengths 200–500km and divide by the corresponding observation sum. Each comparison uses the same four uniformly spaced wavenumber bins, so the shared bin width cancels. These are spectral-power ratios, not skill scores or phase/position accuracy, and are not used for model selection.

Exact input NPZ hashes:

| Source | SHA-256 |
|---|---|
| U-global selected (¼°) | `f985e7311f201428d367d249652243d3047cd0d1240d008a15c1373dc19be313` |
| Observation-only selected (¼°) | `aaa88c50107a029e8952560d0fb06ad7006911d6eeecdcfab4f84e0c1d794502` |
| Legacy U-global (1°, seed 1729) | `a83f237f43bdd12e8304776b60fb49dd5eeef0b7860f92a0c2403f78543c9040` |
| OM4 (¼°) | `763acfc4c0308ee43d4d5ee9d7213933cb2adcf6f05902016ccfa47a34b5d023` |
| Observations (¼°) | `830f24d7b5377133da00b55568262b8b2b89c5c2c1dc116c6c41f3459cf713fa` |

The plotting command above produces uncompressed `spectra.json`; the committed copy is gzip-compressed without numerical changes. Figures are 1820×1040 pixels at 130 dpi with lossless PNG optimization. Six snapshot-power and six mean-field panels share day 30/day 365 axis limits. Full input NPZs are not committed. The extraction harness and receipts cover the successful and failed CPU-only attempts; no GPU time was used.
