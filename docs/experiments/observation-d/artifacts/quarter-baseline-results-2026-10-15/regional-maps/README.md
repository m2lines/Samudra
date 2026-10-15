<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Regional maps matching the spectra

Each image has five source columns (selected quarter U-global, selected quarter observation-only, legacy 1° U-global seed1729, quarter OM4, quarter observations) and six rows (2015 day30/day365, 2018 day30/day365, 2021 day30/day365). These are individual five-day-mean snapshots from the same verified inputs as the [spectra](../spectra/README.md), not fields averaged across starts. Day30 represents January26–31 and day365 December27–January1.

All maps use the identical conservative wet support as the spectral analysis, fixed across sources, origins and leads. North is up and east is right. Gray denotes excluded cells. For each region and variable, one color scale is shared across all 30 panels, using the pooled 1st–99th percentile limits; values beyond these limits are clipped. ADT uses symmetric limits and KE a logarithmic color scale. Map values are physical fields before spatial-plane removal or tapering. Sea-level datums can differ between OM4 and observations; the spectra remove a plane/offset whereas these absolute ADT maps do not.

Quarter-degree cells are exactly 2×2pixels. Legacy 1° cells are exactly 8×8pixels (nearest-neighbor replication) so corresponding geographic areas have similar display sizes; this introduces no fine-scale information. The provenance receipt verifies every panel's constant pixel blocks and records native shapes, display rectangles, source hashes, common-support hashes and color limits. Latitude follows the native grid; this is a grid-index display, not a map projection.

Geostrophic KE maps show ½(u_g²+v_g²), derived from ADT. The matching spectra transform u_g and v_g separately before taking their half-sum; they do not Fourier-transform these scalar KE maps. Niño3.4 KE is omitted because geostrophy is undefined in the excluded equatorial strip.

| Region | SST | ADT | Geostrophic KE |
|---|---|---|---|
| Gulf Stream | [View](gulf-stream-sst.png) | [View](gulf-stream-adt.png) | [View](gulf-stream-geostrophic-ke.png) |
| Kuroshio | [View](kuroshio-sst.png) | [View](kuroshio-adt.png) | [View](kuroshio-geostrophic-ke.png) |
| Agulhas | [View](agulhas-sst.png) | [View](agulhas-adt.png) | [View](agulhas-geostrophic-ke.png) |
| Malvinas | [View](malvinas-sst.png) | [View](malvinas-adt.png) | [View](malvinas-geostrophic-ke.png) |
| Niño 3.4 | [View](nino-34-sst.png) | [View](nino-34-adt.png) | Unavailable |
| Tropical Pacific | [View](tropical-pacific-sst.png) | [View](tropical-pacific-adt.png) | [View](tropical-pacific-geostrophic-ke.png) |

Reproduce with `scripts/quarter_regional_maps.py --config /tmp/quarter-spectra-inputs/plot-config.json --output NEW_DIRECTORY`. Input restoration and source lineage follow the spectral artifact receipts. No new model inference or cluster allocation was required.
