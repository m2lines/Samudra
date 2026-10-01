<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Held-out component table

All 96 origins, January 2015–December 2022. Checkpoints selected on validation. Composite uses fixed validation climatology denominators. Individual components below retain physical units. OHC is displayed in GJ/m²; CSV stores J/m². Lower is better.

| Arm / method | Composite | Spectral (dex) | SST (°C) | Velocity (m/s) | EKE (m²/s²) | OHC 0–700 (GJ/m²) | OHC 700–2000 (GJ/m²) |
|---|---:|---:|---:|---:|---:|---:|---:|
| primary / selected | 0.6578 | 0.4246 | 0.5049 | 0.1409 | 0.02596 | 0.6503 | 0.4155 |
| primary / selected-inferred-persistence | 0.6223 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 0.7241 | 0.4299 |
| primary / selected-inferred-anomaly-persistence | 0.6334 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 0.7381 | 0.4632 |
| primary / source-with-zero-forcing | 2.5800 | 0.4229 | 1.0243 | 0.1475 | 0.02651 | 6.3358 | 4.3127 |
| primary / source-inferred-persistence | 2.4586 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.3507 | 4.2835 |
| primary / inferred-anomaly-persistence | 2.4585 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.3483 | 4.2846 |
| primary / seasonal-climatology | 1.3781 | 1.6403 | 0.9039 | 0.2002 | 0.03093 | 0.8382 | 0.3729 |
| adapter-only / selected | 2.4573 | 0.4378 | 1.1556 | 0.1456 | 0.02634 | 6.3624 | 3.7753 |
| adapter-only / selected-inferred-persistence | 2.3260 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.4829 | 3.7340 |
| adapter-only / selected-inferred-anomaly-persistence | 2.3259 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.4792 | 3.7353 |
| adapter-only / source-with-zero-forcing | 2.5800 | 0.4229 | 1.0243 | 0.1475 | 0.02651 | 6.3358 | 4.3127 |
| adapter-only / source-inferred-persistence | 2.4586 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.3507 | 4.2835 |
| adapter-only / inferred-anomaly-persistence | 2.4585 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.3483 | 4.2846 |
| adapter-only / seasonal-climatology | 1.3781 | 1.6403 | 0.9039 | 0.2002 | 0.03093 | 0.8382 | 0.3729 |
| scratch / selected | 0.9827 | 0.6242 | 1.2228 | 0.1372 | 0.02785 | 1.0765 | 0.6913 |
| scratch / selected-inferred-persistence | 0.7170 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 0.9146 | 0.6789 |
| scratch / selected-inferred-anomaly-persistence | 0.7209 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 0.9121 | 0.6942 |
| scratch / source-with-zero-forcing | 2.5800 | 0.4229 | 1.0243 | 0.1475 | 0.02651 | 6.3358 | 4.3127 |
| scratch / source-inferred-persistence | 2.4586 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.3507 | 4.2835 |
| scratch / inferred-anomaly-persistence | 2.4585 | 0.2155 | 0.7912 | 0.1933 | 0.02472 | 6.3483 | 4.2846 |
| scratch / seasonal-climatology | 1.3781 | 1.6403 | 0.9039 | 0.2002 | 0.03093 | 0.8382 | 0.3729 |

The two selected-inferred controls use that arm’s selected initializer. The other persistence controls use the original source D initializer. Anomaly persistence advances the seasonal mean of interior T/S only; surface fields persist unchanged. Shared source/climatology controls are repeated to make each arm’s complete comparison explicit. Spectra cover broad scales only.
