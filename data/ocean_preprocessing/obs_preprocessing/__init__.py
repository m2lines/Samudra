# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Fetch and prepare observation products for emulator evaluation.

Three products back the observation metrics in `samudra.metrics`:

| Product | Source | Cadence | Credentials |
| --- | --- | --- | --- |
| DUACS | Copernicus Marine (CMEMS) | daily, 0.125 deg | account required |
| OISST | NOAA NCEI | daily, 0.25 deg | none |
| ARGO-IAP | IAP, Chinese Academy of Sciences | monthly, 0.5 deg | none |

The pipeline has two stages. `download` fills a raw archive of the products as
distributed, and is restartable. `prepare` derives the analysis-ready stores:
each product stays on its **own native grid** and is aligned in time onto the
OM4 5-day cadence with a centered 5-day rolling mean, because the metrics
require exactly matching timestamps and refuse to interpolate in time.
"""
