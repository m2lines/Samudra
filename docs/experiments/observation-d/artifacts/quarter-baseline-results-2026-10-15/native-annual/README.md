<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

Native-quarter annual comparison of U-global and observation-only models, each at its validation-selected and fixed-6000 checkpoint. This task only processed saved forecasts.

Grace CPU job **135775** completed with exit 0:0 in 648 seconds on 16 CPUs and no GPUs: 2.88 allocated CPU-core hours, with Slurm MaxRSS 36,936,640 KiB. Earlier jobs 135759 and 135764 failed before Python because the shared image cache on betagg03 was unreadable. Their logs remain in the first two remote staging directories. The successful retry excluded that node and kept the image, source, inputs, and computation unchanged.

The canonical 42-file report, including map PDFs, is at `alpha:/projects/ny/lz1955/multiscale/jrusak/runs/2026-10-11-quarter-global/native-baseline-report-20261015-v3/job-135775/native-results/`. All 42 files (145,881,043 bytes) were downloaded and independently verified against `REPORT_COMPLETE.json.gz`. This repository includes lossless map PNGs without duplicate map PDFs. The complete results JSON and exact reporting script are gzip-compressed; decompression reproduces the hashes in the source receipts.

`rmse-scores.csv` contains per-origin rows and pooled rows. For each component, the three origin MSEs are averaged before taking the square root. Each pooled component RMSE is divided by the shared pooled training-only climatology RMSE at that lead. The score averages four components: SST, geostrophic velocity, OHC 0–700 m, and OHC 700–2000 m. ADT RMSE is reported separately and is excluded from this blend. Day-30 OHC uses the January calendar-month mean; day-365 OHC uses December. Native IAP OHC remains on its 360×720 grid. No test-fitted normalization or checkpoint reselection occurs.

Maps show anomalies for all three January origins and absolute fields for the final 2021 origin. Both checkpoint policies have their own forecast and initialized-persistence panels, plus identical observations. Every saved panel was checked pixel-for-pixel against the source colormap cells repeated exactly 2×2: 2880×1440 pixels for the 720×1440 model grid, without geographic resampling. Each map group's provenance JSON records source hashes, support, and color limits. These are native-quarter figures; common one-degree comparisons are separate artifacts.

Temperature profiles use the saved final inferred initial state and physical day-30/day-365 states at 19 model depths. They show cosine-latitude-weighted mean temperature over each depth's fixed wet mask, averaged equally over the three origins. The CSV also records RMS and individual-origin values. These are model-state diagnostics; no full-depth observational profile truth is inferred or upscaled.

Code, data, and forecast hashes are recorded in `results.json.gz`, `report-config.json.gz`, `REPORT_COMPLETE.json.gz`, and figure provenance files. The evaluation producer is `8bad9c34ac5fce91a7a3d8a7018299b4bcd3f0dc`; the training producer is `f091232ea8d7a8ee04ea14dd4cdbea5fa828f9ff`. Three focused profile tests passed. All 18 pooled four-component scores were independently recomputed from their physical-component RMSEs after download.

All JSON provenance receipts are gzip-compressed in this checkout; decompression restores the original filenames and bytes used by the archived hash manifests. This includes per-map provenance and `LOCAL_ARTIFACTS.json.gz`. Forecasts and figures are unchanged.
