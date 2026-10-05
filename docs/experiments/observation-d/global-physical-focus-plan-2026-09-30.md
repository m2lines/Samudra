<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Focused report: global physical-only at 30 days

Authorized September 30. Existing global physical-only weights only; no retraining.
Compare 50 / 500 / 2,000 / 8,000 observation-update snapshots (324 / 2,167 /
4,949 / 8,000 OM4 updates). Earlier checkpoint monthly and continuous-year
inference uses the same pinned 79025a163817a577ab81af95b401f5cb0563cd12 evaluator.
Existing final evaluations are reused. Slurm inputs bind actual fixed task counts.

Add exact last-history-bin surface observations to initializer maps. Interior
initial maps may use the preceding December IAP monthly mean as explicitly labeled
context, not instantaneous truth. Internal U/V reference panels use a geostrophic
proxy; those fields were not directly observed or targeted on the observation task.

Broader spectra cover the six paper regions, including Niño 3.4 and Tropical
Pacific. Twelve monthly 2015 day-30 forecasts; average powers, not forecast leads.
Show final and training stages with OM4 1°, the actual training-grid observations,
and native observations (SST 0.25°, ADT 0.125°), with exact five-day support.
OM4 native five-day averages are overlap-weighted when calendar bins differ.
For land-containing boxes use explicit masked pseudo-spectra, common coarse wet
support, plane removal and Hann taper. These are additional diagnostics, not
replacement checkpoint selection metrics. Geostrophic KE is half the component
spectra and unavailable within ±5°; it is not the spectrum of scalar EKE nor a
claim to temporal eddy energy from a short cohort.

Annual final-model plots show undetrended absolute, area-weighted SST/SSH and
monthly OHC 0–700 / 700–2000 means, with observation, inferred-state persistence
and training-only climatology. Fixed common support throughout each year;
report its area fraction. Preserve three annual cases and full-grid annual maps.
No restricted-latitude control comparison in this report.

Native extraction: Empire AI Grace job 110827, COMPLETED 0:0. Source stores:
/mnt/home/jrusak/data/obs_full_range/prepared/{oisst,duacs}.zarr.
The 139,335,680-byte archive and every member have verified SHA256 read-back.
Native/coarse temporal/remapping equivalence is checked against existing targets.

Initial six Torch eval jobs 18883746–18883751 failed before inference because
missing torchrun argument separator let --run be parsed by the launcher. Retry
jobs 18884170–18884175 add the separator; retain all failed records/accounting.
CPU compact collector job 18884176 follows all retry evaluations after success.
Monitor without a timer and include actual allocated GPU-hours, including retries.

CPU collector 18884176 was stopped after diagnostics showed 74 GB repeatedly
read from NPZ arrays. The corrected extraction loads each member once and uses
new source/output paths; original source and partial arrays are preserved. The
recovery changes report processing only, not inference or scientific metrics.

All six replacement evaluations completed 0:0. Replacement CPU collector 18886425
completed 0:0 in 164 seconds; canceled collector 18884176 is retained. New
inference accounting including failures is 0.245 allocated GPU-hours. The compact
854,917,120-byte archive and all members passed SHA256 read-back. Native/coarse
exact-bin equivalence, pixel geometry and regional transform checks passed.

[Completed focused report](global-physical-day30-2026-09-30.md) includes all four
stages, observation references, six regional spectra panels at day 30 only, and
three undetrended annual global-mean cases. Source hashes, checkpoint lineage,
accounting, transfer receipts and numerical results accompany the figures.
No training or checkpoint-selection protocol changed.
