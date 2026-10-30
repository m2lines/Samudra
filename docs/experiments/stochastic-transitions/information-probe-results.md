<!--
SPDX-FileCopyrightText: 2026 Samudra Authors
SPDX-License-Identifier: CC-BY-4.0
-->

# Velocity initializer: added-information probe

Adding smoothed gold velocity improves held-out velocity reconstruction after 128 initializer-only updates. The improvement is larger with per-member MSE than with fair CRPS. For the raw MSE model, surface-u (`uo_0`, 2.5 m) RMSE falls from 0.10996 to 0.09144 m/s with the extra input. This surface metric is the square root of the spatially weighted MSE averaged equally over three dates. MSE also produces much lower ensemble coverage. The benefit of the extra information depends on the training objective. This short diagnostic does not establish whether missing inputs, optimization speed, or different objective optima limit the ordinary initializer after longer training.

All four arms start from the same raw B checkpoint at 2,000 mixed updates (1,000 OM4 and 1,000 observation updates). B has a stochastic Samudra ConvNeXt initializer and a deterministic processor. Here the processor is frozen and unused. Each arm receives 128 further OM4 initializer-only updates, effective batch eight, four sampled members, learning rate 1e-4, retained Adam state, and the original 0.1 initializer-loss coefficient over all 77 normalized fields. The sample and noise sequences match across arms. There is no cooldown or observation training in this probe.

Standard inputs are 19 consecutive five-day time bins, each with one SST and one SSH field, plus ERA5 history through the existing learned eight-to-three adapter and calendar/grid context. The scored final bin's SST and SSH are included in these inputs. Oracle inputs add contemporaneous gold u/v at all 19 depths, Gaussian-smoothed with sigma two grid cells and renormalized over wet area. A new bias-free 38-to-138-channel 1x1 adapter adds this information to the initializer conditioning. The adapter starts at zero in every arm; standard-input arms feed it zeros. All four starting predictions are exactly identical. The oracle is target-derived and cannot be used in an observational forecast. Regular-grid smoothing can cross narrow land barriers.

| Objective | Inputs | Weights | u RMSE (m/s) | v RMSE (m/s) | Velocity fair CRPS (m/s) | Coverage |
|---|---|---|---:|---:|---:|---:|
| Fair CRPS | Standard | RAW | 0.07111 | 0.05263 | 0.02216 | 60.4% |
| Fair CRPS | Oracle | RAW | 0.06877 | 0.05176 | 0.02164 | 60.7% |
| Member MSE | Standard | RAW | 0.06693 | 0.05132 | 0.02552 | 23.7% |
| Member MSE | Oracle | RAW | 0.05928 | 0.04908 | 0.02400 | 24.0% |
| Fair CRPS | Standard | EMA | 0.07192 | 0.05277 | 0.02309 | 55.7% |
| Fair CRPS | Oracle | EMA | 0.07118 | 0.05247 | 0.02291 | 55.9% |
| Member MSE | Standard | EMA | 0.07053 | 0.05242 | 0.02468 | 40.1% |
| Member MSE | Oracle | EMA | 0.06915 | 0.05200 | 0.02437 | 40.2% |

These are eight-member evaluations on three held-out initialization cases, identified by sample months 2014-01, 2014-04, and 2014-05. These cases are excluded from the probe and parent training: the OM4 and observation training cohorts end with July 2013 target windows, whose final target interval ends on 2013-08-04. The scored state is the final five-day bin in the preceding input history; these labels are not forecasts into the named month. Exact timestamps are in the [input receipts](results/information-probes/input-2014-01.json).

At each depth, squared error of the ensemble mean is averaged over wet cells with cosine-latitude weights. The u and v columns then take the square root after averaging those MSEs equally over three dates and 19 depths. They are not volume-weighted errors. Fair CRPS and coverage are averaged equally over three dates and all 38 velocity channels, after the same spatial weighting. Fair CRPS is mean absolute member error minus half the mean pairwise member distance using distinct-member pairs. Coverage is the fraction of gold values inside the eight-member minimum-to-maximum interval; it is not a 90% interval. Under independent continuous calibrated draws, that range would cover 7/9, or about 77.8%, in expectation. These physical-unit diagnostics differ from the normalized training loss.

EMA was reset to the raw parent and uses a 100-update half-life. After 128 updates its explicit contribution from the starting weights is still 2^(-128/100), about 41%. Its smaller improvement therefore cannot be read as a matched measure of the final raw model's learning. Raw and EMA results are kept separate throughout this table and the saved artifacts.

Using the equal-date, equal-depth MSE aggregation defined above, the oracle reduces raw u/v RMSE from 0.07111/0.05263 to 0.06877/0.05176 m/s under CRPS, and from 0.06693/0.05132 to 0.05928/0.04908 under MSE. Both objectives can use the extra information. Directly returning the smoothed gold velocity gives 0.05239/0.04511 m/s, so these short trained probes still leave substantial error relative to that available-input reference. This reference is neither a statistical lower bound nor an attainable forecasting baseline. The short budget, newly learned adapter, and single seed prevent separating all remaining optimization and representation limits. No rollout conclusion follows from this initializer-only test.

The [final-128 EMA spatial diagnostics](results/information-probe-placement/placement-amplitude.png) ([PDF](results/information-probe-placement/placement-amplitude.pdf)) show modest increases in member current amplitude and neighborhood scores with oracle inputs. Severe fine-scale power deficits remain. Averaged over three dates and eight individual members, the global cosine-weighted wet area with surface speed ≥0.3 m/s increases from 0.0255% to 0.0396% under CRPS and from 0.0058% to 0.0133% under MSE; OM4 has 6.82%. These are final EMA member statistics, separate from the raw ensemble-mean RMSE improvements above. [Metric definitions](placement-diagnostics.md) explain the neighborhood and spectral calculations; they do not isolate displacement from current amplitude.

Torch job `21423395` completed successfully on `gr103` in 21m54s, using one GPU: 0.365 GPU-hours including startup and evaluation. All four arms completed 128 finite updates. Their first-16-update mean GPU utilizations were 64.4–72.6%, above the 50% qualification gate; measured updates averaged 1.22–1.25 seconds. The CPU completion callback was job `21423435`. The [health record](results/information-probes/health.json), per-arm initialization proofs, and qualification receipts preserve these checks.

The [protocol](results/information-probes/protocol.json) records the common samples, parent checkpoint checksum, cache details, and objective settings. The [source manifest](results/information-probes/SOURCE.json) identifies immutable source `0edd2567556a7b9935a20dccff1da66f3455d56208adfdc768ef2d6bbf219a66`. Full-precision [summary values](results/information-probes/summary.json) and [per-case velocity metrics](results/information-probes/velocity-metrics.json.gz) reproduce the table. Evaluation uses the shared velocity diagnostics in the [frozen source bundle](sources/README.md); the probe is included in the [information-probe source archive](sources/information-probe/code.tar.gz).

Remote checkpoints and maps remain under `torch:/scratch/jr7309/stochastic-information-20261030/results`. Local fetched artifacts are under `/tmp/stochastic-information-results`; `/tmp/stochastic-information-final128-ema` selects only the final EMA maps for separate spatial-placement analysis. Large map arrays are not duplicated in this report's Git assets.
