# Collected wave 1 results

OM4-only hindcasts at five-day cadence; these do not measure observational skill, daily outputs, or continuous long rollouts.

Both forecast models completed full held-out evaluation.

Normalized RMSE uses equal weights across levels within each variable. Subsurface temperature excludes SST.

| Model | Region | Lead | T RMSE vs persistence | S RMSE vs persistence | T RMSE vs climatology | S RMSE vs climatology |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ar | global | 5 | +4.1% | +2.6% | +19.9% | +5.1% |
| ar | global | 15 | +9.9% | +5.3% | +19.2% | +5.3% |
| ar | global | 30 | +16.8% | +8.3% | +16.5% | +3.7% |
| ar | tropics | 5 | +5.6% | +3.6% | +15.2% | +4.5% |
| ar | tropics | 15 | +11.7% | +6.5% | +15.9% | +5.2% |
| ar | tropics | 30 | +16.7% | +9.3% | +13.9% | +3.8% |
| ar | extratropics | 5 | +3.4% | +2.1% | +21.7% | +5.6% |
| ar | extratropics | 15 | +9.2% | +4.6% | +20.6% | +5.5% |
| ar | extratropics | 30 | +16.7% | +7.7% | +17.6% | +3.8% |
| direct | global | 5 | -2.5% | -3.6% | +12.0% | -1.6% |
| direct | global | 15 | +5.9% | +0.4% | +13.2% | -0.7% |
| direct | global | 30 | +10.5% | +0.2% | +8.0% | -6.4% |
| direct | tropics | 5 | -4.4% | -4.3% | +3.2% | -4.0% |
| direct | tropics | 15 | +4.0% | +0.4% | +5.4% | -2.2% |
| direct | tropics | 30 | +6.1% | +0.7% | -0.1% | -7.0% |
| direct | extratropics | 5 | -1.7% | -3.2% | +15.5% | +0.0% |
| direct | extratropics | 15 | +6.7% | +0.5% | +16.3% | +0.4% |
| direct | extratropics | 30 | +12.2% | -0.1% | +11.2% | -5.9% |

Positive percentages mean lower RMSE. Negative values mean worse errors. These aggregate CSVs do not support uncertainty intervals over forecast origins.

Direct pretraining and joint tuning were stopped early based on validation. Its selected joint checkpoint is the pre-joint starting model; joint tuning did not improve the selection metric.

Training was stopped using validation. Realized compute differs substantially between models; this is not an equal-compute architecture ranking. See the final report and final-accounting.psv for all attempts.
