# Surface-initialized ocean prediction: wave 1

OM4-only, five-day-cadence, 5–30-day hindcasts. These results do not
measure observational skill, daily prediction, or an eight-year rollout.

## initializer

Completion marker: **present**.

Code: `fe0f314ebb562f5b658e49fedea9c5d27062d245`; Slurm job: `17977027`.

initializer best validation loss: 0.0923793.

Peak allocated GPU memory (rank 0): 60.71 GiB.

Initializer validation RMSE (surface channels excluded):

| Mode | T | S | u | v |
| --- | ---: | ---: | ---: | ---: |
| inferred | 0.0635 | 0.0671 | 0.4270 | 0.4227 |
| climatology_with_observed_surface | 0.0775 | 0.0700 | 0.6733 | 0.7639 |

No held-out metrics available; no skill conclusion can be drawn.

## ar

Completion marker: **present**.

Code: `fe0f314ebb562f5b658e49fedea9c5d27062d245`; Slurm job: `18071751`.

pretrain best validation loss: 0.0497302.

joint best validation loss: 0.0929513.

Peak allocated GPU memory (rank 0): 73.07 GiB.

Global normalized RMSE, equal weight per depth level:

| Mode | Lead (days) | Subsurface T | S | u | v | SSH | SST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inferred | 5 | 0.0677 | 0.0726 | 0.4475 | 0.4515 | 0.0310 | 0.0246 |
| inferred | 15 | 0.0681 | 0.0723 | 0.4660 | 0.4987 | 0.0387 | 0.0346 |
| inferred | 30 | 0.0706 | 0.0738 | 0.4991 | 0.5567 | 0.0462 | 0.0441 |
| true | 5 | 0.0179 | 0.0219 | 0.2066 | 0.2428 | 0.0298 | 0.0235 |
| true | 15 | 0.0296 | 0.0353 | 0.3250 | 0.3792 | 0.0366 | 0.0342 |
| true | 30 | 0.0420 | 0.0481 | 0.4080 | 0.4801 | 0.0438 | 0.0441 |
| inferred_persistence | 5 | 0.0705 | 0.0745 | 0.5180 | 0.5821 | 0.0376 | 0.0254 |
| inferred_persistence | 15 | 0.0756 | 0.0764 | 0.6515 | 0.8125 | 0.0575 | 0.0582 |
| inferred_persistence | 30 | 0.0848 | 0.0804 | 0.7566 | 0.9342 | 0.0736 | 0.1003 |
| climatology | 5 | 0.0844 | 0.0764 | 0.6854 | 0.7899 | 0.0826 | 0.0684 |
| climatology | 15 | 0.0843 | 0.0764 | 0.6858 | 0.7888 | 0.0826 | 0.0671 |
| climatology | 30 | 0.0845 | 0.0766 | 0.6846 | 0.7895 | 0.0828 | 0.0684 |

30-day inferred velocity second-moment ratio to OM4: 0.911.

This amplitude diagnostic does not establish balanced or realistic circulation.

## direct

Completion marker: **present**.

Code: `fe0f314ebb562f5b658e49fedea9c5d27062d245`; Slurm job: `18013772`.

pretrain best validation loss: 0.0914476.

joint best validation loss: 0.11795.

Peak allocated GPU memory (rank 0): 62.97 GiB.

Global normalized RMSE, equal weight per depth level:

| Mode | Lead (days) | Subsurface T | S | u | v | SSH | SST |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| inferred | 5 | 0.0743 | 0.0776 | 0.5652 | 0.6442 | 0.0476 | 0.0585 |
| inferred | 15 | 0.0732 | 0.0769 | 0.4994 | 0.5538 | 0.0424 | 0.0437 |
| inferred | 30 | 0.0778 | 0.0815 | 0.5417 | 0.6058 | 0.0566 | 0.0569 |
| true | 5 | 0.0329 | 0.0307 | 0.4751 | 0.5995 | 0.0474 | 0.0544 |
| true | 15 | 0.0327 | 0.0357 | 0.4020 | 0.4930 | 0.0414 | 0.0406 |
| true | 30 | 0.0461 | 0.0504 | 0.4866 | 0.5731 | 0.0556 | 0.0543 |
| inferred_persistence | 5 | 0.0725 | 0.0749 | 0.5131 | 0.5743 | 0.0376 | 0.0254 |
| inferred_persistence | 15 | 0.0777 | 0.0772 | 0.6487 | 0.8092 | 0.0575 | 0.0582 |
| inferred_persistence | 30 | 0.0869 | 0.0816 | 0.7532 | 0.9298 | 0.0736 | 0.1003 |
| climatology | 5 | 0.0844 | 0.0764 | 0.6854 | 0.7899 | 0.0826 | 0.0684 |
| climatology | 15 | 0.0843 | 0.0764 | 0.6858 | 0.7888 | 0.0826 | 0.0671 |
| climatology | 30 | 0.0845 | 0.0766 | 0.6846 | 0.7895 | 0.0828 | 0.0684 |

30-day inferred velocity second-moment ratio to OM4: 0.699.

This amplitude diagnostic does not establish balanced or realistic circulation.

## Accounting

- Job 17957045: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17957597: FAILED, exit 1:0, 0.051 allocated GPU-hours.
- Job 17958094: FAILED, exit 1:0, 0.078 allocated GPU-hours.
- Job 17958284: COMPLETED, exit 0:0, 0.089 allocated GPU-hours.
- Job 17958513: CANCELLED by 4684506, exit 0:0, 0.223 allocated GPU-hours.
- Job 17958514: CANCELLED by 4684506, exit 0:0, 0.217 allocated GPU-hours.
- Job 17958779: COMPLETED, exit 0:0, 0.134 allocated GPU-hours.
- Job 17958780: COMPLETED, exit 0:0, 0.137 allocated GPU-hours.
- Job 17958872: CANCELLED by 0, exit 0:0, 4.370 allocated GPU-hours.
- Job 17958873: CANCELLED, exit 0:0, 0.000 allocated GPU-hours.
- Job 17958874: CANCELLED, exit 0:0, 0.000 allocated GPU-hours.
- Job 17958875: CANCELLED, exit 0:0, 0.000 allocated GPU-hours.
- Job 17958876: CANCELLED, exit 0:0, 0.000 allocated GPU-hours.
- Job 17958877: COMPLETED, exit 0:0, 0.000 allocated GPU-hours.
- Job 17959017: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17967105: CANCELLED by 4684506, exit 0:0, 0.117 allocated GPU-hours.
- Job 17968080: COMPLETED, exit 0:0, 0.476 allocated GPU-hours.
- Job 17968081: COMPLETED, exit 0:0, 0.311 allocated GPU-hours.
- Job 17970011: FAILED, exit 1:0, 0.061 allocated GPU-hours.
- Job 17971844: CANCELLED by 4684506, exit 0:0, 4.602 allocated GPU-hours.
- Job 17972528: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17972529: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17972530: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17972531: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17972532: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17977027: COMPLETED, exit 0:0, 0.222 allocated GPU-hours.
- Job 17977416: CANCELLED by 4684506, exit 0:0, 125.721 allocated GPU-hours.
- Job 17977417: CANCELLED by 4684506, exit 0:0, 17.503 allocated GPU-hours.
- Job 17977418: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17977419: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 17977420: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 18006099: COMPLETED, exit 0:0, 0.210 allocated GPU-hours.
- Job 18006100: CANCELLED by 4684506, exit 0:0, 8.748 allocated GPU-hours.
- Job 18006101: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 18013772: COMPLETED, exit 0:0, 0.281 allocated GPU-hours.
- Job 18013777: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 18068421: COMPLETED, exit 0:0, 0.286 allocated GPU-hours.
- Job 18068422: CANCELLED by 4684506, exit 0:0, 11.866 allocated GPU-hours.
- Job 18068423: CANCELLED by 4684506, exit 0:0, 0.000 allocated GPU-hours.
- Job 18071751: COMPLETED, exit 0:0, 0.384 allocated GPU-hours.

Total allocated GPU-hours in supplied job IDs: **176.086 / 576**.

Include every failed/replaced attempt in the job-ID list before approving recovery.

## Suggested next wave — requires approval

- ar: 30-day T/S MSE improvement over inferred persistence: 23.7%.
- direct: 30-day T/S MSE improvement over inferred persistence: 10.7%.
- Prioritize `ar` for replication if its advantage persists by depth and region.
- Inspect the true-versus-inferred validation gap to decide whether to spend the next wave on the initializer or evolution.
- Check velocity maps, spectra, and temporal continuity before claiming plausible circulation.
- Once prepared, introduce ERA5 surface-state forcing and observation initialization/interior evaluation, with daily SSH/SST heads.

No next-wave jobs are submitted by this report.
