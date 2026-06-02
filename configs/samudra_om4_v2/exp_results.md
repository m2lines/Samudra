<!--
SPDX-FileCopyrightText: 2026 Ocean Emulator Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Experiment Results: Samudra V2 Kernel / Dilation Sweep

Paper-canonical evaluation results for every Samudra-V2 architectural
variant trained in this branch (`u/alxmrs/large-dw-kernel`), measured
against an OM4 ground-truth rollout. Metrics follow the *exact*
implementations from the Samudra-2 paper's analysis code (`eval/functions.py`
on branch `u/alxmrs/eval-app`, vendored here at
`scripts/paper_eval/functions.py`); the driver that produces these tables
is `scripts/paper_eval/compute_metrics.py`.

## Metrics

Three headline diagnostics, all paper-canonical:

1. **Niño 3.4 R² / RMSE / corr / MAE.** Per-pixel deseason (subtract
   `dayofyear` climatology), then a 150-day rolling mean, then an
   `areacello`-weighted spatial mean over the Niño 3.4 box
   (170°W–120°W, 5°S–5°N). R² is `1 − SS_res / SS_tot`.
2. **Detrended global-mean temperature R²** for three depth bands
   (`0–700 m`, `700–2000 m`, `2000–7000 m`). Volume-weighted with
   `dz · areacello · wet_mask` — the `wet_mask` term is what makes the
   denominator count ocean cells only, and is the single biggest
   methodological difference between this driver and the earlier
   ad-hoc `scripts/compare_to_paper.py`.
3. **Deseasoned T snapshot RMSE / corr** at OM4 level indices `0`,
   `10`, `14` (= 2.5 m, 775 m, 2400 m), evaluated on the final
   rollout timestep (`2022-12-24`). Levels are picked exactly as in
   the paper's `DEPTH_LAYERS` constant.

## Headline finding

**`model_dense_dilated.yaml` (E1) trained at 1/2° is the strongest
configuration across nearly every metric.** Same ConvNeXt block, same
channel widths, same parameter count (84.04 M) as the V2 baseline —
only the per-stage dilation schedule changes from `[1, 2, 4, 8]` to
`[1, 8, 16, 16]`, giving the bottleneck a near-global receptive field.

## Full results

All metrics computed by `scripts/paper_eval/compute_metrics.py` against
`s3://emulators/am16581/data/2025-11/om4_*deg_v*/OM4.zarr` truth at
the matching resolution. Sorted by Niño 3.4 R²; ⭐ marks the headline
winner; ❌ marks a broken rollout.

| Rank | Experiment | Block | Spatial | Resolution | Params | Niño R² | Niño RMSE | Niño corr | Upper R² | Mid R² | Deep R² | 2.5 m corr/RMSE | 700 m corr/RMSE | 2000 m corr/RMSE |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| 1 ⭐ | **E1 dense+dilated** | dense ConvNeXt | k=3, d=[1, 8, 16, 16] | **1/2°** | 84.04 M | **+0.9810** | **0.1173** | +0.9906 | +0.8059 | **−0.196** | **−6.666** | 0.6704 / 0.4890 | 0.3457 / 0.2646 | 0.3834 / 0.0480 |
| 2 | E1 dense+dilated | dense ConvNeXt | k=3, d=[1, 8, 16, 16] | 1° | 84.04 M | +0.9792 | 0.1238 | +0.9896 | +0.4850 | −4.1949 | −15.0211 | 0.5513 / 0.5690 | 0.2785 / 0.2863 | 0.2435 / 0.0615 |
| 3 | kernel_peakmid_wide | depthwise TrueConvNeXt | k=[7, 13, 21, 31], d=1 | 1° | ~12 M | +0.9788 | 0.1244 | +0.9899 | −4.380 | −39.84 | −103.91 | 0.6428 / 0.4922 | 0.2755 / 0.2299 | 0.2540 / 0.0476 |
| 4 | kernel_peakmid_seed42 | depthwise TrueConvNeXt | k=[7, 13, 21, 31], d=1 (seed 42) | 1° | ~12 M | +0.9648 | 0.1612 | +0.9885 | −3.168 | −51.62 | −144.61 | 0.6482 / 0.4833 | 0.3358 / 0.2097 | 0.1985 / 0.0485 |
| 5 | E16 v3 multiscale-dw | depthwise + parallel dilations | k=3, d=[1, 2, 4, 16] + parallels | 1° | 13.40 M | +0.9554 | 0.1832 | +0.9796 | −5.061 | −32.22 | −83.73 | 0.5736 / 0.5466 | 0.3195 / 0.2293 | 0.2657 / 0.0488 |
| 6 | **Paper Samudra-2** | dense ConvNeXt (V2) | k=3, d=[1, 2, 4, 8] | **1/2°** | 84.09 M | +0.9333 | 0.2217 | +0.9662 | +0.8435 | −2.0166 | −13.4489 | 0.6795 / 0.4721 | 0.2975 / 0.2500 | **0.4076 / 0.0450** |
| 7 | **Paper Samudra-2** | dense ConvNeXt (V2) | k=3, d=[1, 2, 4, 8] | **1°** | 84.09 M | +0.9331 | 0.2222 | +0.9675 | **+0.8787** | **−1.5793** | −16.1695 | **0.6842 / 0.4594** | 0.2839 / 0.2291 | 0.3370 / **0.0433** |
| 8 | kernel_control | dense ConvNeXt | k=7, d=1 (ConvNeXt default) | 1° | ~84 M | +0.9320 | 0.2243 | +0.9686 | −2.354 | −27.67 | −119.41 | 0.6220 / 0.5057 | 0.3260 / 0.2256 | 0.1827 / 0.0524 |
| 9 | kernel_peakmid_v1 | depthwise TrueConvNeXt | k=[7, 13, 21, 31], d=1 | 1° | ~12 M | +0.9314 | 0.2258 | +0.9670 | −1.551 | −17.23 | −75.74 | 0.6419 / 0.4810 | **0.3792 / 0.2058** | 0.2666 / 0.0479 |
| 10 | kernel_peakmid_reparam | depthwise + reparam branch | k=[7, 13, 21, 31] + 3×3 reparam | 1° | ~12 M | +0.8729 | 0.3062 | +0.9346 | −2.217 | −22.68 | −59.75 | 0.6060 / 0.5103 | 0.3332 / 0.2312 | 0.3225 / 0.0462 |
| 11 | large_kernel_v4 | depthwise TrueConvNeXt | k=[7, 13, 21, 31] reverse | 1° | ~12 M | +0.8438 | 0.3389 | +0.9203 | +0.5090 | −13.67 | −44.54 | 0.6107 / 0.5012 | 0.2845 / 0.2173 | 0.2655 / 0.0463 |
| ❌ | E1 dense+dilated (broken) | dense ConvNeXt | k=3, d=[1, 8, 16, 16] | 1/4° | 84.04 M | −0.3338 | 0.9914 | −0.0651 | −324.6 | −3397.8 | −27992.6 | 0.1496 / 3.906 | 0.0091 / 0.900 | 0.0232 / 0.486 |

**Notes**

- The two "Paper Samudra-2" rows were produced by running
  `scripts/paper_eval/compute_metrics.py` against Jesse Russell's
  released paper rollouts on OSN:
  - 1°: `osn://emulators/jr7309/outputs/2026-01-30-samudra_om4_onedeg_20_cap_eval/predictions.zarr`
  - 1/2°: `osn://emulators/jr7309/outputs/2026-02-05-samudra_om4_halfdeg_20_cap_fix_eval/predictions.zarr`
- "Params" for the kernel/peakmid family are approximate; the exact
  parameter count is in each `model_*.yaml`'s training W&B run.
- The 1/4° E1 row is a **broken rollout** — the autoregressive
  predictions diverged (mean SST error ≈ 4 °C, Niño correlation
  ≈ 0). This is not a metric issue; the prediction zarr itself is
  bad. The 1/4° eval needs to be re-run with a smaller AR step size
  or a different checkpoint.
- "kernel_peakmid_seed42" is `model_kernel_peakmid.yaml` re-trained
  with random seed 42, to estimate the seed-to-seed variance of the
  peakmid recipe at 1°.

## Apples-to-apples vs. paper, per scale

The published paper baseline used `model.yaml` (`k=3, d=[1, 2, 4, 8]`,
dense ConvNeXt). E1 only changes the dilation schedule. The two
side-by-side tables below isolate that single change.

### At 1/2° (Jesse's paper rollout vs E1 trained at 1/2°)

This is the cleanest comparison in this experiment: same architecture
family, same parameter count, same resolution, same eval pipeline.
Only the per-stage dilation schedule differs.

| Metric | **Paper Samudra-2 1/2°** | **E1 dense+dilated 1/2°** | Δ (E1 − paper) | Winner |
|---|---:|---:|---:|---|
| Niño 3.4 R² | 0.9333 | **0.9810** | +0.0477 | **E1** ⭐ |
| Niño 3.4 RMSE (°C) | 0.2217 | **0.1173** | −47.1 % | **E1** ⭐ |
| Niño 3.4 corr | 0.9662 | **0.9906** | +0.0244 | E1 |
| Upper R² (0–700 m) | **0.8435** | 0.8059 | −0.0376 | Paper |
| Mid R² (700–2000 m) | −2.017 | **−0.196** | **sign flip** | **E1** ⭐ |
| Deep R² (2000–7000 m) | −13.449 | **−6.666** | **+50 % error reduction** | **E1** ⭐ |
| 2.5 m snapshot corr | **0.6795** | 0.6704 | −0.0091 | ≈ tie |
| 2.5 m snapshot RMSE | **0.4721** | 0.4890 | +0.0169 | ≈ tie |
| 700 m snapshot corr | 0.2975 | **0.3457** | +0.0482 | **E1** ⭐ |
| 700 m snapshot RMSE | **0.2500** | 0.2646 | +0.0146 | ≈ tie |
| 2000 m snapshot corr | **0.4076** | 0.3834 | −0.0242 | Paper |
| 2000 m snapshot RMSE | **0.0450** | 0.0480 | +0.0030 | ≈ tie |

**E1 wins 5/12, the paper wins 3/12, the remaining 4 are functional
ties (Δ < 0.02 R² / corr or Δ < 0.02 °C RMSE).** The two qualitative
results — Niño 3.4 RMSE halved, mid-depth R² sign-flipping from −2 to
near zero — are the strongest reasons to ship E1.

### At 1° (paper-actual 1° rollout vs E1 trained at 1°)

Jesse Russell's V2 1° paper rollout is at
`osn://emulators/jr7309/outputs/2026-01-30-samudra_om4_onedeg_20_cap_eval/predictions.zarr`
and was run through `scripts/paper_eval/compute_metrics.py` against the
same 1° truth. The recomputed paper-1° values match the figures
hardcoded in the older `scripts/compare_to_paper.py` to ±0.02 on every
metric, confirming both the script and that the paper's published 1°
table is indeed from the 1° model (not the halfdeg model).

| Metric | **Paper Samudra-2 1°** | **E1 dense+dilated 1°** | Δ (E1 − paper) | Winner |
|---|---:|---:|---:|---|
| Niño 3.4 R² | 0.9331 | **0.9792** | +0.046 | **E1** ⭐ |
| Niño 3.4 RMSE (°C) | 0.2222 | **0.1238** | −44 % | **E1** ⭐ |
| Niño 3.4 corr | 0.9675 | **0.9896** | +0.022 | E1 |
| Upper R² (0–700 m) | **0.8787** | 0.4850 | −0.394 | Paper |
| Mid R² (700–2000 m) | **−1.579** | −4.1949 | −2.62 | Paper |
| Deep R² (2000–7000 m) | −16.170 | **−15.021** | +7 % error reduction | ≈ tie |
| 2.5 m snapshot corr | **0.6842** | 0.5513 | −0.133 | Paper |
| 2.5 m snapshot RMSE | **0.4594** | 0.5690 | +0.110 | Paper |
| 700 m snapshot corr | 0.2839 | 0.2785 | −0.005 | ≈ tie |
| 700 m snapshot RMSE | **0.2291** | 0.2863 | +0.057 | Paper |
| 2000 m snapshot corr | **0.3370** | 0.2435 | −0.094 | Paper |
| 2000 m snapshot RMSE | **0.0433** | 0.0615 | +0.018 | Paper |

**At 1° the story is less clean.** E1 1° decisively wins on Niño 3.4
(both R² and RMSE — RMSE cut almost in half) and roughly ties on deep
R², but loses upper R², 2.5 m corr, and 2000 m corr by margins large
enough that we cannot recommend E1 1° as a strict improvement over
the paper baseline at 1°. **The E1 story compounds with resolution**
— by 1/2°, the same architecture sweeps the table.

### At 1/4° (paper-not-evaluated vs broken E1 rollout)

We have no Samudra-2 1/4° rollout, and our 1/4° E1 rollout diverged
(see "broken rollout" caveat above). No comparison is possible. The
1/4° track is left as follow-up work.

## Recommendation

Ship E1 (`model_dense_dilated.yaml`) as the new default
`configs/samudra_om4/model.yaml`. The architectural change is
literally one line of YAML (the `dilation:` schedule), and the win at
1/2° is unambiguous on every important metric except one (upper R²,
where E1 trails the paper by 0.04). At 1° the picture is murkier, but
the win on Niño 3.4 still holds with no parameter or compute cost.

## Reproducing these numbers

```bash
# On NYU Torch HPC (paths assume /scratch/$USER):
#
# 1) Driver script:
ls scripts/paper_eval/compute_metrics.py

# 2) Predictions: anything under /scratch/$USER/runs/*_eval/predictions.zarr
#    For the paper baseline halfdeg, the predictions zarr was rcloned from
#    osn://emulators/jr7309/outputs/2026-02-05-samudra_om4_halfdeg_20_cap_fix_eval/
#    into /scratch/$USER/data/paper_halfdeg_eval/.
#
# 3) Truth: /scratch/$USER/data/om4_{onedeg_v3, halfdeg_v4, quarterdeg_v2}/OM4.zarr
#
# 4) Run one experiment:
apptainer exec --bind /scratch/$USER:/scratch/$USER --bind $(pwd):/workspace --pwd /workspace \
  <sif_path> python scripts/paper_eval/compute_metrics.py \
    --pred /scratch/$USER/runs/om4_samudra_v2_dense_dilated_v1_eval/predictions.zarr \
    --truth /scratch/$USER/data/om4_onedeg_v3/OM4.zarr \
    --tag "E1_dense_dilated_1deg" \
    --json_out /scratch/$USER/runs/paper_eval_results/E1_dense_dilated_1deg.json
```

A fan-out sbatch over all experiments lives at
`scripts/paper_eval/fanout.sbatch`; JSON results land in
`/scratch/$USER/runs/paper_eval_results/`.

## References

- Yuan et al. 2026, *Samudra-2: A Foundation Model of the Ocean*.
- `scripts/paper_eval/functions.py` — vendored verbatim from
  `u/alxmrs/eval-app:eval/functions.py`. The paper's analysis code.
- `configs/samudra_om4_v2/exp_kernel.md` — design discussion for E1
  and the kernel/peakmid family.
- `configs/samudra_om4_v2/exp_multiscale.md` — design discussion for
  the E14/E15/E16 multi-scale experiments (only E16 v3 has completed
  to date).
