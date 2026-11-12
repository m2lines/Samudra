<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Compression baselines: bits per value for ocean emulators

**Status: code complete and smoke-tested on synthetic data; not yet run on OM4.**
The sandbox that wrote this could not resolve `nyu1.osn.mghpcc.org`, so every
number below comes from a toy advection model and says nothing about the ocean.
`bash run_all.sh` produces the real report from the public 2° store
(`v2026-09/om4_twodeg`; there is no 4° release).

This is an experiment record, not intended to merge.

## Idea

Any forecaster with a predictive distribution is a lossless compressor: with
arithmetic coding, the true next value costs −log₂ P(its bin) bits. Fix one
quantization per channel and every model, from persistence to Samudra to the
all-fields classifier, gets a score in **bits per wet value**. Lower is better.

Two quests use that score:

1. **Metric (side quest).** Bits per value is a proper scoring rule, additive
   over cells, so it breaks down into maps, depth profiles and per-variable
   budgets. Differences between models on the same bins are exact; absolute
   bits depend on bin width.
2. **Baseline (primary quest).** A ladder of low-parameter predictors borrowed
   from lossless image and audio coding. A neural emulator should beat the top
   rung, in bits and in RMSE, before its complexity is justified.

## Protocol

| Item | Choice |
|---|---|
| Data | 2° OM4 five-day means, all 19 levels of `thetao`, `so`, `uo`, `vo`, plus `zos` (77 channels) |
| Task | Two-state history (t−1, t) to state t+1, five days ahead; no forcing (as in the classifier comparison) |
| Splits | Train 2003–2004, tune on 2010, test on 2015, all contiguous |
| Bins | 8,192 uniform bins per channel. The range comes from the 10⁻⁵ and 1−10⁻⁵ training quantiles plus 5% padding; the end bins absorb tails |
| Scoring | 8,000 test (time, cell) rows per channel, drawn with probability proportional to cell area. Cheap rungs are also scored on every wet test cell |
| Floor | All probabilities are mixed with 10⁻⁹ of a uniform distribution; the floor is the same for every model |

## The ladder

Every rung predicts the tendency x[t+1] − x[t] at one cell from 20 patch
features:

- 8 neighbour differences in a 3×3 patch
- the last tendency and its Laplacian
- the last tendencies of the other variables at the same level
- the tendencies at the levels above and below
- the anomaly from the channel mean
- two upwind-style advection terms, u·∂x/cos φ and v·∂y
- a bias term

Parameter counts are per channel.

| Rung | Predictive distribution | Params | Lineage |
|---|---|---:|---|
| `persistence` | Laplace(0, b) | 1 | |
| `ar1` | Laplace(a·tendency, b) | 2 | |
| `lpc` | Laplace(w·f, b), ridge regression per latitude band | 63 | FLAC linear prediction |
| `lpc_ctx` | `lpc` mean; Laplace scale from a 6×6 context table (gradient × last tendency) | ~110 | JPEG-LS / CALIC |
| `analog` | Kernel density over outcomes of the 32 nearest training patches (20k library) | 1 + library | Kernel analog forecasting |
| `lpc_analog` | `lpc` mean + kernel density over the neighbours' `lpc` residuals | 64 + library | |
| `mix` | Convex mixture of all rungs; weights by EM on validation log-likelihood | 6 | PAQ context mixing |

## Side-quest diagnostics

| Diagnostic | What it measures |
|---|---|
| Bits saved vs persistence, by depth and as maps | Where each model extracts information |
| Blosc residual bound | Blosc-zstd with byte shuffle (as in the OM4 stores) applied to the bin-index residual vs persistence. This is a generic-compressor upper bound on the conditional code length |
| Blosc compressibility ratio, forecast field / truth field | Below 1 means too smooth, above 1 too noisy. Meant for autoregressive rollouts; it is ≈1 for one-step forecasts by construction |
| Zonal correlation of `lpc` errors, and −½·log₂(1−ρ²) | A Gaussian estimate of the bits a joint (field-level) coder would save over any pointwise model |

## Ablation: compression distance vs Euclidean distance (`ncd_ablation.py`)

This tests the original gzip-classifier question. It uses the same library,
features and kernel density, and swaps only the neighbour search: Euclidean
distance, random neighbours, NCD with gzip, and NCD with blosc-zstd.

## What the synthetic smoke test shows

These are properties of the method, not results about the ocean.
`synthetic_smoke/` holds the tables, figure and JSON.

1. **The accounting is right.** `test_coding.py` checks that interval
   probabilities sum to 1, that a correct Gaussian model's cost matches the
   discretized entropy to within 0.01 bits, and that a mis-specified model costs
   more. On toy data the mixture is never worse than its best component.
2. **Bin ranges must be robust.** An unstable first version of the toy model
   produced outliers that widened every bin and drove costs to ≈0 bits. Hence
   the quantile-based ranges. Real channels with rare extreme values (deep `uo`,
   `vo`) should be checked with the `bin_width` column.
3. **Pointwise models leave spatially correlated errors on the table.** For toy
   velocities, the generic blosc residual bound (≈9.8 bits) beat every pointwise
   rung (≈10.1 bits). The `lpc` errors had zonal correlation 0.9–0.97, which
   gives an estimated joint-coding gap of 1.3–2 bits. Any per-cell model, the
   all-fields classifier included, pays this cost. Its size on OM4 is the most
   interesting open number here.
4. **One-step compressibility ratios are ≈1**, as expected. This diagnostic
   only becomes useful on rollouts.
5. **NCD neighbours were barely better than random.** On 19-byte quantized
   patches: Euclidean 4.45 bits, NCD with gzip 5.31, NCD with blosc-zstd 5.31,
   random 5.44. Compressor framing overhead swamps strings this short.

## Run it

```
pip install xarray zarr numcodecs fsspec aiohttp dask scikit-learn scipy matplotlib
bash run_all.sh                       # OM4 2°: caches about 1.5 GB, then fits 77 channels
SYNTHETIC=1 bash run_all.sh /tmp/c /tmp/o   # smoke test, about 2 min
python -m pytest -c /dev/null test_coding.py
```

The run writes `results.json`, `tables.md`, `bits_gain_vs_depth.png`,
`rmse_skill_vs_depth.png`, `bits_gain_maps.png` and `ncd_*.json`. The `mix`
rung's bits and RMSE are the numbers a model has to beat.

## Next steps once OM4 numbers exist

- **Score the all-fields classifier and Samudra in the same units.** For the
  classifier: 2° retrain, or 0.25° with this code at 0.25°, rebinned to these
  bins.
- **Add a causal raster-scan context rung.** Condition on already-coded
  neighbours of t+1, as JPEG-LS does, to measure the joint-coding gap directly
  rather than estimating it.
- **Run the compressibility-ratio diagnostic on Samudra rollouts**, as a cheap
  test for blurring and noise.
- **Include the forcings** (`hfds`, `tau*`, `wfo`) for comparison with
  Samudra's setup.
