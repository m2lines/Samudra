<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Rollout blow-up RCA outputs (2°)

Outputs behind [`../reports/rollout_blowup_rca.md`](../reports/rollout_blowup_rca.md), copied from the DTN.

| File | Produced by | Contents |
|---|---|---|
| `error_curves.json` | ad hoc script over the rollout predictions and `rca_counterfactual.json` | RMSE against lead for `thetao_6` and `thetao_8`: persistence and three ladder variants |
| `rca_counterfactual.json` | `rca_rollout.py --steps 40` (commit `307b89c9`) | Advection weight signs, the training range of the advection features, and the true-velocity counterfactual's RMSE by step. Its `seed` and `trace` fields come from an early, flawed seed criterion (step 2); ignore them, and use `rca_seed_trace.json` |
| `rca_seed_trace.json` | `rca_rollout.py --steps 34 --skip-counterfactual` (commit `c560a8da`) | Seed of the blow-up and its per-step, per-feature trace |
| `modes.json` | `rca_modes.py --steps 60 --eps 1.0` (commit `d094e000`) | Leading growth factor and mode composition of `lpc_linear` |
| `vonneumann_band0.txt` | `rca_vonneumann.py` (commit `69d166fb`) | Own-stencil amplification for four channels, tropical band |
| `coupled.json` | `rca_coupled.py` (commit `fd17ad3e`) | Coupled-operator spectral radius per band, uniform grid |
| `rca_error_growth.png` | rendered from `error_curves.json` and `modes.json` | The report's figure |

Note: "true-velocity counterfactual completed" in `rca_counterfactual.json`
reflects the old float64 finiteness check. Its RMSE series shows exponential
divergence.
