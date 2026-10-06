<!-- SPDX-FileCopyrightText: 2026 Samudra Authors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Final global annual evaluation

Prepared for the final 8k OM4 / 8k observation checkpoint; **not yet GPU-evaluated**.
This evaluation does not change the training producer or the nine-origin monthly
validation procedure. It reports long trajectories separately from selection.

The existing Torch bundle is `/scratch/jr7309/data/obs-d-annual-instance-v1`.
On October 6, all **312 payload checksums** across the fixed January 1 origins
in 2015, 2018 and 2021 passed fresh read-back verification. Each case contains
19 history bins, 73 forecast bins and 12 monthly interior targets. No transfer
or reconstruction of the data is needed. The evaluator also verifies payload
hashes, exact five-day timestamps and monthly overlap weights when reading.

The global run requires two differences from the legacy annual evaluator:
missing surface inputs are zero in normalized space, and finite wet observations
outside ±60° remain in the metric support. The shared annual helpers now respect
the same `Samples` options as training; their defaults preserve the old protocol.
Tests check both conventions, future-observation exclusion, high-latitude errors,
and member exports versus the exact mean and RNG stream.

`samudra.experiments.diffusion_global_annual` loads the immutable final checkpoint
and verifies its data statistics, grid, observation manifest and score-reference
fingerprints. It uses eight members, 32 sampling steps and evaluation seed 4041729.
It refuses nonfinal task counts. The model initializes once per annual origin
and receives subsequent atmospheric forcing, with no future surface corrections.

Point metrics and full-state snapshots retain the existing annual output format.
An additional `members-<origin>.npz` saves every member's SSH, SST, temperature
at 550 m and salinity at 550 m for all 73 five-day leads, plus initial fields, masks
and coordinates. This supports day-30/day-365 member maps, power of the mean versus
mean member power, and temporal increment diagnostics from the same inference
pass. Monthly OHC uses overlap-weighted calendar averages. Instantaneous interior
fields must not be treated as date-matched gold against monthly products.

The members remain independently sampled physical readouts along a shared,
deterministic latent trajectory. They are not coherent sampled ocean trajectories;
temporal metrics must explicitly reflect that limitation. Monthly ensemble
calibration comes from the separate 96-origin final report, not the annual point
metrics alone.

The Torch launcher is `scripts/slurm_diffusion_global_annual.sbatch`. Set
`CODE_COMMIT`, `EVAL_CHECKPOINT`, `EVAL_NAME` and optionally `EVAL_STEPS` explicitly.
Use a spare GPU allocation so evaluation does not displace the eight training
GPUs. GPU execution and resource use remain to be verified and charged to the
existing 900-GPU-hour allowance.
