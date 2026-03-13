# Samudra OM4 V1 Vertical Stem

These configurations mirror `samudra_om4_v1` but enable the vertical convolution
stem before the U-Net backbone.

The current stem uses a two-stage residual design:

- a local depth convolution to encode nearby-level locality
- a residual MLP mixer over the full 19-level depth vector to add cheap
	column-wise expressivity without a large conv bottleneck

The `vertical_conv_stem` block in `model.yaml` is derived from the experiment
variable definitions in `train.yaml`:

- `prognostic_vars_key: thermo_dynamic_all`
- `boundary_vars_key: tau_hfds_hfds_anom`

The source of truth for these variable groups is:

- `PROGNOSTIC_VARS` in `src/ocean_emulators/constants.py`
- `BOUNDARY_VARS` in `src/ocean_emulators/constants.py`
- the `TensorMap` logic in `src/ocean_emulators/constants.py`, which splits the
	selected prognostic variables into 3D variable families, depth levels, and 2D
	variables

For `thermo_dynamic_all`, the prognostic variables are:

- 4 three-dimensional variables: `uo`, `vo`, `thetao`, `so`
- 19 depth levels for each 3D variable
- 1 two-dimensional prognostic variable: `zos`

This comes from the definition:

- `thermo_dynamic_all = [uo_*, vo_*, thetao_*, so_* over DEPTH_I_LEVELS] + [zos]`

and from the `TensorMap` rules that treat names with a depth suffix like
`thetao_0`, `thetao_1`, ... as 3D channels and names without that suffix like
`zos` as 2D channels.

So the stem parameters are set as:

- `num_3d_vars: 4` from `uo`, `vo`, `thetao`, `so`
- `num_depths: 19` because `thermo_dynamic_all` uses all 19 OM4 depth levels
- `num_2d_vars: 1` from `zos`
- `kernel_size: 5` to keep a local vertical neighborhood bias
- `mid_channels: 16` to keep the conv activations small at OM4 resolution
- `depth_mlp_hidden: 32` to add a cheap residual mixer over the full depth vector
- `shared_weights: false` so each of `uo`, `vo`, `thetao`, and `so` gets its own depth-conv stack

With these settings, each variable-specific stem has:

- local Conv1d block: `(1 * 16 * 5 + 16) + (16 * 1 * 5 + 1) = 177` parameters
- residual depth MLP: `(19 * 32 + 32) + (32 * 19 + 19) = 1,267` parameters
- total per variable: `1,444` parameters
- total across 4 variables: `5,776` parameters

`boundary_vars_key: tau_hfds_hfds_anom` contributes 4 boundary variables
(`tauuo`, `tauvo`, `hfds`, `hfds_anomalies`), but that count is inferred by the
model build path rather than set explicitly in the YAML.