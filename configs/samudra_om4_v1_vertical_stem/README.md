# Samudra OM4 V1 Vertical Stem

These configurations mirror `samudra_om4_v1` but enable the vertical convolution
stem before the U-Net backbone.

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
- `kernel_size: 3` as the chosen depthwise Conv1d kernel
- `mid_channels: null` meaning it defaults to `num_depths`, so `19`
- `shared_weights: true` meaning the same depthwise stem is shared across all 3D variables

`boundary_vars_key: tau_hfds_hfds_anom` contributes 4 boundary variables
(`tauuo`, `tauvo`, `hfds`, `hfds_anomalies`), but that count is inferred by the
model build path rather than set explicitly in the YAML.