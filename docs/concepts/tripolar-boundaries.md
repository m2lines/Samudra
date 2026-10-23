<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Native tripolar geometry: area weights and boundary conditions

Research and implementation notes, 2026-10-23. Boundary handling below is a
proposal, not implemented or validated tripolar model support.

## Area weights

`CanonicalSource.area_weights` uses the source's `areacello`, normalized over
the horizontal grid. Static areas are exposed by the reader whether stored as
an xarray coordinate or data variable. Training validation, rollout validation,
and evaluation consume these weights. Source areas take precedence on Gaussian
grids too; legacy Gaussian stores without areas retain the previous cosine
latitude weights. Curvilinear sources without areas fail explicitly.

Areas must match the canonical horizontal grid, be finite and positive wherever
any selected prognostic or boundary channel is wet, and have a finite positive
total. NaN land areas receive zero weight; negative or infinite areas are rejected.
Per-variable/depth wet masks remain the responsibility of metric reductions.
Training checks the primary source's areas before optimization starts.

This fixes existing area-weighted metrics. It does not introduce area weighting
into the optimizer's currently unweighted spatial loss, change which source the
validation aggregator uses, or implement fold connectivity. Metrics from stores
with explicit areas can change relative to historical cosine-weighted results.

## What established models do

The Arctic seam is an identification of neighboring ocean cells on a folded
logical rectangle. It is not an impermeable northern wall. MOM6 explicitly
configures northern tripolar connectivity and uses halo exchanges; see
[MOM6 domain setup](https://mom6.readthedocs.io/en/main/api/generated/source/MOM__domains_8F90.html).

East/west boundaries are periodic. At the north, halo rows come from the other
side of the fold, reversing the zonal ordering and the inward row ordering.
The exact source row and column offset depend on the pivot and field staggering.
CICE documents separate T- and U-fold conventions, including redundant rows
that must be reconciled and excluded from double-counted global sums. See
[CICE tripole grids, section 3.1.2.2](https://cice-consortium-cice.readthedocs.io/en/cice6.3/user_guide/ug_implementation.html#tripole-grids).

Scalars copy without a sign change. Native grid-relative vector components
change sign because both local grid axes reverse across the fold. xgcm documents
this distinction and the four pivot conventions; its implementation is marked
experimental, so it is useful as a reference, not an unquestioned correctness
oracle. See [xgcm grid topology](https://xgcm.readthedocs.io/en/latest/grid_topology/).

For a corner-pivot, cell-centered scalar field with no stored halo or redundant
tracer row, a zero-based illustration is:

```text
north_halo[k, i] = field[Ny - 1 - k, Nx - 1 - i]

last interior row:   a b c d e f
first northern halo: f e d c b a
```

This formula is conditional on that convention. Verify the actual OM4 tracer
coordinates and row layout before using it. Ordinary reflection padding only
reverses the row direction; it does not connect to the far side of the fold.

## A directly relevant CNN precedent

Gregory, Bushuk, Adcroft, Zhang and Zanna (2023),
[Deep Learning of Systematic Sea Ice Model Errors From Data Assimilation Increments](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003757),
section 4.1, uses this treatment in a SPEAR sea-ice CNN. They add a four-cell
physical halo before four valid 3x3 convolutions: periodic zonal columns,
180-degree-rotated Arctic rows with ice-velocity signs reversed, and zeros
at the Antarctic land edge. Their array has the Arctic at the first row;
Samudra must use its own verified axis order. This is an actual ML precedent,
not just a finite-volume analogy. Their result concerns an offline sea-ice error
predictor, not validation of a long autoregressive ocean emulator.

## Implications for Samudra

The current ConvNeXt blocks and U-Net outer convolution use periodic x padding
and constant y padding (`models/modules/blocks.py`, `unet_backbone.py`). The
periodic bilinear upsampler also handles only the x seam. A complete change needs
to account for every spatial stencil, downsampling, and upsampling, not just
padding the raw input once by a small fixed width.

Native preprocessing (`--skip_regridding`) now retains grid-relative
`uo`/`vo` and `tauuo`/`tauvo` after interpolation to tracer centers. It preserves
`angle(y, x)` through flattening and writing, in degrees counterclockwise from
geographic east to native grid x. Each vector field records
`vector_basis="native_grid"`; preprocessing provenance records that basis and
an empty `vector_rotation` list. Regridded products continue to rotate to
geographic east/north and record `vector_basis="geographic"`.

This changes the native output convention. Earlier pipeline versions rotated
vectors even with `--skip_regridding` and discarded the angle. Existing stores
are not changed by this code update: regenerate into a new dataset version and
recompute its normalization statistics before training. Do not mix old
geographic-component statistics or checkpoints with the new native components.
Inspect the selected store's historical preprocessing and vector metadata;
current code alone does not establish the convention of an older release.
The native-grid fold sign rule applies to grid-relative physical vectors;
geographic components at the same physical location do not change sign merely
because the computational axes reverse.

Learned hidden channels also have no automatic physical vector parity. A
reasonable first implementation would use verified scalar fold connectivity
for hidden features and explicit basis-aware transforms for physical channels.
That establishes adjacency but does not make arbitrary learned convolutions
rotation-equivariant. Gregory et al.'s larger physical input halo with valid
convolutions avoids assigning physical vector signs to hidden channels; adapting
that to a deep U-Net would require a receptive-field and multiscale design.

Before selecting a treatment for production training:

1. Verify the prepared grid's north direction, pivot, redundant rows, geographic
   coordinates, wet masks, and vector basis using static metadata and a few rows.
2. Implement the matching index map and exercise it with labeled-cell tests,
   constant scalars, analytic geographic vectors, and corners where east/west
   periodicity meets the fold. For normalized native vector channels, apply the
   transform in physical units or include the mean/std correction.
3. Check each U-Net resolution, odd-size padding/cropping, interpolation, and
   input/output orientation. Test gradients through the halo gather.
4. Run a matched comparison of existing and fold-aware padding. Report Arctic
   seam errors and autoregressive stability as well as global, native-area
   weighted metrics. Connectivity correctness alone is not evidence of better
   forecast skill.

Land-mask treatment and native `dx`/`dy` for transport diagnostics are separate
questions from joining the Arctic seam.
