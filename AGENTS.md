<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Samudra

Samudra is an ocean physics emulator: it uses neural networks to emulate physical ocean simulations, primarily the OM4
ocean simulation (soon LLC4320, and eventually both simulations and Earth observations of the ocean). The emulator
predicts future ocean states autoregressively, one 5-day step at a time. Samudra is intended to compose well with
other Earth system emulators, like Ai2's ACE atmospheric emulator. That coupled system is SamudrACE. One ambition of
the project is to produce a full climate emulation system by coupling multiple emulators.

The project is also a framework layer for developing ocean physics emulators across a variety of tasks and
approaches. Unfamiliar terms are defined in the [glossary](#glossary).

The repository holds three projects with **separate environments**:

| Path | What | Environment |
| --- | --- | --- |
| `/` (`src/samudra`) | The emulator: models, training, evaluation, metrics | `uv` (root `pyproject.toml`) |
| `data/` | `ocean_preprocessing`, which builds training datasets and observation products. Read `data/AGENTS.md` before working there. | mamba (`data/mamba_env.yaml`) |
| `rust/loader/` | Rust data loader extension | `uv sync --project rust/loader` plus the Rust toolchain; see `docs/rust-data-loader.md` |

Root `pytest`, mypy, and `uv sync` do not cover `data/` or `rust/loader/`.

## Rules

**Ask first**

* Before training or evaluating a model, or implementing a significnat feature, please provide a sketch of what you plan to do. Ask before launching SkyPilot or Slurm jobs, or changing significant parts of the codebase.
* Never delete or overwrite existing data in the public OSN bucket without asking. Writing new objects is fine when the
  developer asks for it.

**Repository hygiene**

* Don't commit large files (data, checkpoints, rendered figures) to git. They bloat every clone forever. Put heavy
  artifacts on OSN or a bucket. Log only lightweight figures to Weights & Biases, which charges for storage.
* Use `experiment/<name>` branches for experiment and results PRs (see [Experiments](#experimental-work-and-results-prs)).

**Code design**

* Prefer correctness over politeness.
  * Fail loudly. Don't swallow errors or add silent fallbacks. Fail fast instead of accommodating invalid or ambiguous input states.
  * Avoid hacks. Don't accommodate a bad design by adding cruft; refactor first in a separate change, then make the nice
  change.
* Prefer functional composition over global state. We are moving toward components that are built and wired together
  explicitly from config, and away from global objects with long lifetimes. Concretely:
  * Don't add new multitons. Pass state (e.g. normalization statistics) explicitly. The existing `Normalize` multiton
    is legacy.
  * Don't add new aggregators. Whole-rollout metrics belong in `samudra.metrics`.
* Keep the core train and eval loops performant. Profile (memray, py-spy, scalene) before and after changes there.
* Import from `samudra`, never from the legacy `ocean_emulators` package (context: we renamed this project).
* Don't assume a rectilinear grid. OM4's native grid is tripolar and LLC4320's is lat-lon-cap, so 2-D `lat`/`lon` must
  be carried with the data, not rebuilt by broadcasting 1-D axes. Branch on `is_rectilinear(grid_type)`
  (`src/samudra/constants.py`), not on the grid's name.
* Write Zarr **v2**. The root package pins `zarr<3`.

**Writing** (docs, docstrings, PRs, reports)

* Introduce anything before you refer to it. Readers are users or future maintainers, and either may be new to the
  project.
* PR descriptions: a few-sentence summary, then the bulk inside `<details>` with a `<summary>`. Say in the summary
  which agent or model you are.

## Definition of done

In this research codebase, there are two kinds of work here, and "done" means different things for each.

**Infrastructure** (library code, data pipelines, loaders: anything others build on). A change is done when:

1. The implementing agent believes the feature is complete, as the human described it.
2. One or more adversarial reviewers (another agent or an automated code reviewer) have assessed it for correctness
   and quality. All major feedback is addressed. Whether minor feedback blocks a merge is up to the human.
3. CI passes, and the data loader shows no automated performance regressions.

Tests come in two kinds. *Checks* are ordinary unit tests ("hand tests"). The implementer writes them to mirror CI and
to guard against regressions during refactors. Humans won't review them closely. *Searches* are property-based or
fuzz tests (we use Hypothesis), written by an adversary rather than by the implementer, to hunt for bugs the
implementer didn't think of. Criticism from an adversary is a gift. Not all implementations will undergo adversarial
fuzz testing, but will at least have to pass human and agentic code review.

While iterating, run only the tests that cover the code you changed, and let CI run the full suite.
`uvx pre-commit run` checks staged files only.

**Experiments** (training runs, ablations, results reports): see
[Experimental work and results PRs](#experimental-work-and-results-prs).

## Experimental work and results PRs

For experiments, exploratory work, and result reports that are not intended to merge:

* Use an `experiment/<descriptive-name>` branch, especially when committing results, figures, or run artifacts.
* Open a draft PR and apply the `experiment` label. State in the PR description that it is an experiment/results record
  and is not intended to merge.
* For an existing results PR on another branch, apply the `experiment` label; there is no need to rename its branch.

Either the `experiment/` head-branch prefix or the `experiment` PR label skips automatic PR CI jobs, including
pre-commit, CPU/GPU tests, data tests, container builds, and release smoke builds. Workflow entries may still appear
with skipped jobs, but they do not allocate runners. Adding the label does not cancel jobs already running; cancel
those separately when needed. Normal main-branch, merge-queue, scheduled, release, and manual workflows still run.

If experimental work becomes a merge candidate, create a normal development branch and PR with the intended changes
and no `experiment` label, then require the usual CI checks.
Continue running relevant local checks for experimental code and preserve run/data provenance in result reports.

## Development commands

```bash
uv sync --dev && source .venv/bin/activate          # set up the root environment

uv run pytest tests/test_stepper.py                 # tests for what you changed
uv run pytest -m "not manual and not cuda" -n auto  # the CPU suite CI runs
uv run pytest -m cuda                               # needs a GPU
uv run pytest -m manual -k test_whatever            # manual tests, run one at a time
uv run pytest --benchmark-only --benchmark-autosave # benchmarks

uvx pre-commit run              # lint/format/type-check staged files
uvx pre-commit run --all-files  # everything; also regenerates config JSON schemas
```

### Long-running commands and agent logs

Write logs from long-running commands (visualization, profiling, large test runs) to `agent-logs/` at the repository
root. It is gitignored, and it keeps a record a human can read later. Name each file
`<YYYYmmdd-HHMMSS>-<short-task>.log` so parallel agents don't overwrite each other:

```bash
mkdir -p agent-logs
LOG=agent-logs/$(date +%Y%m%d-%H%M%S)-viz.log
PYTHONUNBUFFERED=1 uv run python -m samudra.viz ... > "$LOG" 2>&1 &
tail -f "$LOG"   # or check back periodically
```

These logs are only for processes you run locally. Remote jobs keep their own logs: SkyPilot on the cluster
(`sky logs <cluster>`), and Slurm wherever the `sbatch` script's `--output` points. Record the cluster name or job ID
in your local log or PR so the human can find them.

## Where to look

Read the relevant doc before changing a subsystem. Use `ls` and search for the current file layout. This file
deliberately doesn't mirror it.

| Topic | Doc |
| --- | --- |
| Architecture overview, components | `docs/concepts/architecture.md` |
| Models (Samudra, samudra-multi, SamudraMini) | `docs/models/` |
| Configuration (YAML, `!include`, overrides, schemas) | `docs/config.md` |
| Stepper (single step vs. autoregressive rollout) | `docs/stepper.md` |
| Training, evaluation, visualization | `docs/train.md`, `docs/eval.md`, `docs/viz.md` |
| Datasets: what's published and where | `docs/data.md`, `docs/datasets.md` |
| Rust data loader | `docs/rust-data-loader.md` |
| Architecture search | `docs/search.md` |
| Running on SkyPilot / NYU Torch HPC | `docs/run.md`, `docs/torch.md` |
| Contributing, releasing | `docs/contributing.md`, `docs/releasing.md` |

Config presets ship inside the package (`src/samudra/configs/`), so an installed user can run
`samudra train samudra_om4/train.yaml` by preset name. Presets are resolved in `config_base.py`, and a filesystem path
takes precedence over a preset with the same name. Test-only fixture configs live under `tests/configs/` and are not
shipped.

## Known gotchas

These are the ones we know about. The project changes quickly, so assume there are others you haven't hit yet, and
add them here when you find them. Data-specific gotchas (mislabeled `areacello` units, calendars, means vs.
snapshots) are in `data/AGENTS.md`.

* **Python is pinned to 3.12** (`>=3.12,<3.13`) in both the root package and `data/`.
* **Shape annotations are enforced at test time.** pytest runs jaxtyping with beartype, so a `Float[Tensor, "batch
  lat lon"]` annotation is checked at runtime, not just documentation. A wrong annotation fails tests.
* **Two heat capacities.** `samudra.constants.CP_SW` is 3992 J kg⁻¹ K⁻¹, while `samudra.metrics.kernels.CP` is
  3850. OHC from the two code paths differs by 3.7%, so don't compare them directly (#975).
* **`Normalize` is global state.** Tests that touch it need a `MultitonScope` to stay isolated.
* **Channels are flattened by depth.** Prognostic variables appear as `thetao_0` … `thetao_18`; level 0 is the
  surface. Variable sets are chosen by key (e.g. `prognostic_vars_key: thermo_dynamic_all`) in
  `src/samudra/constants.py`.
* **Benchmarks run only on pushes to `main`**, not on PRs. A performance regression shows up as an alert after merge.
  If you touch the data loader or core loops, run the benchmarks locally and compare
  (`uv run pytest-benchmark compare`).
* **Config JSON schemas are generated, not committed.** `src/samudra/configs/schemas/` appears after
  `uvx pre-commit run --all-files`.

## Glossary

### Projects and simulations

* **OM4**: GFDL's global ocean–sea-ice model (built on the MOM6 ocean code), the ocean component of CM4. We emulate
  the OM4p25 configuration: nominal 1/4° resolution on a tripolar grid, forced by JRA55-do atmospheric reanalysis
  under the OMIP-2 protocol, 1958–2022. Its output is regridded to 1°, 1/2°, and 1/4° Gaussian grids and remapped to
  19 depth levels.
* **CM4**: GFDL's coupled climate model (atmosphere, ocean, land, sea ice). SamudrACE emulates CM4 pre-industrial
  control runs. Uses a no-leap calendar.
* **LLC4320**: an MITgcm global ocean simulation at 1/48° (~2 km) on a lat-lon-cap grid, with hourly output covering
  about 14 months (Sept 2011 – Nov 2012). It resolves submesoscale motion that OM4 does not.
* **ACE / ACE2**: Ai2 Climate Emulator, an ML atmosphere (and land) emulator with a 6-hour step.
* **SamudrACE**: Samudra coupled to ACE2. The atmosphere passes 5-day-mean surface fluxes and wind stress to the
  ocean; the ocean returns sea-surface temperature and sea-ice fraction.
* **Samudra / Samudra 2**: the original 1° ConvNeXt U-Net emulator (Dheeshjith et al., GRL 2025), and its successor,
  which scales to 1/2° and 1/4° with a dynamic variance-weighted loss.
* **samudra-multi, SamudraMini**: an encoder–processor–decoder model that trains on several resolutions at once, and a
  single PerceiverIO model for lightweight experiments.

### Grids and data

* **Tripolar grid**: OM4's native curvilinear grid. It moves the North Pole singularity onto two poles over land, so
  the Arctic isn't a hole.
* **Lat-lon-cap (LLC)**: MITgcm's grid of 13 faces. Regular within each face, not across faces.
* **Gaussian grid**: our training grid. Regular in longitude, Gaussian-quadrature latitudes (very close to uniform).
  Rectilinear, so 2-D lat/lon are an outer product of 1-D axes. `grid_type: gaussian` in metadata.
* **Prognostic variables**: the ocean state the model predicts and feeds back into itself (`thetao`, `so`, `uo`, `vo`,
  `zos`).
* **Boundary (forcing) variables**: inputs supplied from outside at every step and never predicted (`tauuo`, `tauvo`,
  `hfds`, `hfds_anomalies`, optionally `wfo`).
* **Rollout**: an autoregressive run where each prediction becomes the next input. Training rolls out a few steps;
  evaluation rolls out years (~580 five-day steps for 2014–2022).
* **Time step**: 5 days, so 73 steps per year. 1958–2022 has 4745 steps.
* **ARCO**: analysis-ready, cloud-optimized. Data that is preprocessed and chunked for direct cloud reads.
* **OSN**: Open Storage Network, an S3-compatible object store. Our public bucket is
  `https://nyu1.osn.mghpcc.org/m2lines-pubs/`.
* **Torch**: NYU's HPC cluster (Slurm), not PyTorch. See `docs/torch.md`.

### Variables and units

Typical ranges are order-of-magnitude sanity checks for 5-day means, not hard bounds. A value far outside them usually
means a unit, sign, mask, or normalization bug.

| Name | Meaning | Units | Typical range |
| --- | --- | --- | --- |
| `thetao` | Potential temperature: what a parcel's temperature would be if brought adiabatically to the surface | °C | −2 (seawater freezes ≈ −1.9 °C) to ~32 at the surface; 0–4 in the deep ocean |
| `so` | Practical salinity (PSS-78; dimensionless, written "psu") | psu | 33–37 in the open ocean; up to ~40 in marginal seas; lower near ice melt and river mouths |
| `uo`, `vo` | Eastward, northward velocity (rotated from the native grid's x/y) | m s⁻¹ | mostly < 0.5; ~1–2 in the strongest jets (Gulf Stream, Kuroshio, ACC); < 0.1 at depth |
| `zos` | Sea surface height above the geoid | m | about ±2 globally; lowest around Antarctica |
| `tauuo`, `tauvo` | Eastward, northward wind stress on the ocean surface | N m⁻² | ~0.1 typical; ~0.3–0.5 under strong westerlies |
| `hfds` | Net surface heat flux, **positive into the ocean** | W m⁻² | hundreds in either direction; largest losses over western boundary currents in winter. Global mean ≈ 0–1 |
| `hfds_anomalies` | `hfds` minus its day-of-year climatology | W m⁻² | tens |
| `wfo` | Net water flux into the ocean (precipitation − evaporation + runoff + ice melt), positive into the ocean | kg m⁻² s⁻¹ | ~10⁻⁵–10⁻⁴; 10⁻⁴ kg m⁻² s⁻¹ ≈ 8.6 mm/day |
| `lev` | Depth of each layer's center (19 levels, 2.5–6000 m) | m | |
| `dz` | Layer thickness (5 m at the surface to 1000 m at depth) | m | sums to 6500 |
| `areacello` | Grid cell area | m² (despite the `km2` attribute; see `data/AGENTS.md`) | |
| `ocean_fraction`, `mask_k` | Fraction of a cell that is ocean at each level; boolean wet mask at level `k` | — | 0–1 |
| OHC | Ocean heat content, ρ₀ c_p Σ θ dz with ρ₀ = 1035 kg m⁻³ and c_p = 3992 J kg⁻¹ K⁻¹ (MOM6's value; `samudra.metrics` uses 3850, see gotchas) | J m⁻² per column; J globally (1 ZJ = 10²¹ J) | |

### Physics for sanity checks

* **Heat budget.** The ocean exchanges heat only through its surface. So the change in global OHC over a step should
  approximately equal area-integrated `hfds` × Δt. For scale: 100 W m⁻² for 5 days (4.3 × 10⁷ J m⁻²) warms a 50 m
  mixed layer by about 0.2 °C (ρ₀ c_p ≈ 4.1 × 10⁶ J m⁻³ K⁻¹).
* **Salt conservation.** Total salt is conserved. Surface freshwater flux changes salinity only by dilution, so global
  mean salinity should not drift during a rollout.
* **Geostrophy.** Away from the equator, surface currents balance SSH gradients: u ≈ (g/f) k × ∇η, with
  f = 2Ω sin(lat) ≈ 10⁻⁴ s⁻¹ at mid-latitudes. A 1 m SSH change over 100 km implies ~1 m s⁻¹ flow. Use this to
  check `zos` against `uo`/`vo`. It breaks down near the equator, where f → 0.
* **Ekman transport.** Wind stress drives a depth-integrated surface transport of τ/f, at right angles to the wind
  (to the right in the Northern Hemisphere).
* **Stable stratification.** Density generally increases with depth. Where temperature increases with depth,
  salinity must compensate. Widespread inversions mean something is wrong.

## Ocean state invariants

These are properties that a correct dataset, pipeline, or model must satisfy whatever the resolution or architecture.
Use them as metamorphic tests: transform the input in a way whose effect on the output is known, and check that
effect. Also use them as smoke checks whenever a dataset, pipeline stage, or rollout is new. Violations of the
cross-resolution ones have caught real data-engineering bugs late in this project.

### Reference values

Computed with Xarray directly from the published v2026-09 OM4 stores (to be replaced by published reference statistics, #976), at time index 1000 (Sept 1971), with
ρ₀ = 1035 kg m⁻³ and c_p = 3992 J kg⁻¹ K⁻¹. Unless noted, the 1°, 1/2° and 1/4° stores agree to six significant figures.

| Quantity | v2026-09 OM4 | Real ocean | Note |
| --- | --- | --- | --- |
| Total `areacello` | 5.0767 × 10¹⁴ m² | 5.10 × 10¹⁴ m² (Earth) | 0.47% low: polar radius (#972) |
| Ocean surface area | 3.588 × 10¹⁴ m² | 3.61 × 10¹⁴ m² | same 0.47% |
| Ocean volume (Σ area × `ocean_fraction` × `dz`) | 1.476 × 10¹⁸ m³ | ~1.33–1.34 × 10¹⁸ m³ | ~10% high: bottom cells count their full `dz` (mean depth 4114 m vs ~3700 m). `samudra.metrics.kernels` integrates exact layer overlaps instead. #973 |
| Volume-mean temperature, salinity | 3.334 °C, 34.722 psu | ~3.5 °C, 34.7 psu | |
| Global OHC, relative to 0 °C | 2.0334 × 10²⁵ J | | inherits the volume bias |
| Area-mean `zos` | −0.0001 m | 0 by definition | CMIP `zos` is dynamic sea level with zero global mean |
| Surface KE, area mean | 0.0167 / 0.0186 / 0.0194 m² s⁻² | | rises with resolution (1° / 1/2° / 1/4°) |
| SST range | −1.91…32.52 / −1.98…32.71 / −2.01…32.75 °C | | widens with resolution |
| Minimum salinity | 0.03 psu | | real: river estuaries (Ob Gulf, Río de la Plata, Amazon). Don't clamp. #974 |

### Across resolutions (coarse-graining, regridding, filtering)

* **Linear integrals are conserved.** Total area, ocean area, volume, heat, and salt, and the area-mean `zos`, are all
  preserved. So are volume-mean temperature and salinity. Conservative regridding and the gcm-filters smoothers both
  preserve integrals by construction. A mismatch across resolutions beyond ~0.1% is a bug: wrong weights, mask, `dz`,
  units, or radius.
* **Quadratic quantities only decrease.** Variance, kinetic energy, EKE, and gradient magnitudes all fall as data is
  coarsened, because averaging can only remove variance (the law of total variance). A coarse field with *more* of
  them than its fine parent is a bug. This is also why `samudra.metrics` reduces before it regrids.
* **No new extremes.** A conservative regrid is a weighted average with non-negative weights. Each coarse value lies
  within the range of the fine cells it covers, so the global min and max can only move inward.
* **Spectra agree at shared scales.** At wavelengths that both grids resolve, the coarse and fine power spectra should
  match. They may diverge only near the coarse grid's cutoff.

### Within one dataset

* **Masks nest with depth.** A cell wet at level k+1 is wet at level k (`mask_{k+1} ⊆ mask_k`). `ocean_fraction`
  lies in [0, 1] and never increases with depth. Every 3-D field is NaN exactly where the masks say land, at every
  time step.
* **Physical bounds hold.** Temperature stays at or above the freezing point (≈ −1.9 °C at 35 psu, falling about
  0.054 °C per psu). Salinity is non-negative. Area-mean `zos` is ≈ 0 at every step.
* **The heat budget closes.** Per 5-day step, the change in global OHC tracks area-integrated `hfds` × Δt. On the
  1° store the correlation is 0.994 with a slope of 0.98. Over a year the net residual is ~0.06 W m⁻², about 1% of the
  5.6 × 10²² J seasonal swing.
* **Salt is conserved.** Total salt drifts by ~3 × 10⁻⁸ over a year. Any visible drift is a bug.
* **Time is regular.** 4745 steps exactly 5 days apart, in the dataset's own calendar.

### Model, loader, and rollout

* **Longitude-roll equivariance.** The globe is periodic in longitude, and the models pad circularly. So rolling every
  input (state, forcing, masks) by n columns should roll the output by n. This holds exactly only when n is a multiple
  of the network's total downsampling factor and learned positional channels are off. Latitude has no such symmetry
  (Coriolis and the poles).
* **Land values don't matter.** Perturbing raw values under the land mask must not change any prediction. The loader
  replaces them with `masked_fill_value`.
* **Batch independence.** In eval mode, a sample's prediction doesn't depend on batch size or its batch-mates.
* **Restart invariance.** In eval mode, rolling out 2N steps equals rolling out N steps and then continuing for N more
  from that state with the same forcing.
* **Loader equivalence.** The Rust and Python loaders, and any number of workers, ranks, or chunkings, yield identical
  samples. This is the sort of thing differential tests catch.
* **Round trips.** `denormalize(normalize(x)) == x`. Normalization statistics come only from the training interval.
* **Forcing matters.** Shifting the forcing by one step, or replacing it with climatology, should measurably hurt
  skill. If it doesn't, the model is ignoring forcing or the forcing is misaligned.

### What "good" means

The metrics in `samudra.metrics` score finished multi-year rollouts against observations:

* DUACS for surface geostrophic velocity and EKE, OISST for SST, ARGO-IAP for OHC by depth layer.
* Area-weighted RMSE with calendar-year block-bootstrap intervals.
* Residual variance after removing the trend and seasonal cycle, which catches a rollout that smooths away
  variability (variance collapse).
* Spectra compared in log space.

Training validation reports area-weighted RMSE, bias, and gradient magnitude against OM4. The metrics are about to
shift toward short-term skill. For that, compare against baselines: a forecast must beat **persistence** (x(t+1) =
x(t)) and **climatology**. Deep-ocean fields barely change in 5 days, so raw RMSE looks excellent even for a model that
has learned nothing. Judge skill relative to persistence, or on predicted tendencies.
