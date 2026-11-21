<!--
SPDX-FileCopyrightText: 2026 Samudra Authors

SPDX-License-Identifier: CC-BY-4.0
-->

# Ocean physics for data scientists

This page is for Python programmers who are comfortable with NumPy, Xarray, and PyTorch but are new to oceanography.
It explains what Samudra's data represents and which physical laws it obeys. It also covers how to use those laws to
catch bugs, since a field can be numerically plausible and still physically impossible.

You don't need any of this to run Samudra. You will need it to change the data pipeline, add a metric, or judge
whether a model's output is reasonable. Each claim that isn't textbook material links to a source in
[References](#references).

## The simulation we emulate

An **ocean general circulation model** (GCM) solves the equations of fluid motion on a rotating sphere: momentum,
conservation of mass, heat, and salt, and an equation of state linking temperature and salinity to density. It steps
them forward with time steps of minutes. Running one for decades on a fine grid takes a supercomputer. An
**emulator** is a neural network trained to reproduce the GCM's output at a fraction of the cost.

Samudra's primary training data is **OM4**, the ocean and sea-ice model from NOAA's Geophysical Fluid Dynamics
Laboratory (GFDL), built on the MOM6 ocean code [[Adcroft 2019](#ref-adcroft)]. We use the configuration with a
nominal 1/4° grid. It is driven by an observed atmosphere (a reanalysis) following the OMIP-2 protocol, 1958–2022
[[Griffies 2016](#ref-griffies); [Dheeshjith 2025](#ref-samudra)]. "Driven by" matters: the atmosphere is an *input*.
The ocean simulation never changes the winds or air temperature it receives.

OM4 is also the ocean component of **CM4**, GFDL's coupled climate model, where the atmosphere and ocean do affect
each other.

### What one training sample is

The emulator learns a map from the ocean's state now, plus the forcing over the next interval, to the ocean's state
5 days later. At inference it feeds its own output back in, an **autoregressive rollout**, for thousands of steps.

* **Prognostic variables** make up the state the model predicts: temperature, salinity, two velocity components at
  19 depths, and sea surface height. That's 77 channels.
* **Boundary (forcing) variables** are given at every step and never predicted: wind stress, surface heat flux, and
  optionally freshwater flux.

Small errors compound over a rollout, which is why stability, and not just one-step accuracy, has historically been
the hard part [[Dheeshjith 2025](#ref-samudra); [Samudra 2](#ref-samudra2)].

## How the data is laid out

```python
import xarray as xr

ds = xr.open_zarr("https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_onedeg/OM4.zarr")
```

The store is cloud-optimized: opening it is instant, and data is fetched only when you compute something.

### Horizontal grid

OM4 runs on a **tripolar grid**. It's curvilinear, and it moves the North Pole singularity onto two points over land
so the Arctic Ocean doesn't contain a degenerate cell. Our pipeline regrids it to a **Gaussian grid**: regular in
longitude, with latitudes at Gaussian-quadrature points (nearly uniform). The training releases come at 1°
(180 × 360), 1/2° (360 × 720), and 1/4° (720 × 1440).

Regridding is **conservative**: each target cell's value is the area-weighted average of the source cells it
overlaps, so area integrals are unchanged. That property is the basis of several checks
[below](#across-resolutions).

Grid cells shrink toward the poles. Any global statistic must be **area-weighted**, never a plain `.mean()`:

```python
area = ds.areacello                       # full cell area in m², land included
ocean_area = area * ds.ocean_fraction.isel(lev=0)
sst = ds.thetao_0.isel(time=0)
global_mean_sst = (sst * ocean_area).sum() / ocean_area.where(sst.notnull()).sum()
```

### Vertical grid

The depth axis has 19 fixed levels whose thickness grows with depth: from 5 m at the surface to 1000 m in the abyss.
`lev` holds the depth of each layer's center and `dz` its thickness. Most of the ocean's structure, and most of what
changes in 5 days, is in the top few hundred meters. Thin layers go where the action is.

Depth is flattened into channels: `thetao_0` is surface temperature and `thetao_18` is the deepest. A cell that is
land at some depth is `NaN` there. `mask_k` is the boolean wet mask at level `k`, and `ocean_fraction` is the
fraction of each cell that is ocean at each level (cells along coastlines are partly land).

### Time

One step is nominally 5 days. Each calendar year holds 73 steps, re-anchored to January 3, so the step across a
leap-year boundary (December 28 to January 3) is 6 days. There are 16 of those in 1958–2022, and 4745 steps in all.
Compute Δt from the time coordinate rather than assuming 5 days. OM4 uses a Julian calendar (`cftime`
objects, not `numpy.datetime64`). CM4 uses a no-leap calendar. Never assume these align.

## Variables and units

| Name | What it is | Units | Typical range (5-day means) |
| --- | --- | --- | --- |
| `thetao` | **Potential temperature**: a parcel's temperature if moved to the surface without exchanging heat. It removes the warming from compression at depth. | °C | −2 to ~32 at the surface; 0–4 in the deep ocean |
| `so` | **Practical salinity**: dissolved salt, measured by conductivity. Dimensionless, conventionally written "psu". | psu | 33–37 in the open ocean; ~40 in the Red Sea; near 0 in large estuaries |
| `uo`, `vo` | Eastward and northward velocity | m s⁻¹ | mostly < 0.5; 1–2 in the strongest currents (Gulf Stream, Kuroshio, Antarctic Circumpolar Current); < 0.1 at depth |
| `zos` | **Sea surface height** relative to the geoid (the shape the sea surface would take at rest). Our convention removes the global mean. | m | about ±2 |
| `tauuo`, `tauvo` | Wind stress: the force per unit area the wind exerts on the surface | N m⁻² | ~0.1; up to ~0.5 under strong westerlies |
| `hfds` | Net heat flux into the ocean (sunlight, infrared, evaporation, conduction, sea ice) | W m⁻² | hundreds in either sign; global mean near 0 |
| `hfds_anomalies` | `hfds` minus its day-of-year climatology | W m⁻² | tens |
| `wfo` | Net water flux into the ocean: precipitation − evaporation + rivers + ice melt | kg m⁻² s⁻¹ | 10⁻⁵–10⁻⁴; 10⁻⁴ ≈ 8.6 mm/day |

The ranges are order-of-magnitude sanity checks, not bounds. Sign conventions follow CMIP: fluxes are positive *into*
the ocean. A typical symptom of a sign or unit bug is a value that's off by a factor of 10³–10⁶, or has the wrong sign
everywhere at once.

## Physics in a few ideas

You can go a long way with these. Standard references are Talley et al. (2011) for descriptive oceanography and
Vallis (2017) for the dynamics [[Talley 2011](#ref-talley); [Vallis 2017](#ref-vallis)].

Each idea has an **In Python** block that demonstrates it on the published 1° store. The blocks build on one another
(constants such as `RHO_0` and `OMEGA` carry forward), so run them in order, starting from the `ds` opened
[above](#how-the-data-is-laid-out). Each one reads a few hundred megabytes at most. The results shown came from
running them on the NYU Torch data transfer node; expect them to be slower from a laptop.

### Heat goes in and out only at the surface

The ocean's heat content in a water column is

    OHC = ρ₀ c_p Σ_k θ_k dz_k        (J m⁻²)

<details markdown>
<summary>In Python</summary>

`ds` is the store opened [above](#how-the-data-is-laid-out). The sum over `k` becomes a sum over a stacked `lev`
dimension.

```python
import xarray as xr

RHO_0 = 1035.0  # reference density ρ₀, kg m-3
C_P = 3992.0    # specific heat c_p, J kg-1 K-1

snap = ds.isel(time=1000)
theta = xr.concat([snap[f"thetao_{k}"] for k in range(19)], dim="lev")  # θ_k: (lev, y, x), °C
ohc = RHO_0 * C_P * (theta * ds.dz).sum("lev")                          # (y, x), J m-2
ohc = ohc.where(snap.thetao_0.notnull())                                # keep land as NaN
```

OHC here is measured relative to 0 °C, so a column of sub-zero polar water has negative OHC. Only *changes* in OHC
have physical meaning.

</details>

with reference density ρ₀ = 1035 kg m⁻³ and specific heat c_p = 3992 J kg⁻¹ K⁻¹, the values MOM6 uses. Ocean currents
move heat sideways, which cancels in a global sum, and geothermal heating through the seafloor is tiny. So the
**global** OHC changes only through `hfds`:

    ΔOHC_global ≈ Σ_cells hfds × area × Δt

<details markdown>
<summary>In Python</summary>

Continuing from the previous snippet. `hfds` is a 5-day mean centered on each timestamp, so the heat entering
between two consecutive states is the average of their two fluxes.

```python
volume = ds.areacello * ds.ocean_fraction * ds.dz            # (lev, y, x), m3
surface_area = ds.areacello * ds.ocean_fraction.isel(lev=0)  # (y, x), m2

def global_ohc(t):
    theta = xr.concat([ds[f"thetao_{k}"].isel(time=t) for k in range(19)], dim="lev")
    return RHO_0 * C_P * float((theta * volume).sum())        # J

t = 1000
DT = (ds.time.values[t + 1] - ds.time.values[t]).total_seconds()  # Δt, s: 5 days, or 6 across a leap-year boundary
d_ohc = global_ohc(t + 1) - global_ohc(t)                    # ΔOHC_global, J
hfds = 0.5 * (ds.hfds.isel(time=t) + ds.hfds.isel(time=t + 1))
heat_in = float((hfds * surface_area).sum()) * DT            # Σ hfds × area × Δt, J
print(d_ohc / heat_in)                                       # 0.98
```

Single steps typically land within about 15% of 1. Over a year, the per-step correlation is 0.994.

</details>

For intuition, ρ₀ c_p ≈ 4.1 × 10⁶ J m⁻³ K⁻¹. A strong 100 W m⁻² flux for 5 days warms a 50 m surface layer by only
about 0.2 °C. Water's large heat capacity is why the ocean changes slowly, and why the deep ocean barely changes in
one step.

### Salt is conserved

Evaporation and rain change the amount of water, not the amount of salt. Salt exchange with sea ice is small, so
total salt in the ocean is effectively constant. Mean salinity is total salt divided by volume, so freshwater input
lowers it only by adding volume. Our 19 levels have fixed volume, and the published data's total salt barely moves.
Global-mean salinity should not drift during a rollout.

<details markdown>
<summary>In Python</summary>

Practical salinity is close to grams of salt per kilogram of seawater, so salt mass is ρ₀ × S / 1000 × volume.

```python
volume = ds.areacello * ds.ocean_fraction * ds.dz           # (lev, y, x), m3

def total_salt(t):
    salinity = xr.concat([ds[f"so_{k}"].isel(time=t) for k in range(19)], dim="lev")
    return float((RHO_0 * salinity / 1000 * volume).sum())  # kg

start, end = total_salt(1000), total_salt(1000 + 73)        # 73 steps = one year
print(f"{start:.2e} kg, relative change {end / start - 1:.0e}")  # 5.31e+19 kg, relative change 3e-08
```

Over the same year, global OHC swings by about 5.6 × 10²² J with the seasons. Salt barely moves.

</details>

### Rotation dominates large-scale flow

On a rotating planet, large, slow flows are in **geostrophic balance**: the Coriolis force balances the pressure
gradient, and the surface pressure gradient is set by sea surface height. Water flows *along* contours of `zos`, not
down them:

    f u = −g ∂η/∂y,    f v = g ∂η/∂x,    f = 2Ω sin(latitude)

<details markdown>
<summary>In Python</summary>

η is `zos`. On the Gaussian grid the `y` and `x` coordinates are latitude and longitude in degrees, so the derivatives
are converted to per-metre. A degree of longitude shrinks by cos(latitude).

```python
import numpy as np

G = 9.81            # gravity g, m s-2
OMEGA = 7.2921e-5   # Earth's rotation rate Ω, s-1
R = 6.371e6         # Earth's mean radius, m

eta = ds.zos.isel(time=1000)                                  # η: (y, x), m
lat = np.deg2rad(ds.y)                                        # 1-D latitude, rad
f = 2 * OMEGA * np.sin(lat)                                   # Coriolis parameter, s-1

m_per_deg = np.pi * R / 180                                   # metres per degree of latitude
deta_dy = eta.differentiate("y") / m_per_deg                  # ∂η/∂y
deta_dx = eta.differentiate("x") / (m_per_deg * np.cos(lat))  # ∂η/∂x

u_g = (-G / f * deta_dy).where(abs(ds.y) > 5)                 # drop the equatorial band, where f → 0
v_g = (G / f * deta_dx).where(abs(ds.y) > 5)
```

Outside ±5°, `u_g` correlates about 0.8 with the model's surface current `uo_0`. The rest is wind-driven (Ekman)
flow and other ageostrophic motion. `xarray.differentiate` doesn't wrap at longitude 0°/360°.
`samudra.metrics.kernels.geostrophic_velocity_from_zos` handles that edge properly.

</details>

At mid-latitudes f ≈ 10⁻⁴ s⁻¹, so a 1 m change in `zos` over 100 km means about 1 m s⁻¹ of flow. That's the Gulf
Stream. Near the equator f → 0 and the balance breaks down. `samudra.metrics` derives surface currents from `zos` this
way to compare against satellite altimetry (DUACS), and it masks ±5° of latitude.

### Wind pushes water sideways

The wind's frictional push, combined with rotation, moves the wind-driven surface layer at right angles to the wind
when integrated over the layer's depth: to the right in the Northern Hemisphere and to the left in the Southern. This
is **Ekman transport**, τ / (ρ₀ f) in magnitude (m² s⁻¹, volume per unit width). The surface current itself turns
less, about 45° in the classical theory, and the angle grows with depth (the Ekman spiral). So don't expect `uo_0` and
`vo_0` to sit at 90° to the wind. Ekman transport is why winds blowing along the equator or along coastlines produce
upwelling of cold water.

<details markdown>
<summary>In Python</summary>

The northward Ekman transport is −τ_x / (ρ₀ f). The sign of f flips across the equator, which turns "to the right"
into "to the left". Average the wind over a year first: a single 5-day snapshot can catch a wind belt during a weak
or reversed spell.

```python
tau_x = ds.tauuo.isel(time=slice(1000, 1073)).mean("time")      # annual-mean eastward stress, N m-2
f = 2 * OMEGA * np.sin(np.deg2rad(ds.y))
northward = (-tau_x / (RHO_0 * f)).where(abs(ds.y) > 5)         # m2 s-1

belts = {"NH trades": slice(10, 25), "NH westerlies": slice(35, 55),
         "SH trades": slice(-25, -10), "SH westerlies": slice(-60, -40)}
for name, band in belts.items():
    print(name, float(tau_x.sel(y=band).mean()), float(northward.sel(y=band).mean()))
```

| Wind belt | Mean τ_x (N m⁻²) | Mean northward transport (m² s⁻¹) | Direction |
| --- | --- | --- | --- |
| NH trades (10–25°N), blowing west | −0.042 | +1.00 | poleward |
| NH westerlies (35–55°N), blowing east | +0.036 | −0.33 | equatorward |
| SH trades (10–25°S), blowing west | −0.043 | −1.03 | poleward |
| SH westerlies (40–60°S), blowing east | +0.098 | +0.86 | equatorward |

Trades push surface water poleward and westerlies push it toward the equator. Between them, around 30°, surface
water converges into the subtropical gyres.

</details>

### Dense water sits below light water

Density rises with salinity and falls with temperature. Away from regions of active convection, the water column is
**stably stratified**, with density increasing with depth. Temperature can increase with depth only if salinity
compensates, as in parts of the polar oceans. Widespread density inversions in a model's output mean something is
wrong.

<details markdown>
<summary>In Python</summary>

Stability is measured by the squared buoyancy frequency N², which is positive where density increases with depth. We
compute it with [`gsw`](https://github.com/TEOS-10/GSW-Python) (`pip install gsw`), the official Python
implementation of the TEOS-10 seawater standard [[IOC 2010](#ref-teos10)]. TEOS-10 works in absolute salinity and
conservative temperature, so we convert first.

```python
import gsw

snap = ds.isel(time=1000)
theta = xr.concat([snap[f"thetao_{k}"] for k in range(19)], dim="lev").values  # potential temperature
sp = xr.concat([snap[f"so_{k}"] for k in range(19)], dim="lev").values         # practical salinity
shape = theta.shape                                                           # (lev, y, x)
depth = np.broadcast_to(ds.lev.values[:, None, None], shape)
lat = np.broadcast_to(ds.lat.values, shape)
lon = np.broadcast_to(ds.lon.values, shape)

p = gsw.p_from_z(-depth, lat)                       # pressure, dbar
sa = gsw.SA_from_SP(sp, p, lon, lat)                # absolute salinity, g/kg
ct = gsw.CT_from_pt(sa, theta)                      # conservative temperature, °C
n2, _ = gsw.Nsquared(sa, ct, p, lat, axis=0)        # (lev-1, y, x), s-2, between adjacent levels

for k in range(n2.shape[0]):
    wet = np.isfinite(n2[k])
    print(k, f"{np.mean(n2[k][wet] < 0):.3f}")      # share of unstable pairs at each interface
```

In the ocean interior, roughly 30–4000 m, fewer than 1% of adjacent pairs are unstable. Negative N² clusters at the
two ends of the column:

* **The top interface (45% of pairs).** The surface mixed layer is stirred until it's nearly neutral, so tiny
  negative values are normal.
* **The deepest interface (16%).** There |N²| is only about 10⁻⁸ s⁻², effectively neutral.

Judge a model's stratification in the interior. Instability concentrated there is the warning sign.

</details>

### Resolution decides which motions exist

The ocean's "weather" is **mesoscale eddies**, swirls with a size set by the Rossby deformation radius. That radius
is ~200 km near the equator but only 10–30 km at high latitudes [[Chelton 1998](#ref-chelton)]. A 1° grid (~100 km)
cannot represent most eddies. A 1/4° grid represents them in the subtropics but not near the poles. This is why
kinetic energy grows with resolution, and why "smooth" output from a coarse model isn't automatically a bug.

<details markdown>
<summary>In Python</summary>

Compare surface kinetic energy (KE) in the eddy-rich Gulf Stream across the three releases, at the same timestep.

```python
URL = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/{}/OM4.zarr"
gulf_stream = dict(y=slice(30, 45), x=slice(280, 310))

for name in ("om4_onedeg", "om4_halfdeg", "om4_quarterdeg"):
    r = xr.open_zarr(URL.format(name)).isel(time=1000).sel(**gulf_stream)
    ke = 0.5 * (r.uo_0**2 + r.vo_0**2)                     # m2 s-2
    area = (r.areacello * r.ocean_fraction.isel(lev=0)).where(ke.notnull())  # ocean area only
    mean_ke = float((ke * area).sum() / area.sum())
    print(name, f"{mean_ke:.4f}", f"{float(np.sqrt(2 * ke.max())):.2f} m/s")
```

| Release | Area-mean KE (m² s⁻²) | Peak surface speed (m s⁻¹) |
| --- | --- | --- |
| 1° | 0.0167 | 0.69 |
| 1/2° | 0.0210 | 0.90 |
| 1/4° | 0.0227 | 0.98 |

The 1/4° release keeps 36% more KE than 1°. All three come from the same 1/4° OM4 run, so the difference is purely
what each grid can hold. The same comparison globally gives a smaller gap (0.0167 vs 0.0194), because most of the
ocean is less energetic than the Gulf Stream.

</details>

## Invariants: checking your work with physics

These hold for any correct dataset, pipeline, or model. They make good **metamorphic tests**: transform an input in a
way whose effect on the output is known, and check for that effect. No ground-truth answers are needed. Violations of
the cross-resolution invariants have caught real data-engineering bugs in this project.

### Across resolutions

* **Linear integrals are conserved.** Conservative regridding preserves area integrals by construction, and so do
  the diffusion-based gcm-filters smoothers we use for spatial filtering [[Grooms 2021](#ref-grooms)]. So total area,
  ocean area, volume, heat, and salt, as well as area-mean `zos` and volume-mean temperature and salinity, must agree
  across 1°, 1/2°, and 1/4°. A mismatch beyond ~0.1% points to wrong weights, masks, thickness, units, or radius.
* **Quadratic quantities only decrease.** Averaging removes variance: by the law of total variance, a field's variance
  equals the variance of its block averages plus the mean variance within blocks. So variance, kinetic energy, EKE, and
  gradient magnitudes must not grow as data is coarsened. The same fact explains a rule in `samudra.metrics`: *reduce
  before you regrid*. Regridding first blends cells and biases any quadratic quantity low.
* **No new extremes.** A conservative regrid is a weighted average with non-negative weights. Each coarse value lies
  within the range of the fine values it covers, so global minima and maxima can only move inward.
* **Spectra agree at well-resolved scales.** At wavelengths much longer than the coarse grid spacing, power spectra
  should match. The coarse spectrum rolls off as it approaches its grid's limit. Filtering adds a gradual,
  scale-dependent attenuation, not a sharp cutoff [[Grooms 2021](#ref-grooms)]. So for filtered products, compare
  only where the filter's attenuation is negligible, or compare against the filter's expected response.

### Within one dataset

* **Masks nest with depth.** A cell that is ocean at level k+1 is ocean at level k. `ocean_fraction` lies in [0, 1]
  and never increases with depth. 3-D fields are `NaN` exactly where the masks say land, at every time step.
* **Physical bounds.** Temperature stays at or above the freezing point (about −1.9 °C at 35 psu, falling about
  0.054 °C per psu [[IOC 2010](#ref-teos10)]). Salinity is non-negative. Area-mean `zos` is ≈ 0 at every step, because
  CMIP defines it as dynamic sea level with the global mean removed [[Griffies 2016](#ref-griffies);
  [Gregory 2019](#ref-gregory)].
* **The heat budget closes, and salt is conserved** (see the physics above).
* **Time is regular.** Steps are 5 days apart, except a 6-day step across each leap-year boundary, where each year's
  73 steps re-anchor to January 3. Any other spacing is a bug.

### Model, loader, and rollout

* **Longitude-roll equivariance.** The globe is periodic in longitude, and the models pad circularly. Rolling every
  input (state, forcing, masks) by n columns should roll the output by n. This holds exactly only when n is a multiple
  of the network's total downsampling factor and learned positional channels are off. Latitude has no such symmetry,
  because rotation (f) varies with latitude.
* **Land values don't matter.** Perturbing raw values under the land mask must not change any prediction. The loader
  replaces them with a fill value.
* **Batch independence.** In eval mode, a sample's prediction doesn't depend on batch size or its batch-mates.
* **Restart invariance.** Rolling out 2N steps equals rolling out N, then N more from that state with the same
  forcing.
* **Loader equivalence.** The Rust and Python loaders, and any number of workers, ranks, or chunkings, produce
  identical samples.
* **Round trips.** `denormalize(normalize(x)) == x`. Normalization statistics come only from the training period.
* **Forcing matters.** Shifting the forcing by one step, or replacing it with climatology, should measurably hurt
  skill. If it doesn't, the model is ignoring its forcing or the forcing is misaligned.

### Reference values

These were measured with Xarray from the published v2026-09 OM4 stores, at time index 1000 (Sept 1971), with
ρ₀ = 1035 kg m⁻³ and c_p = 3992 J kg⁻¹ K⁻¹. Unless noted, the 1°, 1/2°, and 1/4° stores agree to six significant figures. We
plan to publish these with each data release instead (#976).

| Quantity | v2026-09 OM4 | Real ocean | Note |
| --- | --- | --- | --- |
| Total `areacello` | 5.0767 × 10¹⁴ m² | 5.10 × 10¹⁴ m² (Earth) | 0.47% low: computed with the polar radius (#972) |
| Ocean surface area | 3.588 × 10¹⁴ m² | 3.61 × 10¹⁴ m² | same 0.47% |
| Ocean volume, Σ area × `ocean_fraction` × `dz` | 1.476 × 10¹⁸ m³ | 1.33 × 10¹⁸ m³ [[Charette 2010](#ref-charette)] | ~10% high: bottom cells count their full `dz` (#973) |
| Volume-mean temperature, salinity | 3.334 °C, 34.722 psu | ~3.5 °C, 34.7 psu | |
| Global OHC, relative to 0 °C | 2.0334 × 10²⁵ J | | inherits the volume bias |
| Area-mean `zos` | −0.0001 m | 0 by definition | |
| Area-mean surface kinetic energy | 0.0167 / 0.0186 / 0.0194 m² s⁻² | | rises with resolution (1° / 1/2° / 1/4°) |
| SST range | −1.91…32.52 / −1.98…32.71 / −2.01…32.75 °C | | widens with resolution |
| Minimum salinity | 0.03 psu | | river estuaries: Ob Gulf, Río de la Plata, Amazon (#974) |
| Heat budget, 1°, one year | per-step correlation 0.994, slope 0.98 | | annual residual ≈ 0.06 W m⁻² |
| Total salt drift, 1°, one year | 3 × 10⁻⁸ (relative) | 0 | |

## What makes a good emulator

`samudra.metrics` scores finished multi-year rollouts against **observations**, not only against OM4:

* Satellite altimetry (DUACS) for surface currents and eddy kinetic energy (EKE).
* Satellite and in-situ sea surface temperature (OISST).
* Argo-float-based ocean heat content by depth layer (ARGO-IAP).

Errors are area-weighted, and uncertainty comes from resampling whole calendar years. Two diagnostics target
failure modes RMSE misses:

* **Residual variance** (after removing trend and seasonal cycle) catches a rollout that drifts toward a smooth
  average, a common emulator failure called *variance collapse*.
* **Spectra** show *at which scales* a model is wrong.

For short-term forecasts, always compare against simple baselines. **Persistence** predicts that nothing changes;
**climatology** predicts the average for that time of year. The deep ocean barely changes in 5 days, so persistence
already has a tiny RMSE there. Report skill relative to these baselines [[Murphy 1988](#ref-murphy)], or score the
predicted *change* rather than the state.

## References

* <a id="ref-adcroft"></a>Adcroft, A., et al. (2019). The GFDL Global Ocean and Sea Ice Model OM4.0: Model
  description and simulation features. *Journal of Advances in Modeling Earth Systems*, 11, 3167–3211.
  <https://doi.org/10.1029/2019MS001726>
* <a id="ref-charette"></a>Charette, M. A., & Smith, W. H. F. (2010). The volume of Earth's ocean. *Oceanography*,
  23(2), 112–114. <https://doi.org/10.5670/oceanog.2010.51>
* <a id="ref-chelton"></a>Chelton, D. B., et al. (1998). Geographical variability of the first baroclinic Rossby radius
  of deformation. *Journal of Physical Oceanography*, 28, 433–460.
  <https://doi.org/10.1175/1520-0485(1998)028%3C0433:GVOTFB%3E2.0.CO;2>
* <a id="ref-samudra"></a>Dheeshjith, S., et al. (2025). Samudra: An AI global ocean emulator for climate.
  *Geophysical Research Letters*. <https://doi.org/10.1029/2024GL114318>
* <a id="ref-samudra2"></a>Samudra 2: Scaling ocean emulators across resolutions (2026). <https://arxiv.org/abs/2606.02610>
* <a id="ref-samudrace"></a>Duncan, J., et al. (2026). SamudrACE: Fast and accurate coupled climate modeling with 3D
  ocean and atmosphere emulators. *Geophysical Research Letters*. <https://doi.org/10.1029/2025GL119340>
* <a id="ref-gregory"></a>Gregory, J. M., et al. (2019). Concepts and terminology for sea level: Mean, variability and
  change, both local and global. *Surveys in Geophysics*, 40, 1251–1289. <https://doi.org/10.1007/s10712-019-09525-z>
* <a id="ref-griffies"></a>Griffies, S. M., et al. (2016). OMIP contribution to CMIP6: Experimental and diagnostic
  protocol for the physical component of the Ocean Model Intercomparison Project. *Geoscientific Model Development*,
  9, 3231–3296. <https://doi.org/10.5194/gmd-9-3231-2016>
* <a id="ref-grooms"></a>Grooms, I., et al. (2021). Diffusion-based smoothers for spatial filtering of gridded
  geophysical data. *Journal of Advances in Modeling Earth Systems*, 13. <https://doi.org/10.1029/2021MS002552>
* <a id="ref-teos10"></a>IOC, SCOR & IAPSO (2010). *The international thermodynamic equation of seawater – 2010
  (TEOS-10)*. <https://www.teos-10.org>
* <a id="ref-murphy"></a>Murphy, A. H. (1988). Skill scores based on the mean square error and their relationships to
  the correlation coefficient. *Monthly Weather Review*, 116, 2417–2424.
  <https://doi.org/10.1175/1520-0493(1988)116%3C2417:SSBOTM%3E2.0.CO;2>
* <a id="ref-talley"></a>Talley, L. D., Pickard, G. L., Emery, W. J., & Swift, J. H. (2011). *Descriptive Physical
  Oceanography: An Introduction* (6th ed.). Academic Press.
* <a id="ref-vallis"></a>Vallis, G. K. (2017). *Atmospheric and Oceanic Fluid Dynamics* (2nd ed.). Cambridge University
  Press.

Further datasets mentioned in this project: LLC4320, an MITgcm simulation at 1/48° covering Sept 2011 – Nov 2012
[[GMD 2023](https://gmd.copernicus.org/articles/16/7143/2023/)].
