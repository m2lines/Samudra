# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Regional native-grid spectra of saved, date-matched quarter/legacy forecasts.

No model execution. Values remain on each source's native grid. Shared support is
formed conservatively on the coarsest grid, then projected back geographically.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from global_physical_spectra import (  # type: ignore[import-not-found]
    REGIONS,
    geostrophic,
    spectrum,
)

VARIABLES = {
    "SST": (0, "°C² km"),
    "SSH": (1, "m² km"),
    "Geostrophic KE": (1, "m² s⁻² km"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def nearest_indices(lat, lon, reference_lat, reference_lon):
    """Nearest geographic reference cell, including cyclic longitude."""
    y = np.abs(np.asarray(lat)[:, None] - np.asarray(reference_lat)[None]).argmin(1)
    x = np.abs(
        (np.asarray(lon)[:, None] - np.asarray(reference_lon)[None] + 180) % 360 - 180
    ).argmin(1)
    return y, x


def common_support(valid_masks, grids, reference_index):
    """Require every mapped native cell valid, for every source.

    A native coastal hole excludes its entire reference cell. The coarsest
    support, rather than interpolated model values, is projected back to native
    grids. Cells unrepresented by any source are unavailable.
    """
    rlat, rlon = grids[reference_index]
    shared = np.ones((len(rlat), len(rlon)), dtype=bool)
    maps = []
    for valid, (lat, lon) in zip(valid_masks, grids, strict=True):
        y, x = nearest_indices(lat, lon, rlat, rlon)
        assert valid.shape == (len(lat), len(lon))
        index = (y[:, None] * len(rlon) + x[None]).ravel()
        allowed = np.ones(shared.size, dtype=bool)
        np.logical_and.at(allowed, index, valid.ravel())
        represented = np.bincount(index, minlength=shared.size) > 0
        shared &= (allowed & represented).reshape(shared.shape)
        maps.append((y, x))
    projected = [shared[y[:, None], x[None]] for y, x in maps]
    return shared, projected


def aggregate_spectra(fields, lat, lon, support):
    """Retain all origin powers; distinguish E[P] from P[E[field]].

    fields has shape [origin, component, y, x]. Two velocity components are
    averaged as 0.5*(P_u+P_v), which is KE spectral density, not PSD of speed².
    """
    fields = np.asarray(fields)
    if fields.ndim != 4 or fields.shape[1] not in (1, 2):
        raise ValueError(
            "Expected [origin, one scalar or two velocity components, y, x]"
        )
    if not np.isfinite(fields[..., support]).all():
        raise ValueError("Fields must be finite on the shared support")
    by_origin = []
    base = None
    for values in fields:
        parts = [spectrum(v, lat, lon, support) for v in values]
        if any(p is None for p in parts):
            return {
                "unavailable": "Insufficient common support (minimum 8 cells per side and 30% coverage)"
            }
        base = parts[0]
        by_origin.append(np.mean([p["power"] for p in parts], axis=0))
    means = [spectrum(v, lat, lon, support) for v in fields.mean(axis=0)]
    assert base is not None and all(p is not None for p in means)
    return {
        **{k: v for k, v in base.items() if k != "power"},
        "per_origin_power": np.asarray(by_origin).tolist(),
        "snapshot_power": np.mean(by_origin, axis=0).tolist(),
        "mean_field": np.mean([p["power"] for p in means], axis=0).tolist(),
        "origins": len(fields),
    }


def validate_source(source):
    surface, lat, lon, mask = (source[k] for k in ("surface", "lat", "lon", "mask"))
    if surface.shape != (3, 2, 2, len(lat), len(lon)) or mask.shape != (
        2,
        len(lat),
        len(lon),
    ):
        raise ValueError(
            "Expected three origins, two leads and SST/SSH on a native grid"
        )
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError("Coordinates must increase; longitude must be canonical 0–360")
    if lon.min() < 0 or lon.max() >= 360:
        raise ValueError("Longitude must be canonical 0–360")
    if source["origins"].tolist() != ["2015-01-01", "2018-01-01", "2021-01-01"]:
        raise ValueError("Unexpected origin cohort")
    if source["lead_days"].tolist() != [30, 365]:
        raise ValueError("Unexpected leads")


def validate_dates(sources):
    """Require actual five-day target bins, not only nominal lead labels."""
    first = next(iter(sources.values()))
    for key in ("bin_start", "bin_midpoint", "bin_end"):
        expected = np.asarray(first[key], dtype="datetime64[s]")
        if expected.shape != (3, 2) or np.isnat(expected).any():
            raise ValueError("Missing or invalid target bin dates")
        for source in sources.values():
            if not np.array_equal(
                np.asarray(source[key], dtype="datetime64[s]"), expected
            ):
                raise ValueError(f"Source target dates differ: {key}")
    start = np.asarray(first["bin_start"], dtype="datetime64[s]")
    expected_start = np.asarray(
        ["2015-01-01", "2018-01-01", "2021-01-01"], dtype="datetime64[D]"
    )[:, None] + np.asarray([25, 360], dtype="timedelta64[D]")
    if not np.array_equal(start, expected_start):
        raise ValueError(
            "Target dates do not correspond to the requested day30/day365 bins"
        )
    end = np.asarray(first["bin_end"], dtype="datetime64[s]")
    middle = np.asarray(first["bin_midpoint"], dtype="datetime64[s]")
    if not np.all(end - start == np.timedelta64(5, "D")) or not np.all(
        middle - start == np.timedelta64(60, "h")
    ):
        raise ValueError("Expected five-day means with centered timestamps")


def calculate(sources):
    """Calculate both aggregation orders with separate scalar/derivative support."""
    names = list(sources)
    grids = [(sources[n]["lat"], sources[n]["lon"]) for n in names]
    reference_index = int(np.argmin([len(y) * len(x) for y, x in grids]))
    output = {"reference_support_grid": names[reference_index], "leads": {}}
    prepared = {}
    for variable, (channel, _) in VARIABLES.items():
        fields, valid_masks = [], []
        for name in names:
            source = sources[name]
            z = np.where(
                source["mask"][channel], source["surface"][:, :, channel], np.nan
            )
            values = (
                np.stack(geostrophic(z, source["lat"], source["lon"]), axis=2)
                if variable == "Geostrophic KE"
                else z[:, :, None]
            )
            fields.append(values)
            valid_masks.append(np.isfinite(values).all(axis=(0, 1, 2)))
        shared, masks = common_support(valid_masks, grids, reference_index)
        prepared[variable] = (fields, valid_masks, masks)
        output.setdefault("support_sha256", {})[variable] = hashlib.sha256(
            shared.tobytes()
        ).hexdigest()
    for lead_index, lead in enumerate((30, 365)):
        output["leads"][str(lead)] = {}
        for variable, (_, unit) in VARIABLES.items():
            all_fields, valid_masks, masks = prepared[variable]
            fields = [values[:, lead_index] for values in all_fields]
            regions = {}
            for region, xb, yb in REGIONS:
                records = {}
                for n, (lat, lon), values, valid, support in zip(
                    names, grids, fields, valid_masks, masks, strict=True
                ):
                    y = np.flatnonzero((lat >= yb[0]) & (lat <= yb[1]))
                    x = np.flatnonzero((lon >= xb[0]) & (lon <= xb[1]))
                    block = np.ix_(y, x)
                    local_valid, local_support = valid[block], support[block]
                    available = int(local_valid.sum())
                    retained = int(local_support.sum())
                    record = aggregate_spectra(
                        values[:, :, y][:, :, :, x], lat[y], lon[x], local_support
                    )
                    fraction = retained / available if available else 0.0
                    records[n] = dict(
                        record,
                        source_valid_cells=available,
                        shared_valid_cells=retained,
                        retained_fraction_of_source_valid=fraction,
                        common_fraction_of_box=float(local_support.mean()),
                        support_warning="More than half the source-valid cells excluded"
                        if fraction < 0.5
                        else None,
                    )
                regions[region] = {
                    "longitude_bounds": xb,
                    "latitude_bounds": yb,
                    "unit": unit,
                    "curves": records,
                }
            output["leads"][str(lead)][variable] = regions
    return output


def plot(results, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    first_regions = next(iter(results["leads"]["30"].values()))
    names = list(next(iter(first_regions.values()))["curves"])
    palette = ["#2167ad", "#c87526", "#7b4caf", "#35a38f", "#222222"]
    limits = {}
    for aggregation in ("snapshot_power", "mean_field"):
        for variable in VARIABLES:
            for region, _, _ in REGIONS:
                curves = [
                    results["leads"][lead][variable][region]["curves"][name]
                    for lead in ("30", "365")
                    for name in names
                ]
                curves = [c for c in curves if aggregation in c]
                if not curves:
                    continue
                x = np.concatenate([c["k_rad_km"] for c in curves])
                y = np.concatenate([c[aggregation] for c in curves])
                y = y[np.isfinite(y) & (y > 0)]
                if len(y):
                    limits[aggregation, variable, region] = (
                        (x.min() * 0.85, x.max() * 1.15),
                        (y.min() * 0.7, y.max() * 1.4),
                    )
    for lead, variables in results["leads"].items():
        for aggregation, subtitle in [
            ("snapshot_power", "mean of three snapshot powers"),
            ("mean_field", "power of the three-snapshot mean field"),
        ]:
            for variable, regions in variables.items():
                fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
                for ax, (region, data) in zip(axes.flat, regions.items(), strict=True):
                    for name, color in zip(names, palette, strict=True):
                        curve = data["curves"][name]
                        if aggregation not in curve:
                            continue
                        ax.loglog(
                            curve["k_rad_km"],
                            curve[aggregation],
                            label=name,
                            color=color,
                            ls="--" if "OM4" in name or "Legacy" in name else "-",
                            lw=1.5,
                        )
                    if (aggregation, variable, region) in limits:
                        xlim, ylim = limits[aggregation, variable, region]
                        ax.set_xlim(xlim)
                        ax.set_ylim(ylim)
                    ax.set_title(region)
                    ax.set_xlabel("Angular wavenumber (rad/km)")
                    ax.set_ylabel("k × PSD (" + data["unit"] + ")")
                    ax.grid(alpha=0.2, which="both")
                    if variable == "Geostrophic KE" and region == "Niño 3.4":
                        ax.set_axis_off()
                        ax.text(
                            0.5,
                            0.5,
                            "Unavailable: geostrophy excludes ±5°",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=9,
                        )
                    elif any(c["support_warning"] for c in data["curves"].values()):
                        ax.text(
                            0.02,
                            0.02,
                            "Restricted wet support; see support fractions",
                            transform=ax.transAxes,
                            fontsize=7,
                        )

                    def inverse(value):
                        return np.divide(
                            2 * np.pi,
                            value,
                            out=np.full_like(np.asarray(value, dtype=float), np.inf),
                            where=np.asarray(value) != 0,
                        )

                    if (aggregation, variable, region) in limits:
                        top = ax.secondary_xaxis("top", functions=(inverse, inverse))
                        top.set_xlabel("Wavelength (km)", fontsize=8)
                        top.tick_params(labelsize=7)
                handles, labels = axes.flat[0].get_legend_handles_labels()
                fig.legend(
                    handles, labels, loc="outside lower center", ncol=3, fontsize=9
                )
                suffix = (
                    "; geostrophic velocity power, no temporal-mean removal"
                    if variable == "Geostrophic KE"
                    else ""
                )
                fig.suptitle(
                    f"{variable}, day {lead}: {subtitle}\n2015 / 2018 / 2021 January origins; native grids, shared wet support{suffix}",
                    fontsize=11,
                )
                name = f"day{lead}-{aggregation.replace('_', '-')}-{variable.lower().replace(' ', '-')}.png"
                fig.savefig(
                    output / name,
                    dpi=130,
                    pil_kwargs={"optimize": True, "compress_level": 9},
                )
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    sources, provenance = {}, {}
    for record in config["sources"]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            raise ValueError(f"Source hash differs: {path}")
        with np.load(path, allow_pickle=False) as bundle:
            data = {key: bundle[key] for key in bundle.files}
        for target, candidates in {
            "surface": ("fields",),
            "mask": ("wet",),
            "lead_days": ("leads_days", "leads"),
            "bin_start": ("target_bin_starts",),
            "bin_end": ("target_bin_ends",),
            "bin_midpoint": ("target_bin_midpoints",),
        }.items():
            if target not in data:
                data[target] = next(data[k] for k in candidates if k in data)
        validate_source(data)
        sources[record["label"]] = data
        provenance[record["label"]] = record
    if len(sources) != 5:
        raise ValueError("Exactly the five specified curves are required")
    validate_dates(sources)
    result = calculate(sources)
    result["provenance"] = provenance
    first = next(iter(sources.values()))
    result["target_bins"] = {
        key: np.asarray(first[key]).astype(str).tolist()
        for key in ("bin_start", "bin_midpoint", "bin_end")
    }
    result["method"] = dict(
        script_sha256=sha256(__file__),
        spectrum_sha256=sha256(Path(__file__).with_name("global_physical_spectra.py")),
        origins=["2015-01-01", "2018-01-01", "2021-01-01"],
        aggregation="Both mean snapshot power and power of mean field; no confidence intervals",
        support="Conservative all-native-cell intersection across both leads and all origins on coarsest grid, projected back; no value resampling",
        kinetic_energy="Half the sum of geostrophic u/v power; no temporal-mean subtraction; not temporal EKE",
        temporal_bins="Five-day means at day30 and365; see source bin bounds",
        schema="fields[origin,lead,channel,lat,lon]; channels SST Celsius, SSH meters; wet[channel,lat,lon]",
        input_units={
            "SST": "Celsius",
            "SSH": "meters",
            "latitude": "degrees north",
            "longitude": "degrees east",
        },
    )
    (args.output / "spectra.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    plot(result, args.output)


if __name__ == "__main__":
    main()
