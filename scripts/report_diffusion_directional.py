# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Grid-angle diagnostics of matched day-30 surface fields; not a scored metric.

Power is integrated in grid-wavelength bands, after the scorer's planar detrend
and periodic Hann with mean-square correction. The full Fourier plane avoids
half-plane multiplicity ambiguity. Coordinates are grid cells, not metres:
cos(4 angle) distinguishes axis (+) from diagonal (-) orientation in the image,
not physical ocean anisotropy. Angle describes the wavevector; stripes are
perpendicular, which does not change cos(4 angle). Rectangular Fourier lattices and the taper bias
this statistic; matched reference differences help but cannot identify artifacts
uniquely. All case/member raw powers are pooled before normalizing moments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from samudra.metrics import spectra

BANDS = ((2, 4), (4, 8), (8, 16))


def band_moments(fields: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return band power and its cos(4 angle) moment per leading array index.

    Bands have low-frequency inclusive/high-frequency exclusive boundaries,
    except the shortest band includes the radial Nyquist at 1/2 cycle/cell.
    Modes outside that Nyquist circle are excluded. No pixel filling is allowed.
    Power has squared field units and sums (not averages) Fourier modes.
    """
    array = np.asarray(fields, dtype=np.float64)
    if array.ndim < 2 or min(array.shape[-2:]) < 8 or not np.isfinite(array).all():
        raise ValueError("Expected finite, fully observed boxes of at least 8 x 8")
    residual = spectra.remove_plane(array)
    # An exactly planar input leaves only least-squares floating-point residue.
    scale = np.max(np.abs(array), axis=(-2, -1), keepdims=True)
    zero = np.max(np.abs(residual), axis=(-2, -1), keepdims=True) <= (
        64 * np.finfo(np.float64).eps * scale
    )
    residual = np.where(zero, 0.0, residual)
    height, width = array.shape[-2:]
    window = np.outer(spectra._periodic_hann(height), spectra._periodic_hann(width))
    power = np.abs(np.fft.fft2(residual * window, norm="forward")) ** 2
    power = power / np.mean(window**2)
    ky, kx = np.meshgrid(np.fft.fftfreq(height), np.fft.fftfreq(width), indexing="ij")
    radius = np.sqrt(kx**2 + ky**2)
    angular = np.cos(4 * np.arctan2(ky, kx))
    powers, moments = [], []
    for low, high in BANDS:
        select = (radius >= 1 / high) & (
            (radius <= 1 / low) if low == 2 else (radius < 1 / low)
        )
        if not select.any():
            raise ValueError(f"Box has no Fourier modes in the {low}-{high} cell band")
        powers.append(power[..., select].sum(-1))
        moments.append((power[..., select] * angular[select]).sum(-1))
    return np.stack(powers, axis=-1), np.stack(moments, axis=-1)


def normalized_moment(power: np.ndarray, moment: np.ndarray) -> np.ndarray:
    """Undefined for zero power; never interpret a missing direction as isotropy."""
    return np.divide(moment, power, out=np.full_like(moment, np.nan), where=power > 0)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root: Path, arms: list[str]) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    evidence: dict = {}
    common: dict = {}
    common_origins = None
    accum: dict = {}
    for arm in arms:
        directory = root / arm / "validation"
        complete_path = directory / "COMPLETE.json"
        done = json.loads(complete_path.read_text())
        origins = done["metrics"]["origins"]
        if len(origins) != 9 or len(set(origins)) != 9:
            raise ValueError("Require the nine distinct matched validation origins")
        if common_origins is None:
            common_origins = origins
        elif origins != common_origins:
            raise ValueError("Validation origins differ across arms")
        evidence[arm] = {
            "complete_sha256": sha(complete_path),
            "checkpoint_sha256": done["checkpoint_sha256"],
            "step": done["step"],
            "counts": done["counts"],
            "sampling_steps": done["sampling_steps"],
            "files": {},
        }
        if done["sampling_steps"] != 32:
            raise ValueError("Expected matching 32-step diffusion evaluations")
        for origin in origins:
            filename = f"members-{origin}.npz"
            path = directory / filename
            checksum = sha(path)
            if checksum != done["files"][filename]:
                raise ValueError(f"Checksum mismatch: {path}")
            evidence[arm]["files"][filename] = checksum
            with np.load(path) as saved:
                lat, lon = saved["lat"], saved["lon"]
                mask = saved["mask"][:2].astype(bool)
                reference = saved["day30_surface_reference"]
                members = saved["day30"][:, :2]
                if members.shape != (8, 2, len(lat), len(lon)):
                    raise ValueError(
                        "Expected eight SST/SSH members on the reference grid"
                    )
                shared = (lat, lon, mask, reference)
                if origin not in common:
                    common[origin] = tuple(x.copy() for x in shared)
                else:
                    for actual, expected in zip(shared, common[origin], strict=True):
                        np.testing.assert_array_equal(actual, expected)
                for region, longitude, latitude in spectra.SPATIAL_REGIONS:
                    yi = np.flatnonzero(
                        (lat >= latitude.start) & (lat <= latitude.stop)
                    )
                    xi = np.flatnonzero(
                        (lon >= longitude.start) & (lon <= longitude.stop)
                    )
                    if not mask[:, yi][:, :, xi].all():
                        raise ValueError(
                            f"Incomplete wet support: {arm}/{origin}/{region}"
                        )
                    ref_box = reference[:, yi][:, :, xi]
                    member_box = members[:, :, yi][:, :, :, xi]
                    if (
                        not np.isfinite(ref_box).all()
                        or not np.isfinite(member_box).all()
                    ):
                        raise ValueError(
                            f"Missing regional values: {arm}/{origin}/{region}"
                        )
                    for mode, values in (
                        ("reference", ref_box),
                        ("ensemble_mean", member_box.mean(0)),
                        ("member_power_mean", member_box),
                    ):
                        power, moment = band_moments(values)
                        if mode == "member_power_mean":
                            power, moment = power.mean(0), moment.mean(0)
                        key = (arm, mode, region)
                        previous = accum.get(
                            key, (np.zeros_like(power), np.zeros_like(moment))
                        )
                        accum[key] = (previous[0] + power, previous[1] + moment)
    for (arm, mode, region), (power_sum, moment_sum) in accum.items():
        power = power_sum / 9
        angular = normalized_moment(power_sum, moment_sum)
        ref_power_sum, ref_moment_sum = accum[arm, "reference", region]
        ref_power = ref_power_sum / 9
        ref_angular = normalized_moment(ref_power_sum, ref_moment_sum)
        for channel, field in enumerate(("SST", "SSH")):
            for index, (low, high) in enumerate(BANDS):
                p, rp = float(power[channel, index]), float(ref_power[channel, index])
                a, ra = (
                    float(angular[channel, index]),
                    float(ref_angular[channel, index]),
                )
                rows.append(
                    dict(
                        arm=arm,
                        mode=mode,
                        region=region,
                        field=field,
                        wavelength_min_cells=low,
                        wavelength_max_cells=high,
                        cases=9,
                        members=8 if mode == "member_power_mean" else 1,
                        band_power=p,
                        reference_band_power=rp,
                        power_ratio=p / rp if rp > 0 else None,
                        cos4=a,
                        reference_cos4=ra,
                        cos4_difference=a - ra,
                        cos4_defined=p > 0,
                        reference_cos4_defined=rp > 0,
                    )
                )
    return rows, dict(origins=common_origins, inputs=evidence)


def plot(rows: list[dict], arms: list[str], output: Path) -> None:
    regions = [name for name, _, _ in spectra.SPATIAL_REGIONS]
    figure, axes = plt.subplots(6, 2, figsize=(12, 16), squeeze=False)
    colors = [plt.get_cmap("tab10")(i) for i in range(10)]
    for region_index, region in enumerate(regions):
        for field_index, field in enumerate(("SST", "SSH")):
            ax_power, ax_angle = axes[region_index * 2 + field_index]
            subset = [
                row for row in rows if row["region"] == region and row["field"] == field
            ]
            ax_power.axhline(1, color="black", linewidth=1, label="Reference")
            ax_angle.axhline(0, color="black", linewidth=1, label="Reference")
            for i, arm in enumerate(arms):
                for mode, marker, offset, suffix in (
                    ("ensemble_mean", "o", -0.06, "ensemble mean"),
                    ("member_power_mean", "x", 0.06, "member powers"),
                ):
                    selected = [
                        row
                        for row in subset
                        if row["arm"] == arm and row["mode"] == mode
                    ]
                    x = np.arange(3) + offset
                    style = dict(
                        color=colors[i % len(colors)],
                        marker=marker,
                        markersize=5,
                        linewidth=0.9,
                        label=f"{arm}: {suffix}",
                    )
                    ax_power.plot(x, [row["power_ratio"] for row in selected], **style)
                    ax_angle.plot(
                        x, [row["cos4_difference"] for row in selected], **style
                    )
            ax_power.set_yscale("log")
            ax_power.set_ylabel(f"{region} · {field}\nBand power / reference")
            ax_angle.set_ylabel("Fourfold coefficient\nminus reference")
            for axis in (ax_power, ax_angle):
                axis.set_xticks(range(3), ["2–4", "4–8", "8–16"])
                axis.set_xlabel("Grid wavelength (cells)")
                axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle(
        "Day-30 grid-angle diagnostic · nine matched validation months\n"
        "Positive difference: more axial; negative: more diagonal than reference\n"
        "Raw powers pooled before normalization; not the blended metric or an artifact detector",
        fontsize=11,
    )
    legend_height = 0.025 * int(np.ceil(len(labels) / 3)) + 0.02
    figure.tight_layout(rect=(0, legend_height, 1, 0.96))
    figure.savefig(output / "directional.png", dpi=140)
    figure.savefig(output / "directional.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True, help="Campaign evaluation directory"
    )
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("Arms must be unique")
    rows, provenance = collect(args.root, args.arms)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "directional.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance["interpretation"] = __doc__
    provenance["zero_power"] = (
        "cos4=NaN and cos4_defined=false: direction unavailable at zero power"
    )
    (args.output / "directional-provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    plot(rows, args.arms, args.output)


if __name__ == "__main__":
    main()
