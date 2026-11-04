# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""CPU member-structure diagnostics for the shared-input initializer capacity screen.

Uses saved U/V members at levels 0, 9 and 18 only. Spectra use computational
cycles per grid cell; derivatives use spherical distances and wet five-point
stencils. This script neither runs a model nor interprets power as realism.
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from diagnose_stochastic_placement import (  # type: ignore[import-not-found]
    REGIONS,
    regional_spectrum,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    "B-learned-control",
    "B-shared",
    "B-shared-deep",
    "B-shared-wide",
    "B-shared-deep-wide",
]
LABELS = [
    "Learned control",
    "Shared",
    "Shared deep",
    "Shared wide",
    "Shared deep + wide",
]
EARTH_RADIUS = 6_371_000.0


def avg(x, weight):
    return float((np.where(weight > 0, x, 0) * weight).sum() / weight.sum())


def corr(x, y, weight):
    x, y = x - avg(x, weight), y - avg(y, weight)
    den = np.sqrt(avg(x * x, weight) * avg(y * y, weight))
    return avg(x * y, weight) / den if den > 1e-30 else np.nan


def derivatives(u, v, lat, lon, mask):
    """Relative vorticity/divergence in s^-1 on wet centered stencils.

    zeta=(dv/dlambda-d(u*cos(phi))/dphi)/(R*cos(phi));
    div=(du/dlambda+d(v*cos(phi))/dphi)/(R*cos(phi)).
    Latitude is nonperiodic; longitude is a uniformly spaced global cycle.
    Exclude |latitude|>=75 degrees, domain edges and any dry stencil neighbor.
    """
    phi, lam = np.deg2rad(lat), np.unwrap(np.deg2rad(lon))
    dl = np.diff(lam)
    np.testing.assert_allclose(dl, dl[0], rtol=1e-5, atol=1e-8)
    if not np.isclose(dl[0] * len(lon), 2 * np.pi, rtol=1e-5):
        raise ValueError("Need a full periodic longitude grid")
    c = np.cos(phi)[:, None]
    u, v = np.where(mask, u, 0), np.where(mask, v, 0)
    du = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dl[0])
    dv = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2 * dl[0])
    zeta = (dv - np.gradient(u * c, phi, axis=0, edge_order=2)) / (EARTH_RADIUS * c)
    divergence = (du + np.gradient(v * c, phi, axis=0, edge_order=2)) / (
        EARTH_RADIUS * c
    )
    support = mask.copy()
    for axis in (0, 1):
        support &= np.roll(mask, -1, axis=axis) & np.roll(mask, 1, axis=axis)
    support[[0, -1]] = False
    support &= (np.abs(lat) < 75)[:, None]
    return zeta, divergence, support


def self_test():
    lat = np.linspace(-80, 80, 161)
    lon = np.arange(360, dtype=float)
    mask = np.ones((161, 360), dtype=bool)
    omega = 1e-7
    u = np.broadcast_to(
        omega * EARTH_RADIUS * np.cos(np.deg2rad(lat))[:, None], mask.shape
    )
    zeta, div, wet = derivatives(u, np.zeros_like(u), lat, lon, mask)
    truth = np.broadcast_to(2 * omega * np.sin(np.deg2rad(lat))[:, None], mask.shape)
    np.testing.assert_allclose(zeta[wet], truth[wet], atol=5e-11, rtol=0)
    np.testing.assert_allclose(div[wet], 0, atol=1e-20)
    # A masked giant land value cannot contaminate any scored derivative.
    mask[80, 100] = False
    v = np.zeros_like(u)
    corrupt = u.copy()
    corrupt[80, 100] = 1e20
    got = derivatives(corrupt, v, lat, lon, mask)
    clean = derivatives(u, v, lat, lon, mask)
    np.testing.assert_array_equal(got[0][got[2]], clean[0][clean[2]])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    self_test()
    a.output.mkdir(parents=True, exist_ok=True)
    rows, spectra, vertical, sources = [], [], [], []
    fixed_noise_cases: dict[
        tuple[str, str], dict[str, tuple[np.ndarray, np.ndarray]]
    ] = {}
    common_grid = None
    gold_by_month: dict[str, np.ndarray] = {}
    for arm in ARMS:
        for weights in ("ema", "raw"):
            root = a.production / arm / f"initializer-{weights}"
            with np.load(root / "grid.npz") as g:
                lat, lon = g["lat"], g["lon"]
                names = list(g["names"][g["selection"]])
                mask = g["mask"][g["selection"]].astype(bool)
                grid = (lat, lon, names, mask)
                if common_grid is None:
                    common_grid = grid
                else:
                    for x, y in zip(grid, common_grid, strict=True):
                        np.testing.assert_array_equal(x, y)
            area = np.cos(np.deg2rad(lat))[:, None] * np.ones((1, len(lon)))
            files = sorted(root.glob("*-02000-*-maps.npz"))
            if len(files) != 3:
                raise ValueError("Expected exactly three final initializer dates")
            for path in files:
                receipt = json.loads(
                    path.with_name(path.name.replace("-maps.npz", ".json")).read_text()
                )
                month = receipt["month"]
                if (
                    receipt["variant"] != arm
                    or receipt["step"] != 2000
                    or receipt["weights"] != weights
                ):
                    raise ValueError("Unexpected initializer receipt")
                with np.load(path) as z:
                    members, gold = z["members"].astype(float), z["gold"].astype(float)
                if members.shape != (8, 6, 180, 360):
                    raise ValueError("Unexpected saved member selection")
                if month in gold_by_month:
                    np.testing.assert_array_equal(gold, gold_by_month[month])
                gold_by_month[month] = gold
                fixed_noise_cases.setdefault((arm, weights), {})[month] = (
                    members.astype(np.float32),
                    gold.astype(np.float32),
                )
                sources.append(
                    dict(
                        path=str(path),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        checkpoint_sha256=receipt["checkpoint_sha256"],
                    )
                )
                fields = [(f"member-{i + 1}", x) for i, x in enumerate(members)] + [
                    ("mean8", members.mean(0)),
                    ("reference", gold),
                ]
                for sample, field in fields:
                    for depth in (0, 9, 18):
                        ui, vi = names.index(f"uo_{depth}"), names.index(f"vo_{depth}")
                        wet = mask[ui] & mask[vi]
                        u, v = field[ui], field[vi]
                        rz, rd, stencil = derivatives(u, v, lat, lon, wet)
                        gz, gd, _ = derivatives(gold[ui], gold[vi], lat, lon, wet)
                        for region, (south, north, west, east) in REGIONS.items():
                            yy = (lat >= south) & (lat <= north)
                            xx = (lon % 360 >= west) & (lon % 360 < east)
                            region_mask = yy[:, None] & xx[None, :]
                            w, sw = (
                                area * wet * region_mask,
                                area * stencil * region_mask,
                            )
                            if w.sum() == 0 or sw.sum() == 0:
                                continue
                            base = dict(
                                variant=arm,
                                weights=weights,
                                month=month,
                                sample=sample,
                                depth=depth,
                                region=region,
                            )
                            rows.append(
                                dict(
                                    **base,
                                    vector_power=avg(u * u + v * v, w),
                                    uv_correlation=corr(u, v, w),
                                    vorticity_power=avg(rz * rz, sw),
                                    divergence_power=avg(rd * rd, sw),
                                    vorticity_correlation_gold=corr(rz, gz, sw),
                                    divergence_correlation_gold=corr(rd, gd, sw),
                                    derivative_cells=int((sw > 0).sum()),
                                )
                            )
                            if region != "global":
                                crop = np.ix_(yy, xx)
                                edges, values, counts = regional_spectrum(
                                    u[crop], v[crop], wet[crop]
                                )
                                for lo, hi, value, n in zip(
                                    edges[:-1], edges[1:], values, counts, strict=True
                                ):
                                    spectra.append(
                                        dict(
                                            **base,
                                            k_low=lo,
                                            k_high=hi,
                                            power=value,
                                            modes=n,
                                        )
                                    )
                    for deep in (9, 18):
                        indices = [
                            names.index(n)
                            for n in ("uo_0", "vo_0", f"uo_{deep}", f"vo_{deep}")
                        ]
                        wet = mask[indices].all(0)
                        u0, v0, ud, vd = field[indices]
                        w = area * wet
                        denom = np.sqrt(
                            avg(u0 * u0 + v0 * v0, w) * avg(ud * ud + vd * vd, w)
                        )
                        vertical.append(
                            dict(
                                variant=arm,
                                weights=weights,
                                month=month,
                                sample=sample,
                                deep_level=deep,
                                vector_alignment=avg(u0 * ud + v0 * vd, w) / denom
                                if denom
                                else np.nan,
                                u_vertical_correlation=corr(u0, ud, w),
                                v_vertical_correlation=corr(v0, vd, w),
                            )
                        )
    sensitivity_rows = []
    for (arm, weights), cases in fixed_noise_cases.items():
        for first, second in combinations(sorted(cases), 2):
            xa, ya = cases[first]
            xb, yb = cases[second]
            pa, pb = xa - xa.mean(0), xb - xb.mean(0)
            for depth in (0, 9, 18):
                ids = [names.index(f"uo_{depth}"), names.index(f"vo_{depth}")]
                w = area * mask[ids].all(0)
                for member in range(8):
                    ua, va = pa[member, ids].astype(float)
                    ub, vb = pb[member, ids].astype(float)
                    power_a, power_b = (
                        avg(ua * ua + va * va, w),
                        avg(ub * ub + vb * vb, w),
                    )
                    diff = avg((ua - ub) ** 2 + (va - vb) ** 2, w)
                    ga, gb = ya[ids].astype(float), yb[ids].astype(float)
                    gold_diff = avg(((ga - gb) ** 2).sum(0), w)
                    gold_norm = avg((ga**2 + gb**2).sum(0), w)
                    sensitivity_rows.append(
                        dict(
                            variant=arm,
                            weights=weights,
                            depth=depth,
                            month_a=first,
                            month_b=second,
                            member=member + 1,
                            perturbation_vector_cosine=avg(ua * ub + va * vb, w)
                            / np.sqrt(power_a * power_b),
                            perturbation_normalized_change=np.sqrt(
                                diff / (power_a + power_b)
                            ),
                            perturbation_u_spatial_correlation=corr(ua, ub, w),
                            perturbation_v_spatial_correlation=corr(va, vb, w),
                            perturbation_rms_a=np.sqrt(power_a),
                            perturbation_rms_b=np.sqrt(power_b),
                            gold_normalized_change=np.sqrt(gold_diff / gold_norm),
                        )
                    )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(a.output / "fixed-noise-conditioning-by-date.csv", index=False)
    sensitivity.groupby(["variant", "weights", "depth"], as_index=False)[
        [
            "perturbation_vector_cosine",
            "perturbation_normalized_change",
            "perturbation_u_spatial_correlation",
            "perturbation_v_spatial_correlation",
            "gold_normalized_change",
        ]
    ].mean().to_csv(a.output / "fixed-noise-conditioning-summary.csv", index=False)
    frame, spec, vert = (
        pd.DataFrame(rows),
        pd.DataFrame(spectra),
        pd.DataFrame(vertical),
    )
    summary_rows = []
    for (arm, weights, depth, region), group in frame.groupby(
        ["variant", "weights", "depth", "region"]
    ):
        members = group[group["sample"].str.startswith("member-")]
        truth = group[group["sample"] == "reference"]
        mean = group[group["sample"] == "mean8"]
        row = dict(variant=arm, weights=weights, depth=depth, region=region)
        for metric in ("vector_power", "vorticity_power", "divergence_power"):
            row[metric + "_ratio"] = members[metric].mean() / truth[metric].mean()
            row[metric + "_mean8_ratio"] = mean[metric].mean() / truth[metric].mean()
        row["vorticity_correlation_gold"] = members.vorticity_correlation_gold.mean()
        row["divergence_correlation_gold"] = members.divergence_correlation_gold.mean()
        row["uv_correlation"] = members.uv_correlation.mean()
        row["gold_uv_correlation"] = truth.uv_correlation.mean()
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    bands = spec.assign(
        band=np.where(
            spec.k_low >= np.sqrt(0.5) * 3 / 16, "fine_approx_le_7.5cells", "coarse"
        )
    )
    bands = bands.groupby(
        ["variant", "weights", "month", "sample", "depth", "region", "band"],
        as_index=False,
    ).agg(power=("power", "sum"))
    band_summary_rows = []
    for key, g in bands.groupby(["variant", "weights", "depth", "region", "band"]):
        member = g[g["sample"].str.startswith("member-")].power.mean()
        reference = g[g["sample"] == "reference"].power.mean()
        band_summary_rows.append(
            dict(
                zip(
                    ["variant", "weights", "depth", "region", "band"], key, strict=True
                ),
                member_power=member,
                reference_power=reference,
                ratio=member / reference,
            )
        )
    band_summary = pd.DataFrame(band_summary_rows)
    for name, x in (
        ("structure-by-date", frame),
        ("structure-summary", summary),
        ("spectra", spec),
        ("spectral-bands", band_summary),
        ("vertical-by-date", vert),
    ):
        x.to_csv(a.output / f"{name}.csv", index=False)
    for weights in ("ema", "raw"):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        for ai, (arm, label, marker) in enumerate(
            zip(ARMS, LABELS, ["o", "s", "^", "D", "P"][: len(ARMS)], strict=True)
        ):
            for col, depth in enumerate((0, 9)):
                s = spec[
                    (spec.variant == arm)
                    & (spec.weights == weights)
                    & (spec.depth == depth)
                    & (spec.region == "north-pacific")
                ]
                m = (
                    s[s["sample"].str.startswith("member-")]
                    .groupby(["k_low", "k_high"])
                    .power.mean()
                    .reset_index()
                )
                axes[0, col].plot(
                    (m.k_low + m.k_high) / 2,
                    m.power,
                    marker=marker,
                    markersize=4,
                    label=label,
                )
                if ai == 0:
                    r = (
                        s[s["sample"] == "reference"]
                        .groupby(["k_low", "k_high"])
                        .power.mean()
                        .reset_index()
                    )
                    axes[0, col].plot(
                        (r.k_low + r.k_high) / 2,
                        r.power,
                        color="black",
                        marker="x",
                        label="OM4 reference",
                    )
            plot_row = summary[
                (summary.variant == arm)
                & (summary.weights == weights)
                & (summary.depth == 0)
                & (summary.region == "global")
            ].iloc[0]
            vals = [
                plot_row.vector_power_ratio,
                plot_row.vorticity_power_ratio,
                plot_row.divergence_power_ratio,
            ]
            axes[1, 0].bar(
                np.arange(3) + (ai - (len(ARMS) - 1) / 2) * (0.8 / len(ARMS)),
                vals,
                width=0.8 / len(ARMS),
                label=label,
            )
            v = vert[
                (vert.variant == arm)
                & (vert.weights == weights)
                & (vert.deep_level == 9)
            ]
            value = v[v["sample"].str.startswith("member-")].vector_alignment.mean()
            axes[1, 1].bar(ai, value)
            if ai == 0:
                axes[1, 1].axhline(
                    v[v["sample"] == "reference"].vector_alignment.mean(),
                    color="black",
                    label="OM4 reference",
                )
        for col, depth in enumerate((0, 9)):
            axes[0, col].set(
                yscale="log",
                xlabel="Cycles / computational grid cell",
                ylabel="Band vector power (m²/s²)",
                title=f"North Pacific, depth level {depth}; mean member spectra",
            )
            axes[0, col].grid(alpha=0.2)
        axes[0, 0].legend(fontsize=8)
        axes[1, 0].axhline(1, color="black")
        axes[1, 0].set(
            xticks=np.arange(3),
            xticklabels=["Velocity power", "Vorticity power", "Divergence power"],
            ylabel="Mean member power / OM4 power",
            title="Surface; derivatives use wet stencils, |latitude| < 75°",
        )
        axes[1, 1].set(
            xticks=np.arange(len(ARMS)),
            xticklabels=LABELS,
            ylabel="Normalized surface–550 m vector dot product",
            title="Vertical alignment; paired wet support",
        )
        axes[1, 1].tick_params(axis="x", labelrotation=20)
        axes[1, 1].legend()
        fig.suptitle(
            f"Final 2k initializer members, {weights.upper()}; three dates × eight members"
        )
        for ext in ("png", "pdf"):
            fig.savefig(a.output / f"member-structure-{weights}.{ext}", dpi=150)
        plt.close(fig)
    # A fixed draw under three different conditioning dates exposes whether
    # the stochastic perturbation itself follows the inputs.
    fig, axes = plt.subplots(
        len(ARMS), 3, figsize=(16, 2.75 * len(ARMS)), constrained_layout=True
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#cccccc")
    ui = names.index("uo_0")
    months = sorted(next(iter(fixed_noise_cases.values())))
    for plot_index, (arm, label) in enumerate(zip(ARMS, LABELS, strict=True)):
        panels = []
        for month in months:
            members, _ = fixed_noise_cases[(arm, "ema")][month]
            panels.append(members[0, ui] - members[:, ui].mean(0))
        limit = max(float(np.abs(x[mask[ui]]).max()) for x in panels)
        for col, (month, field) in enumerate(zip(months, panels, strict=True)):
            ax = axes[plot_index, col]
            im = ax.imshow(
                np.where(mask[ui], field, np.nan),
                origin="lower",
                extent=(0, 360, float(lat[0]), float(lat[-1])),
                cmap=cmap,
                vmin=-limit,
                vmax=limit,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(f"{label} · {month}", fontsize=10)
            if plot_index == len(ARMS) - 1:
                ax.set_xlabel("Longitude (°E)")
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.set_ylabel("Latitude (°N)")
        fig.colorbar(
            im, ax=axes[plot_index].tolist(), shrink=0.8, label="u perturbation (m/s)"
        )
    fig.suptitle(
        "Same random draw, different dates: surface u member 1 minus mean of 8\nEMA at 2k; each row has one shared color scale, scales differ between rows"
    )
    for ext in ("png", "pdf"):
        fig.savefig(a.output / f"fixed-noise-perturbations-ema.{ext}", dpi=150)
    plt.close(fig)
    (a.output / "COMPLETE.json").write_text(
        json.dumps(
            dict(
                complete=True,
                sources=sources,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                notes=[
                    "Final 2000-update initialization only; raw and EMA remain separate.",
                    "All 24 members receive equal weight within arm; ratios divide their mean power by mean reference power across three dates.",
                    "Relative vorticity and horizontal divergence use spherical metric, centered differences, five-point all-wet support and |latitude|<75degrees.",
                    "Spectral bands are masked Hann-window computational-grid frequencies; not DEX or physical isotropic wavenumbers. Fine band starts at0.13258cycles/cell (~7.54cells), aligned to existing histogram bins.",
                    "Vector alignment includes mean flow, while component vertical correlations subtract spatial means.",
                    "Producer resets generator seed904003 for each date; same member index uses identical random input within an architecture. Cross-date perturbation correlation/change measures sensitivity to all conditioning changes together (surface history, forcing, calendar). It does not establish correct jet placement or compare identical noise across different architectures.",
                    "No SSH or temperature maps are present: no velocity-SSH or heat-transport coherence claim is possible from this export.",
                    "Power or vertical alignment alone does not establish realism; exact-position correlation penalizes plausible displaced stochastic structures.",
                ],
            ),
            indent=2,
        )
        + "\n"
    )
    for path in a.output.iterdir():
        if path.is_file() and not path.name.endswith(".license"):
            Path(str(path) + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )


if __name__ == "__main__":
    main()
