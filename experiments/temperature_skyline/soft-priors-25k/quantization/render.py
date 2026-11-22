# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched CPU-only quantization decomposition of two North Pacific patches."""

import csv
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from samudra.skyline.channel_vocabulary import ChannelVocabulary

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / "report/source-fields.npz"
spec = importlib.util.spec_from_file_location(
    "spectral_helpers", BASE.parent / "pre-spike-noise/render.py"
)
assert spec and spec.loader
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
MODELS = [
    ("new-temperature", "T-only classifier"),
    ("new-all-fields", "All-fields classifier"),
]


def encode(x, centers):
    # Physical midpoint assignment; ties choose lower center.
    return np.searchsorted((centers[:-1] + centers[1:]) / 2, x)


def fft(x):
    h, w = x.shape
    window = np.outer(np.hanning(h + 1)[:-1], np.hanning(w + 1)[:-1])
    transformed = np.fft.fft2(
        helpers.remove_plane(x) * window, norm="forward"
    ) / np.sqrt(np.mean(window**2))
    yy, xx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    return transformed, np.hypot(xx, yy)


def rmse(x, wet, area):
    return float(np.sqrt(np.average(x[wet] ** 2, weights=area[wet])))


def native_axes(fig, x, y, size=256):
    width, height = fig.get_size_inches() * fig.dpi
    return fig.add_axes((x / width, y / height, size / width, size / height))


def raster(ax, x, wet, title, cmap, lo, hi):
    im = ax.imshow(
        np.ma.masked_where(~wet, x),
        origin="lower",
        interpolation="nearest",
        resample=False,
        cmap=cmap,
        vmin=lo,
        vmax=hi,
        aspect="auto",
    )
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save(fig, name):
    fig.savefig(HERE / name, dpi=100, facecolor="white")
    plt.close(fig)


def main():
    cfg = json.loads((BASE / "all-fields-both-25k.json").read_text())
    with gzip.open(BASE / "matched-oracle.json.gz", "rt") as f:
        receipt = json.load(f)
    norm = receipt["source"]
    vocab = ChannelVocabulary(cfg["vocabulary_bounds"], cfg["model"]["bins"])
    metrics, spectra = [], []
    payload: dict[str, np.ndarray] = {}
    spectfig, spectaxes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    tranfig, tranaxes = plt.subplots(2, 3, figsize=(15, 8), layout="constrained")
    with np.load(SOURCE, allow_pickle=False) as archive:
        for iz, (z, depth) in enumerate(((0, 2.5), (13, 1850))):
            prefix = f"north-pacific-z{z:02d}/thetao/"
            data = {
                key[len(prefix) :]: archive[key]
                for key in archive.files
                if key.startswith(prefix)
            }
            wet, area = data["wet"], data["area"].astype(float)
            assert wet.shape == (64, 64)
            gold = data["gold"].astype(float)
            centers = (
                vocab.centers[38 + z].numpy().astype(float) * norm["std"][38 + z]
                + norm["mean"][38 + z]
            )
            target_ids = encode(gold, centers)
            oracle = centers[target_ids]
            q = oracle - gold
            assert np.all((gold[wet] >= centers[0]) & (gold[wet] <= centers[-1]))
            regression = data["original-temperature-regression"].astype(float)
            rounded_regression = centers[encode(regression, centers)]
            reg_round = rounded_regression - regression
            width = (
                centers[np.minimum(target_ids + 1, len(centers) - 1)]
                - centers[np.maximum(target_ids - 1, 0)]
            ) / 2
            tile = helpers.wet_patch(wet)
            assert tile is not None
            y, x, side = tile
            sl = np.s_[y : y + side, x : x + side]
            fq, frequency = fft(q[sl])
            band = (frequency >= 0.25) & (frequency <= 0.5)
            qp = float((abs(fq) ** 2)[band].sum())
            frr, _ = fft(reg_round[sl])
            regp = float((abs(frr) ** 2)[band].sum())
            totalerrors = [(data[key].astype(float) - gold)[wet] for key, _ in MODELS]
            limit = max(float(np.max(np.abs(e))) for e in totalerrors)
            fieldlo, fieldhi = float(np.min(gold[wet])), float(np.max(gold[wet]))
            fig = plt.figure(figsize=(17.4, 8.3), dpi=100)
            fig.suptitle(
                f"North Pacific · {depth:g} m · target {str(data['target_date'])}\n"
                "8192-bin oracle versus final EMA classifiers · each cell is 4×4 pixels; no interpolation",
                fontsize=14,
                y=0.98,
            )
            detail = plt.figure(figsize=(13.4, 4.6), dpi=100)
            detail.suptitle(
                f"North Pacific · {depth:g} m · quantization detail and wrong-bin maps",
                fontsize=14,
                y=0.98,
            )
            image_axes = []
            for j, (values, title, color, low, high) in enumerate(
                [
                    (
                        q,
                        "Oracle error, MAGNIFIED scale (°C)",
                        "RdBu_r",
                        -float(np.max(abs(q[wet]))),
                        float(np.max(abs(q[wet]))),
                    ),
                    (
                        width,
                        "Local center spacing (°C)",
                        "viridis",
                        float(np.min(width[wet])),
                        float(np.max(width[wet])),
                    ),
                ]
            ):
                ax = native_axes(detail, 45 + j * 330, 120)
                im = raster(ax, values, wet, title, color, low, high)
                detail.colorbar(
                    im,
                    cax=detail.add_axes(
                        ((45 + j * 330) / 1340, 70 / 460, 256 / 1340, 12 / 460)
                    ),
                    orientation="horizontal",
                    ticks=np.linspace(low, high, 3),
                    format="%.3g",
                )
            for m, (key, label) in enumerate(MODELS):
                prediction = data[key].astype(float)
                pred_ids = encode(prediction, centers)
                # Cached physical float32 predictions can differ from float64 centers by rounding.
                rounding = float(
                    np.max(np.abs(prediction[wet] - centers[pred_ids][wet]))
                )
                assert rounding < 1e-5
                residual = prediction - oracle
                error = prediction - gold
                np.testing.assert_allclose(
                    error[wet], (q + residual)[wet], atol=1e-12, rtol=0
                )

                def mse(a):
                    return float(np.average(a[wet], weights=area[wet]))

                cross = 2 * mse(q * residual)
                np.testing.assert_allclose(
                    mse(error**2), mse(q**2) + mse(residual**2) + cross, atol=1e-12
                )
                fr, _ = fft(residual[sl])
                fe, _ = fft(error[sl])
                rp, ep = (
                    float((abs(fr) ** 2)[band].sum()),
                    float((abs(fe) ** 2)[band].sum()),
                )
                cp = float((2 * (fq * fr.conj()).real)[band].sum())
                np.testing.assert_allclose(ep, qp + rp + cp, rtol=1e-10, atol=1e-15)
                diff = pred_ids - target_ids
                row = dict(
                    depth_m=depth,
                    model=key,
                    wet_cells=int(wet.sum()),
                    oracle_rmse_C=rmse(q, wet, area),
                    selection_rmse_C=rmse(residual, wet, area),
                    total_rmse_C=rmse(error, wet, area),
                    spatial_cross_C2=cross,
                    oracle_mse_ratio=mse(q * q) / mse(error * error),
                    exact_bin_fraction=float(np.mean(diff[wet] == 0)),
                    within_one_bin_fraction=float(np.mean(abs(diff[wet]) <= 1)),
                    median_absolute_bin_error=float(np.median(abs(diff[wet]))),
                    p95_absolute_bin_error=float(np.percentile(abs(diff[wet]), 95)),
                    max_absolute_bin_error=int(np.max(abs(diff[wet]))),
                    bin_spacing_min_C=float(width[wet].min()),
                    bin_spacing_max_C=float(width[wet].max()),
                    spectral_tile_y=y,
                    spectral_tile_x=x,
                    spectral_tile_side=side,
                    band_oracle_C2=qp,
                    band_selection_C2=rp,
                    band_total_C2=ep,
                    band_cross_C2=cp,
                    band_oracle_ratio=qp / ep,
                    band_selection_ratio=rp / ep,
                    band_cross_ratio=cp / ep,
                    regression_rmse_C=rmse(regression - gold, wet, area),
                    rounded_regression_rmse_C=rmse(
                        rounded_regression - gold, wet, area
                    ),
                    regression_rounding_rmse_C=rmse(reg_round, wet, area),
                    regression_rounding_band_C2=regp,
                    prediction_center_roundoff_max_C=rounding,
                    outside_support=0,
                )
                metrics.append(row)
                images = [
                    (gold, "OM4 target", "viridis", fieldlo, fieldhi),
                    (oracle, "OM4 rounded to bins", "viridis", fieldlo, fieldhi),
                    (q, "Oracle quantization error", "RdBu_r", -limit, limit),
                    (residual, "Prediction − rounded OM4", "RdBu_r", -limit, limit),
                    (error, "Prediction − OM4", "RdBu_r", -limit, limit),
                ]
                for j, (values, title, color, low, high) in enumerate(images):
                    ax = native_axes(fig, 100 + j * 325, 430 - m * 350)
                    im = raster(
                        ax,
                        values,
                        wet,
                        title
                        + (
                            "\nRMSE " + f"{rmse(values, wet, area):.5f} °C"
                            if j >= 2
                            else ""
                        ),
                        color,
                        low,
                        high,
                    )
                    if j == 0:
                        ax.set_ylabel(label, fontsize=11)
                    if m == 1:
                        cax = fig.add_axes(
                            ((100 + j * 325) / 1740, 45 / 830, 256 / 1740, 12 / 830)
                        )
                        fig.colorbar(im, cax=cax, orientation="horizontal", label="°C")
                    image_axes.append(ax)
                bins = np.digitize(diff, [-8.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 8.5])
                cmap = ListedColormap(
                    [
                        "#053061",
                        "#2166ac",
                        "#4393c3",
                        "#92c5de",
                        "#f7f7f7",
                        "#f4a582",
                        "#d6604d",
                        "#b2182b",
                        "#67001f",
                    ]
                )
                ax = native_axes(detail, 45 + (m + 2) * 330, 120)
                im = ax.imshow(
                    np.ma.masked_where(~wet, bins),
                    origin="lower",
                    interpolation="nearest",
                    resample=False,
                    cmap=cmap,
                    norm=BoundaryNorm(np.arange(10) - 0.5, 9),
                    aspect="auto",
                )
                ax.set_title(label + "\npredicted bin − true bin", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                cb = detail.colorbar(
                    im,
                    cax=detail.add_axes(
                        ((45 + (m + 2) * 330) / 1340, 70 / 460, 256 / 1340, 12 / 460)
                    ),
                    orientation="horizontal",
                    ticks=range(9),
                )
                cb.ax.set_xticklabels(
                    ["≤−9", "−8:−3", "−2", "−1", "0", "1", "2", "3:8", "≥9"], fontsize=7
                )
                edges = np.linspace(0, 0.5, side // 2 + 1)
                ax = spectaxes[iz, m]
                for name, power, color in [
                    ("Total error", abs(fe) ** 2, "black"),
                    ("Wrong-bin error", abs(fr) ** 2, "#d55e00"),
                    ("Oracle quantization", abs(fq) ** 2, "#0072b2"),
                ]:
                    vals = []
                    freqs = []
                    for i in range(len(edges) - 1):
                        select = (frequency > edges[i]) & (frequency <= edges[i + 1])
                        if not select.any():
                            continue
                        value = float(power[select].sum())
                        freq = float(frequency[select].mean())
                        vals.append(value)
                        freqs.append(freq)
                        spectra.append(
                            dict(
                                depth_m=depth,
                                model=key,
                                component=name,
                                cycles_per_cell=freq,
                                power_C2=value,
                            )
                        )
                    ax.loglog(freqs, vals, label=name, color=color)
                ax.axvspan(0.25, 0.5, color="gray", alpha=0.12)
                ax.set_title(
                    f"{depth:g} m · {label}\n2–4-cell power: Q {100 * qp / ep:.3f}% + R {100 * rp / ep:.3f}% + cross {100 * cp / ep:+.3f}%"
                )
                ax.set_xlabel("Cycles / native grid cell")
                ax.set_ylabel("Power per radial shell (°C²)")
                ax.legend(fontsize=8)
                ax.grid(alpha=0.2)
                for name, array_value in [
                    ("prediction", prediction),
                    ("selection_error", residual),
                    ("total_error", error),
                    ("bin_error", diff),
                ]:
                    payload[f"z{z:02d}/{key}/{name}"] = array_value
            fig.text(
                0.5,
                0.90,
                "All three error columns share one symmetric color scale at this depth. Oracle error is shown enlarged in the detail figure.",
                ha="center",
                fontsize=10,
            )
            save(fig, f"decomposition-z{z:02d}.png")
            save(detail, f"detail-z{z:02d}.png")
            # Predeclared central, fully wet 16-cell zonal transect; no error-based selection.
            candidates = [r for r in range(64) if wet[r, 24:40].all()]
            rowindex = min(candidates, key=lambda r: abs(r - 31.5))
            sel = np.s_[rowindex, 24:40]
            lon = data["lon"][24:40]
            for a in tranaxes[iz]:
                a.grid(alpha=0.2)
                a.set_xlabel("Longitude (°E)")
            for values, label, color in [
                (gold, "OM4", "#333333"),
                (oracle, "Rounded OM4", "#0072b2"),
                (data["new-temperature"], "T-only", "#009e73"),
                (data["new-all-fields"], "All-fields", "#d55e00"),
            ]:
                tranaxes[iz, 0].step(
                    lon, values[sel], where="mid", label=label, color=color, lw=1.4
                )
            tranaxes[iz, 0].set_title(f"{depth:g} m · {data['lat'][rowindex]:.3f}°N")
            tranaxes[iz, 0].set_ylabel("Temperature (°C)")
            tranaxes[iz, 0].legend(fontsize=8)
            for values, label, color in [
                (q, "Oracle quantization", "#0072b2"),
                (data["new-temperature"] - gold, "T-only error", "#009e73"),
                (data["new-all-fields"] - gold, "All-fields error", "#d55e00"),
            ]:
                tranaxes[iz, 1].step(
                    lon, values[sel], where="mid", label=label, color=color
                )
            tranaxes[iz, 1].set_title("Error at the same cells")
            tranaxes[iz, 1].set_ylabel("Error (°C)")
            tranaxes[iz, 1].legend(fontsize=8)
            tranaxes[iz, 2].step(
                lon, q[sel], where="mid", label="Rounded OM4 − OM4", color="#0072b2"
            )
            tranaxes[iz, 2].step(
                lon,
                reg_round[sel],
                where="mid",
                label="Rounded regression − regression",
                color="#cc79a7",
            )
            tranaxes[iz, 2].set_title("Quantization only · magnified scale")
            tranaxes[iz, 2].set_ylabel("Rounding change (°C)")
            tranaxes[iz, 2].legend(fontsize=8)
            for name, array_value in [
                ("gold", gold),
                ("oracle", oracle),
                ("oracle_error", q),
                ("wet", wet),
                ("area", area),
                ("centers", centers),
                ("regression", regression),
                ("rounded_regression", rounded_regression),
                ("lat", data["lat"]),
                ("lon", data["lon"]),
            ]:
                payload[f"z{z:02d}/{name}"] = array_value
            payload[f"z{z:02d}/transect_row"] = np.asarray(rowindex)
    spectfig.suptitle(
        "North Pacific error spectra · planar detrend + periodic Hann · identical fully wet tile\nQ = oracle error; R = prediction − rounded OM4; total = Q + R; signed cross-term included in band accounting"
    )
    save(spectfig, "error-spectra.png")
    save(tranfig, "transects.png")
    for name, rows in [("metrics.csv", metrics), ("spectra.csv", spectra)]:
        with (HERE / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    np.savez_compressed(HERE / "fields.npz", **payload)
    (HERE / "receipt.json.gz").write_bytes(
        gzip.compress(
            (
                json.dumps(
                    dict(
                        source=str(SOURCE.relative_to(BASE.parent)),
                        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                        config_sha256=hashlib.sha256(
                            (BASE / "all-fields-both-25k.json").read_bytes()
                        ).hexdigest(),
                        bins=8192,
                        models="Final 25k EMA, 18.5M / 17x17; regression rounding control is older 208M / 33x33 T-only model, not pending matched regression",
                        origin=4273,
                        target_date="2016-07-21",
                        spatial_metric="Area-weighted wet-cell RMSE; bin fractions unweighted wet-cell counts",
                        spectral_metric="Unweighted fully wet 64/32 tile, planar detrend, periodic Hann, radial frequency 0.25..0.5 cycles/cell, normalized by mean Hann squared",
                        checks="pointwise, area-MSE and band-power decompositions verified; zero outside-support targets; prediction rounding <1e-5 C",
                        metrics=metrics,
                    ),
                    indent=2,
                )
                + "\n"
            ).encode(),
            mtime=0,
        )
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
