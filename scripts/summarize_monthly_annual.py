#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify annual pilot artifacts and publish physical lead curves and error maps."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from samudra.experiments.monthly_annual import fixed_ohc_support
from samudra.experiments.observation_metrics import weighted_rmse
from samudra.experiments.observation_pilot import digest


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def bias_and_centered(prediction, reference, area, support):
    truth = np.where(support, reference, np.nan)
    valid = np.isfinite(truth)
    weights = np.broadcast_to(area, truth.shape) * valid
    error = np.where(valid, prediction - truth, 0)
    bias = float((error * weights).sum() / weights.sum())
    rmse = weighted_rmse(prediction, truth, area)
    return rmse, bias, float(np.sqrt(max(0, rmse**2 - bias**2)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    config = json.loads((a.root / "paths.json").read_text())
    audit = json.loads((a.root / "AUDITED.json").read_text())
    qualified = json.loads((a.root / "QUALIFIED.json").read_text())
    assert audit["config_sha256"] == digest(a.root / "paths.json")
    assert (
        qualified["first_month_readout_exact"]
        and qualified["fixed_ohc_scoring_on_reused_cases"]
    )
    assert qualified["audit_sha256"] == digest(a.root / "AUDITED.json")
    grid = dict(np.load(a.root / "grid.npz"))
    grid_source = next(k for k in audit["contracts"] if k.endswith("/grid.npz"))
    assert digest(a.root / "grid.npz") == audit["contracts"][grid_source]
    support = fixed_ohc_support(grid)
    area = np.cos(np.deg2rad(grid["lat"]))[:, None]
    normal = json.loads(a.normalization.read_text())["normalization"]
    keys = ["sst_rmse", "velocity_rmse", "eke_rmse"]
    models = {}
    lead_rows = []
    ohc_rows = []
    case_rows = []
    checksums = []
    archives = {}
    references = {}
    baseline = None
    output_count = 0
    for row in config["models"]:
        name = row["name"]
        root = a.root / name
        complete = json.loads((root / "COMPLETE.json").read_text())
        m = json.loads((root / "metrics.json").read_text())
        assert (
            complete["origins"] == config["origins"]
            and complete["producer"] == config["producer"] == m["producer"]
        )
        assert (
            complete["model"] == row
            and m["checkpoint_sha256"] == audit["contracts"][row["checkpoint"]]
        )
        assert (
            m["fixed_ohc_support_cells"]
            == support.sum((1, 2)).tolist()
            == [38933, 37190]
        )
        assert len(m["leads"]) == 73 and len(m["monthly_ohc"]) == 12
        checksums.append(
            f"{m.pop('checkpoint_sha256')}  checkpoint:{name}  # pragma: allowlist secret"
        )
        for filename, sha in complete["outputs"].items():
            assert digest(root / filename) == sha
            checksums.append(f"{sha}  {name}/{filename}  # pragma: allowlist secret")
            output_count += 1
        if baseline is None:
            baseline = m["climatology_leads"]
        else:
            assert m["climatology_leads"] == baseline
        m["checkpoint_checksum_record"] = "checksums.txt: checkpoint:" + name
        m["reused_origin_count"] = len(m.pop("reused_rollouts", {}))
        cases = {}
        latents = []
        surface = []
        truth = []
        for origin in config["origins"]:
            case = dict(np.load(root / (origin + ".npz")))
            assert (
                case["surface"].shape == (73, 2, 180, 360)
                and case["ohc"].shape == (12, 2, 180, 360)
                and case["latent_rms"].shape == (74,)
            )
            assert (
                np.isfinite(case["surface"]).all()
                and np.isfinite(case["latent_rms"]).all()
            )
            assert (
                not np.isinf(case["ohc"]).any()
                and np.isfinite(np.where(support, case["ohc"], 0)).all()
            )
            if origin not in references:
                references[origin] = (case["reference_surface"], case["reference_ohc"])
            else:
                np.testing.assert_array_equal(
                    case["reference_surface"], references[origin][0]
                )
                np.testing.assert_array_equal(
                    case["reference_ohc"], references[origin][1]
                )
            cs = []
            for i in range(73):
                rmse, bias, centered = bias_and_centered(
                    case["surface"][i, 0],
                    case["reference_surface"][i, 0],
                    area,
                    grid["mask"][0],
                )
                cs.append(
                    dict(
                        day=(i + 1) * 5,
                        sst_rmse=rmse,
                        sst_bias=bias,
                        sst_centered_rmse=centered,
                    )
                )
                case_rows.append(dict(model=name, origin=origin, **cs[-1]))
            cases[origin] = cs
            latents.append(case["latent_rms"])
            surface.append(case["surface"])
            truth.append(case["reference_surface"])
            if origin == "2021-01-01" and name in [
                "A-1729",
                "E-1729",
                "clip1-1730",
                "clip5-1730",
            ]:
                archives[name] = case
        surface_stack, truth_stack = np.stack(surface), np.stack(truth)
        for i, v in enumerate(m["leads"]):
            rmse, bias, centered = bias_and_centered(
                surface_stack[:, i, 0], truth_stack[:, i, 0], area, grid["mask"][0]
            )
            assert abs(rmse - v["sst_rmse"]) < 1e-10
            v.update(
                sst_bias=bias,
                sst_centered_rmse=centered,
                fixed_validation_surface_blend=float(
                    np.mean([v[k] / normal[k] for k in keys])
                ),
            )
            lead_rows.append(dict(model=name, **v))
        for v in m["monthly_ohc"]:
            ohc_rows.append(dict(model=name, **v))
        m["latent_rms_mean"] = np.mean(latents, 0).tolist()
        m["latent_rms_min"] = np.min(latents, 0).tolist()
        m["latent_rms_max"] = np.max(latents, 0).tolist()
        m["per_origin_sst"] = cases
        m["wandb_url"] = (root / "wandb-url.txt").read_text().strip()
        models[name] = m
    assert len(models) == 8 and output_count == 32
    names = list(models)
    colors = dict(zip(names, plt.get_cmap("tab10")(np.arange(8)), strict=True))
    assert baseline is not None
    baseline_rows = list(baseline)

    def draw_curves(axes, fields, scale=None):
        axes = list(axes)
        for name, m in models.items():
            for ax, key in zip(axes, fields, strict=True):
                ax.plot(
                    [v["day"] for v in m["leads"]],
                    [v[key] / (scale or 1) for v in m["leads"]],
                    color=colors[name],
                    ls="--" if "1730" in name else "-",
                    label=name,
                    lw=1.4,
                )
        for ax, key in zip(axes, fields, strict=True):
            if key in baseline_rows[0]:
                ax.plot(
                    [v["day"] for v in baseline_rows],
                    [v[key] / (scale or 1) for v in baseline_rows],
                    "k:",
                    lw=1.7,
                    label="Climatology",
                )
            ax.axvline(30, color="grey", ls=":", lw=0.8)
            ax.set_xlabel("Forecast lead (days)")
            ax.grid(alpha=0.2)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    draw_curves(axes.flat, ["sst_rmse", "adt_rmse", "velocity_rmse", "eke_rmse"])
    for ax, title, unit in zip(
        axes.flat,
        ["SST", "ADT", "Geostrophic velocity", "EKE across three origins at each lead"],
        ["°C", "m", "m/s", "m²/s²"],
        strict=True,
    ):
        ax.set_title(title)
        ax.set_ylabel("RMSE (" + unit + ")")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out / "surface-lead-curves.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, m in models.items():
        for ax, key in zip(axes, ["0_700", "700_2000"], strict=True):
            ax.plot(
                range(1, 13),
                [v[key] / 1e9 for v in m["monthly_ohc"]],
                color=colors[name],
                ls="--" if "1730" in name else "-",
                label=name,
            )
    for ax, key, title in zip(
        axes, ["0_700", "700_2000"], ["OHC 0–700 m", "OHC 700–2,000 m"], strict=True
    ):
        ax.plot(
            range(1, 13),
            [
                v[key + "_climatology"] / 1e9
                for v in next(iter(models.values()))["monthly_ohc"]
            ],
            "k:",
            lw=1.7,
            label="Climatology",
        )
        ax.set_title(title)
        ax.set_xlabel("Calendar month after January initialization")
        ax.set_ylabel("Monthly OHC RMSE (GJ/m²)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out / "ohc-monthly-curves.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for name, m in models.items():
        axes[0].plot(
            [v["day"] for v in m["leads"]],
            [v["fixed_validation_surface_blend"] for v in m["leads"]],
            color=colors[name],
            label=name,
            ls="--" if "1730" in name else "-",
        )
        axes[1].plot(
            np.arange(74) * 5,
            m["latent_rms_mean"],
            color=colors[name],
            label=name,
            ls="--" if "1730" in name else "-",
        )
    axes[0].set_title("Surface blend, original fixed validation normalization")
    axes[0].set_ylabel("Mean normalized SST / velocity / EKE RMSE")
    axes[1].set_title("Mean latent RMS across annual cases")
    axes[1].set_ylabel("Latent RMS")
    for ax in axes:
        ax.set_xlabel("Forecast lead (days)")
        ax.axvline(30, color="grey", ls=":", lw=0.8)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out / "latent-and-blend.png", dpi=110)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, origin in zip(axes, config["origins"], strict=True):
        for name, m in models.items():
            ax.plot(
                [v["day"] for v in m["per_origin_sst"][origin]],
                [v["sst_rmse"] for v in m["per_origin_sst"][origin]],
                color=colors[name],
                ls="--" if "1730" in name else "-",
                label=name,
            )
        ax.set_title(origin)
        ax.set_xlabel("Forecast lead (days)")
        ax.set_ylabel("SST RMSE (°C)")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out / "per-origin-sst.png", dpi=110)
    plt.close(fig)
    # Native1:1 grid-cell maps. Exact integer axes dimensions preserve pixel layout.
    for tag, pair in [
        ("A-vs-E", ["A-1729", "E-1729"]),
        ("clip1-vs-clip5", ["clip1-1730", "clip5-1730"]),
    ]:
        width, height = 1220, 480
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        images = []
        for row, name in enumerate(pair):
            c = archives[name]
            fields = [
                np.where(
                    grid["mask"][0],
                    c["surface"][-1, 0] - c["reference_surface"][-1, 0],
                    np.nan,
                )
            ] + [
                np.where(
                    support[k],
                    (c["ohc"][-1, k] - c["reference_ohc"][-1, k]) / 1e9,
                    np.nan,
                )
                for k in range(2)
            ]
            for col, field in enumerate(fields):
                ax = fig.add_axes(
                    (
                        (70 + col * 390) / width,
                        (260 - row * 220) / height,
                        360 / width,
                        180 / height,
                    )
                )
                im = ax.imshow(
                    field,
                    origin="lower",
                    interpolation="nearest",
                    cmap=plt.get_cmap("RdBu_r", 240),
                    vmin=-[20, 30, 30][col],
                    vmax=[20, 30, 30][col],
                    aspect="auto",
                )
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(
                    name
                    + ": "
                    + ["day365 SST", "December OHC0–700", "December OHC700–2000"][col],
                    fontsize=9,
                )
                images.append(im)
                assert np.allclose(
                    ax.get_position().bounds[2:] * np.array([width, height]), [360, 180]
                )
        for col in range(3):
            cax = fig.add_axes(
                ((70 + col * 390) / width, 22 / height, 360 / width, 12 / height)
            )
            cb = fig.colorbar(
                images[col], cax=cax, orientation="horizontal", extend="both"
            )
            cb.set_label(
                "Prediction − observation (" + ("°C" if col == 0 else "GJ/m²") + ")",
                fontsize=8,
            )
            cb.ax.tick_params(labelsize=7)
        fig.savefig(
            a.out / ("errors-2021-" + tag + ".png"),
            dpi=100,
            pil_kwargs={"compress_level": 9},
        )
        plt.close(fig)
        # Preserve every map-cell RGB value with a palette shared by map and
        # colorbar; only antialiased text shades need quantization.
        path = a.out / ("errors-2021-" + tag + ".png")
        with Image.open(path) as source:
            source_rgb = source.convert("RGB")
            before = np.asarray(source_rgb)
            rectangles = [
                (40 + r * 220, 70 + c * 390) for r in range(2) for c in range(3)
            ]
            rgb = np.unique(
                np.concatenate(
                    [
                        before[y : y + 180, x : x + 360].reshape(-1, 3)
                        for y, x in rectangles
                    ]
                ),
                axis=0,
            )
            assert len(rgb) <= 241
            grays = np.repeat(
                np.linspace(0, 255, 256 - len(rgb), dtype=np.uint8)[:, None], 3, axis=1
            )
            palette_rgb = np.concatenate([rgb, grays])
            palette = Image.new("P", (1, 1))
            palette.putpalette(palette_rgb.flatten().tolist())
            indexed = source_rgb.quantize(palette=palette, dither=Image.Dither.NONE)
            indices = np.asarray(indexed).copy()
            lookup = {tuple(color): i for i, color in enumerate(rgb)}
            # Pillow's nearest-palette cache may approximate even exact colors.
            # Assign every actual cell color directly to its exact palette slot.
            for y, x in rectangles:
                colors, inverse = np.unique(
                    before[y : y + 180, x : x + 360].reshape(-1, 3),
                    axis=0,
                    return_inverse=True,
                )
                indices[y : y + 180, x : x + 360] = np.array(
                    [lookup[tuple(c)] for c in colors], dtype=np.uint8
                )[inverse].reshape(180, 360)
            indexed = Image.fromarray(indices, mode="P")
            indexed.putpalette(palette_rgb.flatten().tolist())
            after = np.asarray(indexed.convert("RGB"))
            for y, x in rectangles:
                np.testing.assert_array_equal(
                    before[y : y + 180, x : x + 360], after[y : y + 180, x : x + 360]
                )
            indexed.save(path, optimize=True)
    verification = dict(
        producer=config["producer"],
        origins=config["origins"],
        model_count=8,
        annual_payloads=24,
        verified_output_hashes=output_count,
        source_checkpoints_match_audit=True,
        first_month_readout_exact=True,
        fixed_ohc_support_cells=support.sum((1, 2)).tolist(),
        reused_payloads=12,
        new_payloads=12,
        annual_gpu_hours_including_retries=2.888611,
        campaign_gpu_hours=100.513611,
    )
    for model in models.values():
        model.pop("per_origin_sst")  # Full values are in the per-origin CSV.
        model.pop("climatology_leads")
    payload = dict(
        models=models,
        climatology_leads=baseline,
        normalization=normal,
        verification=verification,
        surface_blend_note="surface_blend in source metrics uses same-lead annual climatology. fixed_validation_surface_blend in plots uses original nine-origin fixed denominators.",
        eke_definition="Across three January origins at each fixed lead; not within-year temporal-anomaly EKE.",
    )
    (a.out / "results.json").write_text(
        json.dumps(payload, separators=(",", ":")) + "\n"
    )
    (a.out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (a.out / "checksums.txt").write_text("\n".join(checksums) + "\n")
    write_csv(a.out / "lead-metrics.csv", lead_rows)
    write_csv(a.out / "monthly-ohc.csv", ohc_rows)
    write_csv(a.out / "per-origin-sst.csv", case_rows)
    for f in a.out.iterdir():
        if not f.name.endswith(".license"):
            f.with_name(f.name + ".license").write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
    for name, m in models.items():
        v = m["leads"][-1]
        o = m["monthly_ohc"][-1]
        print(
            name,
            "SST",
            round(v["sst_rmse"], 3),
            "bias",
            round(v["sst_bias"], 3),
            "centered",
            round(v["sst_centered_rmse"], 3),
            "vel",
            round(v["velocity_rmse"], 3),
            "OHC",
            round(o["0_700"] / 1e9, 3),
            round(o["700_2000"] / 1e9, 3),
        )


if __name__ == "__main__":
    main()
