#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare verified annual replay or model-pretraining runs with original pilots."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from samudra.experiments.monthly_annual import fixed_ohc_support
from samudra.experiments.observation_pilot import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--comparison", choices=["replay", "pretraining"], default="replay")
    p.add_argument("--qualification-seconds", type=int, default=288)
    p.add_argument("--evaluation-seconds", type=int, default=994)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    models, cases, references, checksums = {}, {}, {}, []
    origins, baseline, grid = None, None, None
    output_count = 0
    original_arms = "AE" if a.comparison == "replay" else "ABCDE"
    original_names = [arm + "-1729" for arm in original_arms]
    for source in [a.original, a.root]:
        config = json.loads((source / "paths.json").read_text())
        audit = json.loads((source / "AUDITED.json").read_text())
        qualified = json.loads((source / "QUALIFIED.json").read_text())
        assert audit["config_sha256"] == digest(source / "paths.json")
        assert qualified["audit_sha256"] == digest(source / "AUDITED.json")
        assert qualified["producer"] == config["producer"]
        assert qualified["first_month_readout_exact"]
        finished = json.loads((source / "COMPLETE.json").read_text())
        assert finished["producer"] == config["producer"]
        assert finished["models"] == config["models"]
        assert finished["origins"] == config["origins"]
        grid_path = next(k for k in audit["contracts"] if k.endswith("/grid.npz"))
        assert digest(source / "grid.npz") == audit["contracts"][grid_path]
        current_grid = dict(np.load(source / "grid.npz"))
        if grid is None:
            grid, origins = current_grid, config["origins"]
        else:
            assert config["origins"] == origins
            for key in grid:
                np.testing.assert_array_equal(grid[key], current_grid[key])
        support = fixed_ohc_support(current_grid)
        assert support.sum((1, 2)).tolist() == [38933, 37190]
        for row in config["models"]:
            if source == a.original and row["name"] not in original_names:
                continue
            name = row["name"].removesuffix("-1729")
            directory = source / row["name"]
            complete = json.loads((directory / "COMPLETE.json").read_text())
            metrics = json.loads((directory / "metrics.json").read_text())
            assert complete["model"] == row and complete["origins"] == origins
            assert complete["producer"] == metrics["producer"] == config["producer"]
            assert metrics["checkpoint_sha256"] == audit["contracts"][row["checkpoint"]]
            assert metrics["fixed_ohc_support_cells"] == [38933, 37190]
            assert len(metrics["leads"]) == 73 and len(metrics["monthly_ohc"]) == 12
            checksums.append(
                f"{metrics.pop('checkpoint_sha256')}  checkpoint:{name}  # pragma: allowlist secret"
            )
            for filename, sha in complete["outputs"].items():
                assert digest(directory / filename) == sha
                checksums.append(
                    f"{sha}  {name}/{filename}  # pragma: allowlist secret"
                )
                output_count += 1
            if baseline is None:
                baseline = metrics["climatology_leads"]
            else:
                assert metrics["climatology_leads"] == baseline
            latents, per_origin = [], {}
            for origin in origins:
                with np.load(directory / (origin + ".npz")) as z:
                    c = dict(z)
                assert c["surface"].shape == (73, 2, 180, 360)
                assert c["ohc"].shape == (12, 2, 180, 360)
                assert c["latent_rms"].shape == (74,)
                assert np.isfinite(c["surface"]).all()
                assert np.isfinite(c["latent_rms"]).all()
                assert np.isfinite(np.where(support, c["ohc"], 0)).all()
                if origin not in references:
                    references[origin] = (c["reference_surface"], c["reference_ohc"])
                else:
                    np.testing.assert_array_equal(
                        c["reference_surface"], references[origin][0]
                    )
                    np.testing.assert_array_equal(
                        c["reference_ohc"], references[origin][1]
                    )
                latents.append(c["latent_rms"])
                area = np.cos(np.deg2rad(grid["lat"]))[:, None]
                error = c["surface"][:, 0] - c["reference_surface"][:, 0]
                valid = np.isfinite(c["reference_surface"][:, 0]) & grid["mask"][0]
                weights = area * valid
                per_origin[origin] = np.sqrt(
                    np.sum(np.where(valid, error**2, 0) * weights, axis=(1, 2))
                    / weights.sum((1, 2))
                ).tolist()
                if origin == "2021-01-01":
                    cases[name] = c
            metrics.pop("climatology_leads")
            metrics["reused_origin_count"] = len(metrics.pop("reused_rollouts", {}))
            metrics["latent_rms_mean"] = np.mean(latents, 0).tolist()
            metrics["per_origin_sst"] = per_origin
            models[name] = metrics
    assert len(models) == len(original_arms) + 4 and output_count == len(models) * 4
    assert grid is not None and baseline is not None and origins is not None
    colors = dict(
        zip(models, plt.get_cmap("tab10")(np.arange(len(models))), strict=True)
    )
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, key, title in zip(
        axes.flat,
        ["sst_rmse", "adt_rmse", "velocity_rmse", "eke_rmse"],
        ["SST RMSE (°C)", "ADT RMSE (m)", "Velocity RMSE (m/s)", "EKE RMSE (m²/s²)"],
        strict=True,
    ):
        for name, model in models.items():
            ax.plot(
                [v["day"] for v in model["leads"]],
                [v[key] for v in model["leads"]],
                label=name,
                color=colors[name],
                ls="--" if name in original_arms else "-",
            )
        ax.plot(
            [v["day"] for v in baseline],
            [v[key] for v in baseline],
            "k:",
            label="Climatology",
        )
        ax.set_title(title)
        ax.set_xlabel("Forecast lead (days)")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out / "annual-surface-curves.png", dpi=115)
    plt.close(fig)
    if a.comparison == "pretraining":
        path = a.out / "annual-surface-curves.png"
        with Image.open(path) as saved:
            saved.convert("RGB").quantize(colors=256).save(path, optimize=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for name, model in models.items():
        for ax, key in zip(axes[:2], ["0_700", "700_2000"], strict=True):
            ax.plot(
                range(1, 13),
                [v[key] / 1e9 for v in model["monthly_ohc"]],
                label=name,
                color=colors[name],
            )
        axes[2].plot(
            np.arange(74) * 5, model["latent_rms_mean"], label=name, color=colors[name]
        )
    for ax, key, title in zip(
        axes[:2], ["0_700", "700_2000"], ["OHC 0–700 m", "OHC 700–2,000 m"], strict=True
    ):
        ax.plot(
            range(1, 13),
            [
                v[key + "_climatology"] / 1e9
                for v in next(iter(models.values()))["monthly_ohc"]
            ],
            "k:",
            label="Climatology",
        )
        ax.set_title(title + " RMSE (GJ/m²)")
        ax.set_xlabel("Calendar month")
    axes[2].set_title("Mean latent RMS")
    axes[2].set_xlabel("Forecast lead (days)")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(a.out / "annual-ohc-latent.png", dpi=110)
    plt.close(fig)
    map_groups = (
        {arm: [arm, arm + "-extend", arm + "-replay"] for arm in "AE"}
        if a.comparison == "replay"
        else {
            "OM4": ["A", "P-OM4-surface", "P-OM4-multitask"],
            "mix": ["C", "P-mix-surface", "P-mix-multitask"],
        }
    )
    for arm, names in map_groups.items():
        width, height = 1220, 740
        fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
        rectangles, images = [], []
        for row, name in enumerate(names):
            c = cases[name]
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
                x, y = 70 + 390 * col, 510 - 220 * row
                ax = fig.add_axes((x / width, y / height, 360 / width, 180 / height))
                im = ax.imshow(
                    field,
                    origin="lower",
                    interpolation="nearest",
                    cmap=plt.get_cmap(
                        "RdBu_r", 128 if a.comparison == "replay" else 64
                    ),
                    vmin=-[20, 30, 30][col],
                    vmax=[20, 30, 30][col],
                    aspect="auto",
                )
                ax.set_axis_off()
                ax.set_title(
                    name
                    + ": "
                    + ["day365 SST", "December OHC0–700", "December OHC700–2000"][col],
                    fontsize=9,
                )
                rectangles.append((height - y - 180, x))
                images.append(im)
                assert np.allclose(
                    ax.get_position().bounds[2:] * np.array([width, height]), [360, 180]
                )
        for col in range(3):
            cax = fig.add_axes(
                ((70 + col * 390) / width, 40 / height, 360 / width, 12 / height)
            )
            cb = fig.colorbar(
                images[col], cax=cax, orientation="horizontal", extend="both"
            )
            cb.set_label(
                "Prediction − observation (" + ("°C" if col == 0 else "GJ/m²") + ")",
                fontsize=8,
            )
            cb.ax.tick_params(labelsize=7)
        path = a.out / ("annual-errors-2021-" + arm + ".png")
        fig.savefig(path, dpi=100, pil_kwargs={"compress_level": 9})
        plt.close(fig)
        with Image.open(path) as image:
            source_rgb = image.convert("RGB")
            before = np.asarray(source_rgb)
            rgb = np.unique(
                np.concatenate(
                    [
                        before[y : y + 180, x : x + 360].reshape(-1, 3)
                        for y, x in rectangles
                    ]
                ),
                axis=0,
            )
            assert len(rgb) <= 129
            grays = np.repeat(
                np.linspace(0, 255, 256 - len(rgb), dtype=np.uint8)[:, None], 3, axis=1
            )
            palette_rgb = np.concatenate([rgb, grays])
            palette = Image.new("P", (1, 1))
            palette.putpalette(palette_rgb.flatten().tolist())
            indices = np.asarray(
                source_rgb.quantize(palette=palette, dither=Image.Dither.NONE)
            ).copy()
            lookup = {tuple(color): i for i, color in enumerate(rgb)}
            for y, x in rectangles:
                rgb_values, inverse = np.unique(
                    before[y : y + 180, x : x + 360].reshape(-1, 3),
                    axis=0,
                    return_inverse=True,
                )
                indices[y : y + 180, x : x + 360] = np.array(
                    [lookup[tuple(c)] for c in rgb_values], dtype=np.uint8
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
        model_count=len(models),
        annual_cases=len(models) * len(origins),
        output_hashes_verified=output_count,
        origins=origins,
        exact_cell_pixels=1,
        support_cells=support.sum((1, 2)).tolist(),
        new_annual_gpu_hours=(a.qualification_seconds + a.evaluation_seconds * 4)
        / 3600,
    )
    (a.out / "annual-results.json").write_text(
        json.dumps(
            dict(models=models, climatology_leads=baseline, verification=verification),
            separators=(",", ":"),
        )
        + "\n"
    )
    (a.out / "annual-checksums.txt").write_text("\n".join(checksums) + "\n")
    for f in a.out.iterdir():
        license_path = f.with_name(f.name + ".license")
        if not f.name.endswith(".license") and not license_path.exists():
            license_path.write_text(
                "SPDX-FileCopyrightText: 2026 Samudra Authors\nSPDX-License-Identifier: CC-BY-4.0\n"
            )
    for name, model in models.items():
        print(
            name,
            json.dumps(
                dict(
                    leads={
                        day: model["leads"][day // 5 - 1] for day in [30, 90, 180, 365]
                    },
                    december=model["monthly_ohc"][-1],
                    latent_start=model["latent_rms_mean"][0],
                    latent_end=model["latent_rms_mean"][-1],
                )
            ),
        )


if __name__ == "__main__":
    main()
