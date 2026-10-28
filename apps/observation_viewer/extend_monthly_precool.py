#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Add frozen pre-cooldown forecasts and OM4-input initial states to a catalog."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from export_interior import sha
from extend_monthly_temperature import integrated_heat
from prepare_monthly_replay import NAMES, REPORT

NAME = "A-replay-pre-cooldown"


def add_contemporaneous_context(meta):
    """Share matched simulation references without changing observation inputs."""
    for example in meta["interior_examples"].values():
        if example["source"] != "obs":
            continue
        matched = meta["interior_examples"].get(example["origin"] + "-om4")
        if matched is None:
            continue
        assert matched["source"] == "om4" and matched["origin"] == example["origin"]
        reference = matched["references"]["om4"]
        assert reference in meta["files"]
        example["references"]["om4"] = reference
        example["description"] += (
            " OM4 contemporaneous context is the full simulation state for the same final five-day history interval as the matched OM4-input example. It is simulation context, not observational truth; selecting it does not change the observation-history initialization or forcing."
        )


def annotate_readouts(meta):
    """Make the measured decoder-task limitation explicit without changing arrays."""
    for record in meta["records"].values():
        om4 = record.get("source") == "om4"
        record["surface_readout_description"] = (
            "This OM4 forecast jointly decodes all 77 physical channels. Observation surface forecasts decode only SST/SSH. Full-field decodes retain substantial diffusion noise in these checkpoints, so the source comparison also changes the decoder task."
            if om4
            else "This surface forecast decodes only SST/SSH; initialized interiors use a separate joint 77-channel readout."
        )
        record["heat_readout_description"] = (
            "The monthly OM4 decoder requests all 77 physical channels; the observation monthly decoder requests T/S only. These are different readout tasks."
            if om4
            else "The monthly decoder requests T/S only; initialized interiors use a separate joint 77-channel readout."
        )
    for example in meta["interior_examples"].values():
        example["description"] += (
            " Full-field initialization jointly decodes all 77 channels for both input sources. These checkpoints retain substantial diffusion noise in that readout, including their surface fields; this is not a validated full-state reconstruction."
        )
    for info in meta["models"].values():
        info["description"] += (
            "\n\n**Readout limitation:** full 77-channel reconstructions retain much more diffusion noise than two-channel surface forecasts, for both observation and OM4 inputs. OM4 annual readouts use the full task; observation annual readouts use surface-only or T/S-only tasks. Source comparisons therefore also change the decoder task. Holding the latent state fixed reproduces this difference; increasing the sampler from 32 to 128 steps barely helps. The existing arrays are retained so this failure remains visible."
        )


def extend(base, export, output):
    meta = json.loads((base / "catalog.json").read_text())
    config = json.loads((export / "paths.json").read_text())
    audit = json.loads((export / "AUDITED.json").read_text())
    qualified = json.loads((export / "QUALIFIED.json").read_text())
    assert audit["config_sha256"] == sha(export / "paths.json")
    assert qualified["audit_sha256"] == sha(export / "AUDITED.json")
    assert qualified["producer"] == config["producer"]
    for flag in (
        "paired_means_exact",
        "observation_monthly_overlap_exact",
        "om4_initialization_matches_annual",
        "surface_only_initialization",
    ):
        assert qualified[flag]
    assert qualified["retained_member"] == 0
    assert audit["own_model_forcing"] and audit["initialized_from_surface_history_only"]
    assert config["model"]["name"] == NAME and config["model"]["step"] == 2000
    assert config["obs_origins"] == ["2015-01-01", "2018-01-01", "2021-01-01"]
    assert config["om4_origins"] == ["2015-01-01", "2018-01-01"]
    assert "2021-01-01" in audit["excluded_om4_origins"]
    assert tuple(row["name"] for row in config["comparison_models"]) == NAMES
    old_files = dict(meta["files"])
    for name, info in old_files.items():
        assert sha(base / name) == info["sha256"], name
    shutil.copytree(base, output)
    surface_mask = np.load(output / meta["surface_mask"])
    interior_mask = np.load(output / meta["interior_mask"])
    heat_mask = np.load(output / meta["heat_mask"])
    h, w = len(meta["lat"]), len(meta["lon"])
    grid_path = next(p for p in audit["contracts"] if p.endswith("/grid.npz"))
    assert sha(export / "grid.npz") == audit["contracts"][grid_path]
    with np.load(export / "grid.npz", allow_pickle=False) as grid:
        np.testing.assert_array_equal(grid["lat"], meta["lat"])
        np.testing.assert_array_equal(grid["lon"], meta["lon"])
        np.testing.assert_array_equal(grid["depth"][:14], meta["depths"])
        slots = np.array([[38 + i, 57 + i, i, 19 + i] for i in range(14)])
        np.testing.assert_array_equal(grid["mask"][[38, 76]], surface_mask)
        np.testing.assert_array_equal(grid["mask"][slots], interior_mask)

    def write(name, value):
        name += ".npy"
        assert name not in meta["files"]
        path = output / name
        np.save(path, value, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
        meta["files"][name] = dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            bytes=path.stat().st_size,
            sha256=sha(path),
        )
        return name

    def verified(directory, filename):
        receipt = json.loads((directory / "COMPLETE.json").read_text())
        assert receipt["producer"] == config["producer"]
        assert (
            receipt["checkpoint_sha256"]
            == audit["contracts"][config["model"]["checkpoint"]]
        )
        path = directory / filename
        assert sha(path) == receipt["files"][filename]
        with np.load(path, allow_pickle=False) as saved:
            return dict(saved)

    def initial_values(c, single):
        prefix = "member_" if single else ""
        values = np.where(interior_mask, c[prefix + "initial_interior"], np.nan)
        assert values.shape == (14, 4, h, w)
        assert np.isfinite(values[interior_mask]).all()
        heat = np.where(heat_mask, c[prefix + "initial_ohc"] / 1e9, np.nan)
        np.testing.assert_allclose(
            integrated_heat(values[None, :, 0], heat_mask)[0],
            heat,
            rtol=2e-7,
            atol=1e-6,
            equal_nan=True,
        )
        return values, heat

    def om4_example(origin, gold):
        key = origin + "-om4"
        gold = np.where(interior_mask, gold, np.nan)
        assert gold.shape == (14, 4, h, w)
        assert np.isfinite(gold[interior_mask]).all()
        if key not in meta["interior_examples"]:
            reference = write("precool-" + origin + "-om4-gold", gold)
            meta["interior_examples"][key] = dict(
                origin=origin,
                label=origin + " (om4)",
                source="om4",
                models={},
                references=dict(om4=reference),
                description="Decoder reconstruction of the initial latent state from 19 five-day OM4 SST/SSH history bins, with OM4's own three normalized forcing fields. Only surface temperature and sea level are visible; full-depth T/S/U/V gold is held out for comparison. Gold is the final native five-day history mean immediately preceding the forecast origin, not a monthly analysis. OM4's native Julian calendar is retained. The 2021 OM4 example is excluded because its history-to-forecast boundary has a one-day source gap. Observation-only E models have never received full-field OM4 training; their OM4-input reconstructions are out-of-distribution probes. Initial decoder readouts are not fed back into the forecast.",
            )
        example = meta["interior_examples"][key]
        np.testing.assert_array_equal(
            np.load(output / example["references"]["om4"]), gold
        )
        return key, example

    for single in (False, True):
        readout = "draw1" if single else "mean8"
        model = "monthly-" + NAME + ("-draw1" if single else "")
        assert model not in meta["models"]
        description = (
            f"**{NAME}** — **48,702,016 parameters (~48.7M)**; global 1° (180×360). Shared masked-history initializer, deterministic two-slot/128-channel latent processor on a 45×90 grid, and window-conditioned diffusion decoder. Seed 1729; **2,000 main updates: 1,000 OM4 + 1,000 observation updates**, with detached long-lead replay on half the observation examples. **No observation cooldown**: fixed main-endpoint EMA before its reset at cooldown. No latent magnitude penalty or bound.\n\n"
            "Initialize once from 19 five-day SST/SSH bins, then evolve 73 five-day steps. Observation origins use ERA5 through the learned forcing adapter; OM4 origins use OM4's own normalized stress/heat-flux forcing. Observations have three starts (2015/2018/2021); OM4 has two (2015/2018), excluding the one-day 2021 source gap. OM4 dates preserve the native Julian calendar. Full-depth OM4 gold is never input to the initializer.\n\n"
            + (
                "Displayed fields are member zero of each eight-member, 32-step diffusion batch. Noise is sampled anew at each readout; this shares the deterministic latent trajectory with the paired mean and is not an independently evolving stochastic trajectory. "
                if single
                else "Displayed fields average eight diffusion draws with 32 sampler steps. "
            )
            + "Monthly temperature profiles are the exact decoder fields used in the displayed OHC integrals, with the surface layer replaced by calendar-weighted native SST from the same readout. Initial interiors use a separate diffusion batch of the initial latent state and are not fed back. IAP T/S is preceding-December context, not instantaneous truth; OM4 initialization gold is the actual preceding five-day full simulation state. Initialized observation U/V has no observed-current reconstruction supervision. Own persistence holds the decoded initial physical state fixed."
        )
        meta["models"][model] = dict(
            label=f"Monthly diffusion · {NAME} · {'1 draw' if single else '8-draw mean'}",
            short_label=f"{NAME} ({readout})",
            report_url=REPORT,
            description=description,
            lineage=dict(
                checkpoint=config["model"]["checkpoint"],
                checkpoint_sha256=audit["contracts"][config["model"]["checkpoint"]],
                checkpoint_policy="fixed main endpoint EMA before cooldown reset",
                global_step=2000,
                task_counts=dict(om4=1000, observation=1000),
                export_producer=config["producer"],
                decoder_readout=readout,
                retained_member=0 if single else None,
            ),
        )
        for i, case in enumerate(config["cases"]):
            c = verified(export / f"case-{i}", "forecast.npz")
            origin, source = case["origin"], case["source"]
            key = origin + ("-om4" if source == "om4" else "")
            prefix = "member_" if single else ""
            assert c["surface"].shape == (73, 2, h, w)
            assert c["temperature"].shape == (12, 14, h, w)
            assert c["months"].tolist() == [
                origin[:4] + f"-{m:02d}" for m in range(1, 13)
            ]
            if source == "om4":
                _, example = om4_example(origin, c["reference_initial"])
                if key not in meta["origins"]:
                    meta["origins"].append(key)
                record = meta["records"].setdefault(
                    key,
                    dict(
                        origin=origin,
                        label=origin + " (om4)",
                        source="om4",
                        calendar="julian",
                        heat_months=c["months"].tolist(),
                        models={},
                    ),
                )
            else:
                example = meta["interior_examples"][key]
                record = meta["records"][key]
            references = dict(
                surface_reference=np.where(
                    surface_mask, c["reference_surface"][:, :2], np.nan
                ),
                heat_reference=np.where(heat_mask, c["reference_ohc"] / 1e9, np.nan),
                temperature_reference=np.where(
                    interior_mask[:, 0], c["reference_temperature"], np.nan
                ),
            )
            if source == "om4":
                np.testing.assert_allclose(
                    integrated_heat(references["temperature_reference"], heat_mask),
                    references["heat_reference"],
                    rtol=2e-7,
                    atol=1e-6,
                    equal_nan=True,
                )
            for field, value in references.items():
                if field in record:
                    np.testing.assert_array_equal(
                        np.load(output / record[field]), value
                    )
                else:
                    record[field] = write(f"precool-{key}-{field}", value)
            initial, held_heat = initial_values(c, single)
            temperature = np.where(
                interior_mask[:, 0], c[prefix + "temperature"], np.nan
            )
            heat = np.where(heat_mask, c[prefix + "ohc"] / 1e9, np.nan)
            surface = np.where(surface_mask, c[prefix + "surface"], np.nan)
            assert np.isfinite(temperature[:, interior_mask[:, 0]]).all()
            assert np.isfinite(surface[:, surface_mask]).all()
            np.testing.assert_allclose(
                integrated_heat(temperature, heat_mask),
                heat,
                rtol=2e-7,
                atol=1e-6,
                equal_nan=True,
            )
            stem = f"precool-{readout}-{key}"
            initial_name = write(stem + "-interior", initial)
            record["models"][model] = dict(
                interior=initial_name,
                surface=write(stem + "-surface", surface),
                surface_persistence=write(
                    stem + "-held-surface",
                    np.broadcast_to(
                        np.where(surface_mask, c[prefix + "initial_surface"], np.nan),
                        surface.shape,
                    ),
                ),
                temperature=write(stem + "-temperature", temperature),
                temperature_persistence=write(
                    stem + "-held-temperature",
                    np.broadcast_to(initial[:, 0], temperature.shape),
                ),
                heat=write(stem + "-heat", heat),
                heat_persistence=write(
                    stem + "-held-heat", np.broadcast_to(held_heat, heat.shape)
                ),
            )
            example["models"][model] = initial_name
            print(f"Verified {NAME} / {source} / {origin} / {readout}", flush=True)

    directory = export / "om4-initializations"
    receipt = json.loads((directory / "COMPLETE.json").read_text())
    assert len(receipt["files"]) == 12
    for row in config["comparison_models"]:
        name = row["name"]
        for origin in config["om4_origins"]:
            c = verified(directory, name + "-" + origin + ".npz")
            key, example = om4_example(origin, c["reference_initial"])
            for single in (False, True):
                model = "monthly-" + name + ("-draw1" if single else "")
                assert (
                    meta["models"][model]["lineage"]["checkpoint_sha256"]
                    == audit["contracts"][row["checkpoint"]]
                )
                initial, _ = initial_values(c, single)
                filename = write(
                    f"precool-init-{name}-{'draw1' if single else 'mean8'}-{key}",
                    initial,
                )
                example["models"][model] = filename
                meta["records"][key]["models"][model] = dict(interior=filename)
                info = meta["models"][model]
                if origin == config["om4_origins"][0]:
                    info["description"] += (
                        "\n\nOM4-input initialization examples also use 19 five-day SST/SSH bins and OM4's own forcing, with full T/S/U/V simulation gold held out for comparison. These are initialization readouts only; annual OM4 forecasts are saved only for A-replay-pre-cooldown. Observation-only E models have no full-field model-data supervision and their OM4-input examples are out-of-distribution probes."
                    )
            print(f"Verified OM4 initial state: {name} / {origin}", flush=True)
    provenance = output / "precool-provenance"
    provenance.mkdir()
    for name in ("paths.json", "AUDITED.json", "QUALIFIED.json"):
        shutil.copyfile(export / name, provenance / name)
    for directory in [
        *(export / f"case-{i}" for i in range(5)),
        export / "om4-initializations",
    ]:
        shutil.copyfile(
            directory / "COMPLETE.json",
            provenance / (directory.name + "-COMPLETE.json"),
        )
    meta["precool_export_provenance"] = dict(
        producer=config["producer"],
        previous_catalog_sha256=sha(base / "catalog.json"),
        checkpoint_policy=audit["checkpoint_policy"],
        retained_member=0,
        paired_readouts=True,
        own_model_forcing=True,
        surface_only_initialization=True,
        excluded_om4_origins=audit["excluded_om4_origins"],
        om4_forecast_origins=config["om4_origins"],
        observation_forecast_origins=config["obs_origins"],
        om4_initialization_physical_models=7,
        no_retraining=True,
    )
    add_contemporaneous_context(meta)
    annotate_readouts(meta)
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, info in meta["files"].items():
        assert sha(output / name) == info["sha256"], name
    for name, info in old_files.items():
        assert meta["files"][name] == info
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    meta = extend(args.base, args.export, args.output)
    print(
        json.dumps(
            dict(
                models=len(meta["models"]),
                origins=meta["origins"],
                arrays=len(meta["files"]),
            )
        )
    )


if __name__ == "__main__":
    main()
