#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Add frozen diffusion ensemble means and draws, without inference or smoothing."""

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np


def sha(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extend(base, evaluation, output, key, label):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", key):
        raise ValueError("Invalid model key")
    if output.exists():
        raise ValueError("Choose a new immutable output bundle")
    meta = json.loads((base / "catalog.json").read_text())
    inp = json.loads((evaluation / "input.json").read_text())
    complete = json.loads((evaluation / "COMPLETE.json").read_text())
    annual = evaluation / "annual"
    if sha(annual / "COMPLETE.json") != complete["receipts"]["annual"]:
        raise ValueError("Annual receipt checksum differs")
    if any(key + suffix in meta["models"] for suffix in ["-mean8", "-draw1"]):
        raise ValueError("Model already exists")
    for name, info in meta["files"].items():
        if sha(base / name) != info["sha256"]:
            raise ValueError("Base checksum differs: " + name)
    shutil.copytree(base, output)
    surface_mask = np.load(base / meta["surface_mask"])
    interior_mask = np.load(base / meta["interior_mask"])
    sources = {}

    def write(name, values):
        filename = name + ".npy"
        path = output / filename
        if path.exists():
            raise ValueError("Existing array: " + filename)
        np.save(path, values, allow_pickle=False)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), values)
        meta["files"][filename] = dict(
            sha256=sha(path),
            shape=list(values.shape),
            dtype=str(values.dtype),
            bytes=path.stat().st_size,
        )
        return filename

    for suffix, display in [("-mean8", "mean of 8"), ("-draw1", "draw 1")]:
        name = key + suffix
        meta["models"][name] = dict(
            label=f"{label} · {display}",
            short_label=f"{label} · {display}",
            report_url="https://github.com/m2lines/Samudra/blob/codex/diffusion-interior-beta/docs/experiments/diffusion-interior-beta/scratch-replay.md",
            description=f"**{label} · {display}.** Latent processor with a physical-field diffusion decoder; eight members, 32 sampling steps. Independently sampled decodes at each lead: draw 1 is not a coherent stochastic trajectory. Surface forecasts are five-day means. The ensemble mean initializer has all saved physical fields; draw 1 initialization only has temperature at 2.5 and 550 m and salinity at 550 m. Other draw initialization fields are unavailable (white). IAP initialization comparisons are preceding-month context, not instantaneous truth. These experiments differ in architecture, training exposure and objectives from U-global and observation-only controls. No spatial smoothing is applied.",
            lineage=dict(
                checkpoint_sha256=inp["checkpoint_sha256"],
                global_step=inp["step"],
                task_counts=inp["counts"],
                evaluation_input=inp,
            ),
        )
    for origin in meta["origins"]:
        src = annual / f"{origin}.npz"
        member_src = annual / f"members-{origin}.npz"
        sources[origin] = {
            "arrays": str(src.resolve()),
            "sha256": sha(src),
            "members": str(member_src.resolve()),
            "members_sha256": sha(member_src),
        }
        with (
            np.load(src, allow_pickle=False) as data,
            np.load(member_src, allow_pickle=False) as members,
        ):
            np.testing.assert_array_equal(members["lat"], meta["lat"])
            np.testing.assert_array_equal(members["lon"], meta["lon"])
            np.testing.assert_array_equal(members["leads_days"], np.arange(5, 366, 5))
            np.testing.assert_array_equal(
                members["channel_names"], ["thetao_0", "zos", "thetao_9", "so_9"]
            )
            np.testing.assert_array_equal(members["mask"][:2], surface_mask)
            np.testing.assert_array_equal(
                np.where(surface_mask, data["reference"][:, :2], np.nan),
                np.load(base / meta["records"][origin]["surface_reference"]),
            )
            fields = members["fields"]
            for suffix in ["-mean8", "-draw1"]:
                name = key + suffix
                if suffix == "-mean8":
                    surface = data["surface"]
                    # Historical physical layout: U[0:19], V[19:38], T[38:57], S[57:76], SSH.
                    ids = np.array([[38 + i, 57 + i, i, 19 + i] for i in range(14)])
                    interior = data["initial"][-1][ids]
                    np.testing.assert_allclose(
                        surface, fields[:, :, :2].mean(0), rtol=2e-5, atol=2e-5
                    )
                else:
                    surface = fields[0, :, :2]
                    interior = np.full(interior_mask.shape, np.nan, dtype=np.float32)
                    initial = members["initial"][0, -1]
                    interior[0, 0], interior[9, 0], interior[9, 1] = (
                        initial[0],
                        initial[2],
                        initial[3],
                    )
                record = dict(
                    surface=write(
                        f"diffusion-{name}-{origin}-surface",
                        np.where(surface_mask, surface, np.nan),
                    ),
                    interior=write(
                        f"diffusion-{name}-{origin}-interior",
                        np.where(interior_mask, interior, np.nan),
                    ),
                )
                meta["records"][origin]["models"][name] = record
                meta["interior_examples"][origin]["models"][name] = record["interior"]
    provenance = output / ("diffusion-provenance-" + key)
    provenance.mkdir()
    for file in ["input.json", "COMPLETE.json"]:
        shutil.copy2(evaluation / file, provenance / file)
    (provenance / "sources.json").write_text(json.dumps(sources, indent=2) + "\n")
    (output / "catalog.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["base", "evaluation", "output"]:
        parser.add_argument("--" + arg, required=True, type=Path)
    parser.add_argument("--key", required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    extend(args.base, args.evaluation, args.output, args.key, args.label)
