#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Export frozen initializer examples in the report's pinned training runtime.

This offline tool requires the original Samudra code layer and Torch environment,
not the Panel environment. The contract pins modules, checkpoints and saved arrays.
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEPTHS = [2.5, 10, 22.5, 40, 65, 105, 165, 250, 375, 550, 775, 1050, 1400, 1850]
# The 2021 observation history has no exact match in this OM4 time series.
ORIGINS = ["2015-01-01", "2018-01-01"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main(root):
    # Import the historical implementation only after verifying its contract.
    contract = json.loads((root / "contract.json").read_text())
    assert os.environ["SAMUDRA_CODE_COMMIT"] == contract["producer"]
    modules = {}
    for name, digest in contract["module_sha256"].items():
        module = importlib.import_module(name)
        assert module.__file__ and sha(module.__file__) == digest
        modules[name.rsplit(".", 1)[-1]] = module
    torch = importlib.import_module("torch")
    pd = importlib.import_module("pandas")
    rust = importlib.import_module("samudra.rust_data")
    location = importlib.import_module("samudra.utils.location")
    spec = contract["stages"]["obs08000"]
    run = Path(spec["run"])
    manifest = json.loads((run / "manifest.json").read_text())
    state, lineage = modules["observation_checkpoint"].load_fixed_checkpoint(
        run, spec["checkpoint"], "test", 8000, 8000
    )
    assert lineage["checkpoint_sha256"] == spec["checkpoint_sha256"]
    assert lineage["training_manifest_sha256"] == spec["manifest_sha256"]
    options = SimpleNamespace(**manifest["arguments"])
    options.output, options.name = str(root / "loader"), "viewer-initialization-export"
    om4 = modules["observation_joint"].load_om4(options)
    data = modules["observation_training"].Samples(
        options.data, "cuda", surface_fill="zero", global_observations=True
    )
    data.use_observation_normalization()
    torch.testing.assert_close(om4.mask.bool(), data.mask.bool())
    model = (
        modules["observation_model"]
        .ObservationTransfer.from_arguments(
            data.grid["names"].tolist(), manifest["arguments"]
        )
        .cuda()
        .eval()
    )
    model.load_state_dict(state, strict=True)
    del state
    source = om4.context_bundle.inference_source
    assert source is not None
    source = rust.native_om4_source(
        source,
        location.LocalLocation(path=str(Path(options.om4_data) / "OM4.zarr")),
        rust.create_rust_io_runtime(4),
    )
    dataset = om4.dataset(source)
    times = pd.DatetimeIndex([str(t) for t in source.time.values])
    names = list(om4.names)
    assert names == data.grid["names"].tolist()
    indices = np.array(
        [
            [names.index(f"{v}_{d}") for v in ["thetao", "so", "uo", "vo"]]
            for d in range(14)
        ]
    )
    mask = data.grid["mask"][indices].astype(bool)
    out = root / "output"
    out.mkdir()  # Refuse to overwrite an earlier export.
    meta = dict(
        producer=contract["producer"],
        script_sha256=sha(__file__),
        lineage=lineage,
        cases={},
        sources={},
        saved_observations=contract["saved_observations"],
    )

    def write(name, **arrays):
        path = out / name
        np.savez_compressed(path, **arrays)
        with np.load(path, allow_pickle=False) as loaded:
            for key, values in arrays.items():
                np.testing.assert_array_equal(loaded[key], values)

    def fields(value):
        result = np.where(mask, np.asarray(value)[indices], np.nan).astype(np.float32)
        assert np.isfinite(result[mask]).all()
        return result

    write(
        "grid.npz",
        lat=data.grid["lat"],
        lon=data.grid["lon"],
        depths=np.array(DEPTHS),
        mask=mask,
    )
    for path in [
        run / "manifest.json",
        Path(options.data) / "grid.npz",
        Path(options.data) / "statistics.npz",
        Path(options.om4_data) / "OM4.zarr/.zmetadata",
        Path(options.om4_data) / "OM4_means.zarr/.zmetadata",
        Path(options.om4_data) / "OM4_stds.zarr/.zmetadata",
    ]:
        meta["sources"][str(path)] = sha(path)
    shape = tuple(data.mask.shape[-2:])
    for origin in ORIGINS:
        sample_path = Path(options.data) / "test" / (origin[:7] + ".npz")
        sample = data.load(sample_path)
        desired = pd.DatetimeIndex(sample["raw"]["midpoints"][:19])
        matches = np.flatnonzero(times == desired[-1])
        assert len(matches) == 1
        index = int(matches[0]) - 18
        np.testing.assert_array_equal(times[index : index + 19].values, desired.values)
        loader = om4.native_loader(dataset, [[index]])
        batch = next(iter(loader))
        history, boundary = batch.get_initial_input()
        history = history.reshape(1, 19, len(names), *shape)
        past = boundary.reshape(1, 19, 3, *shape)
        native_mean = om4.native_mean[None, None, :, None, None]
        native_std = om4.native_std[None, None, :, None, None]

        def convert(value):
            return (
                (value * native_std + native_mean - data.mean) / data.std
            ) * data.mask

        truth = convert(history[:, -2:])
        surface = convert(history)[:, :, model.initializer.surface]
        t = source.time.values[index + 18]
        phase = 2 * math.pi * (t.dayofyr - 1) / 365.25
        season = data.tensor([math.sin(phase), math.cos(phase)])[
            None, :, None, None
        ].expand(1, 2, *shape)
        context = torch.cat((om4.geo[None], season), 1)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            initial = model.initializer(
                surface,
                past,
                context,
                data.mask,
                data.mask[model.initializer.surface].expand_as(surface),
                task="om4",
            )
            observation = model.initializer(
                sample["surface"][:, :19],
                model.adapt(sample["atmosphere"][:, :19]),
                sample["contexts"][:, 18],
                data.mask,
                sample["validity"][:, :19],
                task="observation",
            )
        predicted = data.physical(initial)[0, -1].cpu().numpy()
        target = data.physical(truth)[0, -1].cpu().numpy()
        reproduced = data.physical(observation)[0, -1].cpu().numpy()
        saved = contract["saved_observations"]["obs08000"]["files"][origin]
        assert sha(saved["path"]) == saved["sha256"]
        with np.load(saved["path"], allow_pickle=False) as archive:
            original = archive["initial"][-1]
        np.testing.assert_allclose(reproduced, original, rtol=1e-4, atol=1e-4)
        # Check normalization round trip against the physical native OM4 state.
        physical_native = (history * native_std + native_mean)[0, -1].cpu().numpy()
        np.testing.assert_allclose(
            target[data.grid["mask"].astype(bool)],
            physical_native[data.grid["mask"].astype(bool)],
            rtol=1e-5,
            atol=1e-5,
        )
        write(f"om4-{origin}.npz", prediction=fields(predicted), gold=fields(target))
        meta["cases"][origin] = {
            "history_midpoints": [str(v) for v in desired],
            "state_midpoint": str(desired[-1]),
            "observation_reproduction_max_abs": float(
                np.abs(reproduced - original).max()
            ),
        }
        meta["sources"][str(sample_path)] = sha(sample_path)
        print("OM4_COMPLETE", origin, flush=True)
        del batch, loader, history, past, initial, observation, sample
    # Observation examples reuse the original saved states; no new predictions.
    for stage, record in contract["saved_observations"].items():
        for origin, saved in record["files"].items():
            assert sha(saved["path"]) == saved["sha256"]
            with np.load(saved["path"], allow_pickle=False) as archive:
                initial_saved = archive["initial"][-1]
            write(f"{stage}-obs-{origin}.npz", prediction=fields(initial_saved))
            meta["sources"][saved["path"]] = saved["sha256"]
        print("OBS_COMPLETE", stage, flush=True)
    shutil.copy2(__file__, out / "export_interior.py")
    shutil.copy2(root / "contract.json", out / "contract.json")
    meta["files"] = {
        p.name: dict(sha256=sha(p), bytes=p.stat().st_size) for p in out.iterdir()
    }
    (out / "COMPLETE.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("EXPORT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    main(parser.parse_args().root)
