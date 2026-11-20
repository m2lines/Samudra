# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Create a mask-only Zarr view and matched configs; never modify source data."""

import argparse
import copy
import json
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
import yaml


def write_view_metadata(source: Path, masks: Path, destination: Path) -> None:
    # DirectoryStore consolidation does not descend into symlinked arrays.
    # Copy their authoritative metadata and replace only the local mask entries.
    metadata = json.loads((source / ".zmetadata").read_text())
    replacements = json.loads((masks / ".zmetadata").read_text())["metadata"]
    for key, value in replacements.items():
        if key.startswith("mask_"):
            metadata["metadata"][key] = value
    (destination / ".zmetadata").write_text(json.dumps(metadata))
    original = xr.open_zarr(source, consolidated=True, decode_times=False)
    view = xr.open_zarr(destination, consolidated=True, decode_times=False)
    assert set(view.variables) == set(original.variables)
    assert dict(view.sizes) == dict(original.sizes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("data", type=Path)
    a = p.parse_args()
    root, data = a.root.resolve(), a.data.resolve()
    view = root / "nonpartial-data"
    view.mkdir(exist_ok=True)
    dest = view / "OM4.zarr"
    dest.mkdir(exist_ok=True)
    masks = root / "geometry/masks.zarr"
    original = xr.open_zarr(data / "OM4.zarr", decode_times=False)
    strict = xr.open_zarr(masks, decode_times=False)
    np.testing.assert_array_equal(original.x, strict.x)
    np.testing.assert_array_equal(original.y, strict.y)
    for k in range(19):
        assert not np.any(strict[f"mask_{k}"].values & ~original[f"mask_{k}"].values)
    for entry in (data / "OM4.zarr").iterdir():
        if (dest / entry.name).exists() or entry.name == ".zmetadata":
            continue
        if entry.name.startswith("mask_") and entry.is_dir():
            shutil.copytree(masks / entry.name, dest / entry.name)
        elif entry.name.startswith("."):
            shutil.copy2(entry, dest / entry.name)
        else:
            (dest / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
    write_view_metadata(data / "OM4.zarr", masks, dest)
    for name in ["OM4_means.zarr", "OM4_stds.zarr"]:
        if not (view / name).exists():
            (view / name).symlink_to(data / name, target_is_directory=True)
    source = root / "source"
    model = yaml.safe_load(
        (source / "src/samudra/configs/samudra_om4_v2/model.yaml").read_text()
    )
    model["use_bfloat16"] = False
    common = dict(
        disk_mode=True,
        save_freq=5,
        validation_image_log_freq=10,
        rollout_validation=dict(days=[360], frequency=1, steps_forward=1),
        validation_score="rollout_rmse",
        epochs=70,
        batch_size=4,
        learning_rate=0.0006,
        gradient_accumulation_steps=1,
        scheduler=dict(type="cosine"),
        loss=dict(type="dynamic", metric="mse", limit=20),
        finetune=False,
        resume_ckpt_path=None,
        data_stride=[1],
        steps=[4],
        step_transition=[],
        preemptible=False,
        backend="auto",
        test_using_ema=True,
        ema_decay=0.999,
        faster_decay_at_start=True,
        model=model,
        data=dict(
            sources=[
                dict(
                    type="om4",
                    prognostic_vars_key="thermo_dynamic_all",
                    boundary_vars_key="tau_hfds",
                    grid_type="gaussian",
                    train_time=dict(start="1975-01-03", end="2013-09-30"),
                    val_time=dict(start="2013-10-05", end="2014-10-05"),
                    inference_time=dict(start="2015-01-01", end="2016-01-10"),
                    data_location="OM4.zarr",
                    data_means_location="OM4_means.zarr",
                    data_stds_location="OM4_stds.zarr",
                )
            ],
            loading=dict(
                type="rust",
                max_concurrent_reads=8,
                prefetch_batches=2,
                prefetch_to_device=True,
            ),
            inference_loading=dict(num_workers=4, persistent_workers=True),
            input_steps=2,
            output_steps=2,
            loader_version="om4-torch",
            normalize_before_mask=True,
            masked_fill_value=0.0,
        ),
    )
    for arm in ["baseline", "nonpartial"]:
        folder = root / arm
        folder.mkdir(exist_ok=True)
        cfg = copy.deepcopy(common)
        cfg["experiment"] = dict(
            name=arm,
            rand_seed=15,
            base_output_dir=str(root / "output"),
            data_root=str(data if arm == "baseline" else view),
            wandb=dict(
                mode="online",
                project="default",
                entity="ocean_emulators",
                group="om4-nonpartial-v2-20261105",
                tags=["nonpartial", "v2", "1deg", arm],
            ),
        )
        (folder / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        # Four independent 360-day forecasts, each using two genuine history states.
        time_axis = xr.decode_cf(original[["time"]]).time
        decoded = time_axis.values
        for year in [2015, 2017, 2019, 2021]:
            start = int(np.flatnonzero(time_axis.dt.year.values == year)[0])
            end = (
                start + 73
            )  # 2 histories + 72 forecast states = 360 days from latest history.
            ev = dict(
                debug=False,
                save_zarr=True,
                disk_mode=True,
                backend="cuda",
                num_model_steps_forward=6,
                ckpt_path=str(root / "output" / arm / "saved_nets/ema_ckpt.pt"),
                experiment=copy.deepcopy(cfg["experiment"]),
                data=copy.deepcopy(cfg["data"]),
                model=model,
            )
            ev["experiment"].update(
                name=f"{arm}-{year}", base_output_dir=str(root / "eval")
            )
            ev["experiment"]["wandb"]["mode"] = "disabled"
            ev["data"]["sources"][0]["inference_time"] = dict(
                start=str(decoded[start])[:10], end=str(decoded[end])[:10]
            )
            (folder / f"eval-{year}.yaml").write_text(
                yaml.safe_dump(ev, sort_keys=False)
            )
    (root / "preparation.json").write_text(
        json.dumps(
            dict(
                data=str(data),
                view=str(view),
                config_only_difference=[
                    "experiment.name",
                    "experiment.data_root",
                    "experiment.wandb.tags",
                ],
                seed=15,
                epochs=70,
                effective_batch=16,
                eval_years=[2015, 2017, 2019, 2021],
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
