# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the browser-based Colab quickstart."""

import json
import logging
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

import cftime
import nbformat
import numpy as np
import pytest
import torch
import xarray as xr

from samudra.config import DynamicLossConfig, SamudraConfig, TrainConfig
from samudra.constants import build_om4_spec
from samudra.train import Trainer
from samudra.utils.multiton import MultitonScope
from scripts.clone_data import rechunk_for_output, select_om4_variables
from scripts.download_quickstart_data import (
    DOWNLOAD_FINGERPRINT,
    EXPECTED_STORES,
    clear_incomplete_download,
    download_is_current,
)

QUICKSTART_CONFIG = "quickstart/train.yaml"
REPO_ROOT = Path(__file__).resolve().parent.parent

_MOCK_OVERRIDES = [
    "--train_time.start",
    "1975-08-15",
    "--train_time.end",
    "1975-09-25",
    "--val_time.start",
    "1975-10-20",
    "--val_time.end",
    "1975-11-20",
    "--data.sources",
    json.dumps(
        [
            {
                "data_location": "data.zarr",
                "data_means_location": "means.nc",
                "data_stds_location": "stds.nc",
            }
        ]
    ),
]


def test_quickstart_download_selects_only_requested_variables_and_depths(tmp_path):
    dataset_spec = build_om4_spec(
        prognostic_vars_key="thermo_dynamic_5",
        boundary_vars_key="tau_hfds",
    )
    coords = {
        "time": list(range(21)),
        "lev": list(dataset_spec.depth_levels),
        "lat": [0.0],
        "lon": [0.0],
    }
    field = np.zeros((21, 1, 1), dtype=np.float32)
    variables = {
        f"{base}_lev_{str(depth).replace('.', '_')}": (
            ("time", "lat", "lon"),
            field,
        )
        for base in ("uo", "vo", "thetao", "so")
        for depth in dataset_spec.depth_levels[:6]
    }
    variables.update(
        {
            name: (("time", "lat", "lon"), field)
            for name in ("zos", "tauuo", "tauvo", "hfds", "unused")
        }
    )
    variables[dataset_spec.mask_all_levels_var] = (
        ("lev", "lat", "lon"),
        np.ones((len(dataset_spec.depth_levels), 1, 1), dtype=np.int8),
    )
    source = xr.Dataset(variables, coords=coords)
    for variable in source.data_vars.values():
        if "time" in variable.dims:
            variable.encoding["chunks"] = (1, 1, 1)

    selected = select_om4_variables(source, dataset_spec)
    selected = rechunk_for_output(selected, time_steps=10)

    expected = set(
        dataset_spec.prognostic_var_names
        + dataset_spec.boundary_var_names
        + [dataset_spec.mask_all_levels_var]
    )
    assert set(selected.data_vars) == expected
    assert "thetao_5" not in selected
    assert "unused" not in selected
    assert selected.sizes["lev"] == len(dataset_spec.depth_levels)
    assert selected["thetao_0"].chunks is not None
    assert selected["thetao_0"].chunks[0] == (10, 10, 1)

    output = tmp_path / "filtered.zarr"
    selected.to_zarr(output)
    reopened = xr.open_zarr(output)
    assert reopened["thetao_0"].encoding["chunks"][0] == 10


def test_quickstart_config_contract():
    cfg = TrainConfig.from_yaml_and_cli(
        [str(REPO_ROOT / "configs" / QUICKSTART_CONFIG)]
    )

    assert cfg.epochs == 5
    assert cfg.steps == [1]
    assert cfg.data.hist == 0
    assert cfg.data.dataset.prognostic_vars_key == "thermo_dynamic_5"
    assert cfg.data.dataset.boundary_vars_key == "tau_hfds"
    assert isinstance(cfg.model, SamudraConfig)
    assert cfg.model.unet.ch_width == [280, 380, 480, 520]
    assert cfg.model.unet.core_block.upscale_factor == 2
    assert cfg.model.unet.up_sampling_block == "zonally_periodic_upsample"
    assert isinstance(cfg.loss, DynamicLossConfig)
    assert cfg.loss.metric == "mse"
    assert cfg.loss.limit == 20

    download_start = cftime.DatetimeJulian(1958, 1, 3, 12)
    download_last = download_start + timedelta(days=5 * (250 - 1))
    assert cfg.val_time.end.datetime == download_last


def test_quickstart_download_cache_is_versioned_and_safely_cleaned(tmp_path):
    marker = tmp_path / ".quickstart_download_complete"
    unrelated = tmp_path / "keep-me.txt"
    unrelated.write_text("unrelated\n", encoding="utf-8")
    marker.write_text(DOWNLOAD_FINGERPRINT + "\n", encoding="utf-8")
    for store in EXPECTED_STORES:
        (tmp_path / store).mkdir()

    assert download_is_current(tmp_path)

    marker.write_text("old-version\n", encoding="utf-8")
    assert not download_is_current(tmp_path)
    clear_incomplete_download(tmp_path)

    assert not marker.exists()
    assert all(not (tmp_path / store).exists() for store in EXPECTED_STORES)
    assert unrelated.read_text(encoding="utf-8") == "unrelated\n"


def test_quickstart_notebook_is_valid_and_current():
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "quickstart.ipynb", as_version=4)
    nbformat.validate(notebook)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "build_quickstart_notebook.py"),
            "--check",
        ],
        check=True,
    )
    sources = "\n".join(cell.source for cell in notebook.cells)
    cells_by_id = {cell.id: cell for cell in notebook.cells}

    assert "https://github.com/m2lines/Samudra.git" in sources
    assert "https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/" in sources
    assert "import samudra" in sources
    assert "configs/quickstart/train.yaml" in sources
    assert "# Samudra 2 — Colab Quickstart" in sources
    for section in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next steps"):
        assert section in sources
    assert "scripts/download_quickstart_data.py" in sources
    assert cells_by_id["checkout-code"].metadata["cellView"] == "form"
    assert len(cells_by_id["install-code"].source.splitlines()) <= 15
    assert len(cells_by_id["data-code"].source.splitlines()) <= 10
    assert 'REPO_REF = "colab-quickstart"' in sources
    assert "required_paths = [" not in sources
    assert "embedded_quickstart_files" not in sources
    assert "__QUICKSTART_BOOTSTRAP_FILES__" not in sources
    assert "requirements_fingerprint" not in sources
    assert "DOWNLOAD_FINGERPRINT" not in sources
    assert "shutil" not in sources
    assert "os.kill" not in sources
    assert "runtime.restart_session()" not in sources
    assert "--editable" not in sources
    assert 'sys.path.insert(0, str(REPO_DIR / "src"))' in sources
    assert 'source_dir = Path("/content/Samudra/src")' in sources
    assert 'if not (source_dir / "samudra").is_dir()' in sources
    assert "importlib.invalidate_caches()" in sources
    assert "subprocess.run(" in sources
    assert 'ds.sizes["time"] == 250' in sources
    legacy_display_name = "Samudra " + "v2"
    assert legacy_display_name not in sources
    assert "Samudra 2" in sources
    assert "Ocean_Emulator" not in sources
    assert "ocean_emulators" not in sources

    download_helper = (REPO_ROOT / "scripts" / "download_quickstart_data.py").read_text(
        encoding="utf-8"
    )
    assert '"--time_end",\n        "250"' in download_helper
    assert '"--write_time_chunks",\n        "10"' in download_helper
    assert '"--prognostic_vars_key",\n        "thermo_dynamic_5"' in download_helper
    assert '"--boundary_vars_key",\n        "tau_hfds"' in download_helper


@pytest.mark.parametrize(
    "data_source,config_name,extra_config_args",
    [("mock-om4", QUICKSTART_CONFIG, _MOCK_OVERRIDES)],
    indirect=True,
)
def test_quickstart_model_forward(train_config):
    """Run the production-width Samudra 2 model for one full-resolution step."""
    with MultitonScope():
        trainer = Trainer(train_config)
        trainer.init_data_loaders(cur_step=1)
        batch = trainer.train_loader[0]
        with torch.no_grad():
            prediction = trainer.model(batch)[0]

    assert prediction.shape == batch.get_label(0).shape


@pytest.mark.parametrize(
    "data_source,config_name,extra_config_args",
    [("mock-om4", QUICKSTART_CONFIG, _MOCK_OVERRIDES)],
    indirect=True,
)
def test_quickstart_training_smoke(train_config, caplog):
    """Exercise the real quickstart plumbing with a CI-sized U-Net."""
    caplog.set_level(logging.INFO)
    train_config.debug = True
    train_config.epochs = 1
    train_config.save_freq = 1

    assert isinstance(train_config.model, SamudraConfig)
    train_config.model.checkpointing = None
    train_config.model.unet.ch_width = [2, 2]
    train_config.model.unet.dilation = [1, 2]
    train_config.model.unet.n_layers = [1, 1]

    with MultitonScope():
        trainer = Trainer(train_config)
        trainer.run()
