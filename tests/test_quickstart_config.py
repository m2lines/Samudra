# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the browser-based Colab quickstart."""

import subprocess
import sys
from pathlib import Path

import nbformat

from samudra.config import SamudraConfig, TrainConfig
from samudra.utils.location import S3Location
from scripts.build_quickstart_notebook import render_quickstart_yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


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

    assert "# Samudra 2 — Colab Quickstart" in sources
    for section in (
        "## Goal",
        "## Setup",
        "## Steps",
        "## Checks",
        "## Next steps",
    ):
        assert section in sources
    assert "Scientific question" not in sources
    intro_source = cells_by_id["intro"].source
    assert "The workflow runs entirely in Colab" in intro_source
    assert "By the end of this notebook, you will be able to:" in intro_source
    assert "interpret its bias" in intro_source
    assert "No HPC system" not in intro_source
    assert "There is no bulk download" not in intro_source

    install_source = cells_by_id["install-code"].source
    assert '"samudra[cuda]"' in install_source
    assert "--progress-bar off" in install_source
    assert "--no-deps" not in install_source
    assert install_source.count("!pip install") == 1
    assert "from importlib" not in install_source
    assert "RuntimeError" not in install_source
    assert "restart" not in install_source
    restart_source = cells_by_id["restart-code"].source
    assert "loaded_numpy_version != installed_numpy_version" in restart_source
    assert "os.kill(os.getpid(), signal.SIGKILL)" in restart_source
    assert "Runtime → Run all once more" in restart_source
    assert "torch.cuda.is_available()" in restart_source
    assert "torch.backends.cuda.is_flash_attention_available()" in restart_source
    assert "runtime.restart_session()" not in sources
    assert "subprocess" not in sources
    assert "samudra==" not in sources
    assert "TrainConfig.from_yaml_and_cli([str(CONFIG_PATH)])" in sources
    assert "%%writefile /content/samudra_quickstart.yaml" in sources
    assert "TRAIN_YAML_TEMPLATE" not in sources
    assert "dilation: [1, 2, 4, 8]" in sources
    assert "EXPERIMENTS" not in sources
    assert "epochs: 1" in sources
    assert "batch_size: 8" in sources
    assert "two full batches per epoch" in sources
    assert "Samudra presets commonly use" in sources
    assert "70 epochs" in sources
    assert "Autoregressive forecast steps" in sources
    assert "steps: [1, 4]" in sources
    assert "Number of additional past ocean states" in sources
    assert "Channels per stage control model capacity" in sources
    assert "broader spatial context" in sources
    assert "backend: cuda" in sources
    assert "Bias (prediction − truth)" in sources
    assert "class ColabTrainer(Trainer)" in sources
    assert "from tqdm.auto import tqdm" in sources
    assert "Epoch progress" not in sources
    assert "self.epoch_progress" not in sources
    assert 'desc=f"Training epoch {epoch}/{self.epochs}"' in sources
    assert 'desc=f"Validating epoch {epoch}/{self.epochs}"' in sources
    assert "progress.set_postfix" in sources
    assert "leave=True" in sources
    assert "with tqdm(" in sources
    assert "progress.close()" not in sources
    assert "### Training complete" in sources
    assert "Latest checkpoint" in sources
    assert "good first issues" in sources

    assert "s3://m2lines-pubs/Samudra/v2026-07/om4_twodeg/" in sources
    assert "data/om4_demo.yaml" in sources
    assert "source.data_location.open()" in sources
    for location_field in (
        "data_location",
        "data_means_location",
        "data_stds_location",
    ):
        assert f"{location_field}:" in sources
    assert 'train_window.sizes["time"] == 17' in sources
    assert 'val_window.sizes["time"] == 7' in sources
    assert "Requested chunks streamed directly from public storage." in sources
    assert "thermo_dynamic_5" in sources
    assert "tau_hfds" in sources
    assert "(90, 180)" in sources
    assert "model_dump" not in sources

    assert "checkout-code" not in cells_by_id
    assert "https://github.com/m2lines/Samudra.git" not in sources
    assert "git clone" not in sources
    assert "requirements-quickstart.txt" not in sources
    assert "download_quickstart_data.py" not in sources
    assert "/content/Samudra" not in sources
    assert "/content/data_cache" not in sources
    assert "embedded_quickstart_files" not in sources
    assert "verify-install-code" not in cells_by_id
    assert "--editable" not in sources

    legacy_display_name = "Samudra " + "v2"
    assert legacy_display_name not in sources
    assert "Samudra 2" in sources
    assert "Ocean_Emulator" not in sources
    assert "ocean_emulators" not in sources


def test_quickstart_python_cells_compile():
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "quickstart.ipynb", as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.id in {
            "runtime-code",
            "install-code",
            "config-yaml",
        }:
            continue
        compile(cell.source, f"notebooks/quickstart.ipynb:{cell.id}", "exec")


def test_quickstart_yaml_config_validates(tmp_path):
    config_path = tmp_path / "train.yaml"
    quickstart_yaml = render_quickstart_yaml()
    assert "static_data_vars:" not in quickstart_yaml
    assert "corrector:" not in quickstart_yaml
    config_path.write_text(quickstart_yaml)
    cfg = TrainConfig.from_yaml_and_cli([str(config_path)])

    assert cfg.experiment.name == "samudra_quickstart"
    assert cfg.epochs == 1
    assert cfg.batch_size == 8
    assert cfg.backend == "cuda"
    assert isinstance(cfg.model, SamudraConfig)
    assert cfg.model.unet.dilation == [1, 2, 4, 8]
    assert cfg.model.unet.ch_width == [280, 380, 480, 520]
    assert cfg.data.sources[0].prognostic_vars_key == "thermo_dynamic_5"
    data_location = cfg.data.sources[0].data_location
    assert isinstance(data_location, S3Location)
    assert data_location.anon is True
