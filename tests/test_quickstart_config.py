# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the browser-based Colab quickstart."""

import subprocess
import sys
from pathlib import Path

import nbformat

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
    for section in ("## Goal", "## Setup", "## Steps", "## Checks", "## Next steps"):
        assert section in sources

    assert "import samudra" in sources
    assert "version('samudra')" in sources
    assert '"--no-deps", "samudra"' in sources
    assert "samudra==" not in sources
    assert "samudra_om4/train.yaml" in sources
    assert "@data/om4_demo.yaml" in sources
    assert "Default model: Samudra 2" in sources
    assert "historical name `Samudra`" in sources
    assert "cfg.model.unet.ch_width == [280, 380, 480, 520]" in sources
    assert "batch_size=4" in sources
    assert 'backend="cuda"' in sources
    assert "Bias (prediction − truth)" in sources
    assert "class ColabTrainer(Trainer)" in sources
    assert 'print(f"Training epoch {epoch}/{self.epochs}")' in sources
    assert 'print(f"Validating epoch {epoch}/{self.epochs}")' in sources

    assert "s3://m2lines-pubs/Samudra/v2026-07/om4_twodeg/" in sources
    assert "source.data_location.open()" in sources
    assert "quickstart_source[location_field] = getattr(" in sources
    for location_field in (
        "data_location",
        "data_means_location",
        "data_stds_location",
    ):
        assert f'"{location_field}"' in sources
    assert 'window.sizes["time"] == 23' in sources
    assert "logical_bytes / 2**20" in sources
    assert "No local data copy was created." in sources
    assert "thermo_dynamic_5" in sources
    assert "tau_hfds" in sources
    assert "(90, 180)" in sources

    assert "checkout-code" not in cells_by_id
    assert "https://github.com/m2lines/Samudra.git" not in sources
    assert "git clone" not in sources
    assert "requirements-quickstart.txt" not in sources
    assert "download_quickstart_data.py" not in sources
    assert "/content/Samudra" not in sources
    assert "/content/data_cache" not in sources
    assert "embedded_quickstart_files" not in sources
    assert "runtime.restart_session()" not in sources
    assert "os.kill" not in sources
    assert "--editable" not in sources

    legacy_display_name = "Samudra " + "v2"
    assert legacy_display_name not in sources
    assert "Samudra 2" in sources
    assert "Ocean_Emulator" not in sources
    assert "ocean_emulators" not in sources


def test_quickstart_python_cells_compile():
    notebook = nbformat.read(REPO_ROOT / "notebooks" / "quickstart.ipynb", as_version=4)

    for cell in notebook.cells:
        if cell.cell_type != "code" or cell.id == "runtime-code":
            continue
        compile(cell.source, f"notebooks/quickstart.ipynb:{cell.id}", "exec")
