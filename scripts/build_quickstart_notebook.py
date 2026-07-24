# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Generate ``notebooks/quickstart.ipynb`` from reviewable cell sources.

Run this script whenever a quickstart cell changes:

    uv run python scripts/build_quickstart_notebook.py
"""

import argparse
from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "quickstart.ipynb"


INTRO_MD = """\
# Samudra 2 — Colab Quickstart

## Goal

Train a Samudra 2 ocean emulator on a public slice of 1° OM4 data using a
free-tier Google Colab GPU. This is an onboarding smoke test, not a
production-quality or scientifically useful model.

No HPC system, local installation, cloud credentials, or Weights & Biases
account is required.

**Expected runtime:** approximately 20–40 minutes, depending mostly on download
and GPU speed. The filtered data slice uses approximately 2–2.5 GB of disk.

You will:

1. Check the Colab runtime.
2. Install the training-only dependencies.
3. Download three and a half years of public 1° OM4 data.
4. Train a Samudra 2 model.
5. Compare one prediction with ground truth.
"""

RUNTIME_MD = """\
## Setup

### 1. Check the runtime

Choose **Runtime → Change runtime type → T4 GPU** before continuing. Samudra
requires Python 3.12 or newer; the explicit check below gives a useful error
before any project module is imported.
"""

RUNTIME_PY = """\
import sys

MIN_PYTHON = (3, 12)
if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        "Samudra requires Python 3.12+, but this runtime is "
        f"Python {sys.version.split()[0]}. Select Colab's latest runtime "
        "version before running this notebook."
    )

print(f"Python: {sys.version.split()[0]}")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""

INSTALL_MD = """\
### 2. Install Samudra 2

The repository is public, so the notebook clones it directly from
`m2lines/Samudra`. The slim requirements file avoids preprocessing and cloud
orchestration packages that this example does not use. The revision field is
shown as a compact Colab form; keep its default value when running this version
of the notebook.
"""

CHECKOUT_PY = """\
import subprocess
from pathlib import Path

REPO_DIR = Path("/content/Samudra")
REPO_REF = "colab-quickstart"  # @param {"type":"string"}

if not (REPO_DIR / ".git").is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_REF,
         "https://github.com/m2lines/Samudra.git", str(REPO_DIR)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "fetch", "--depth", "1", "origin", REPO_REF],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(REPO_DIR), "checkout", "--quiet", "--detach", "FETCH_HEAD"],
        check=True,
    )
"""

INSTALL_DEPENDENCIES_MD = """\
Install the dependencies and expose the repository's `src` directory to this
kernel. A failed install stops the cell immediately.
"""

INSTALL_PY = """\
import sys

%cd {REPO_DIR}
requirements_path = REPO_DIR / "requirements-quickstart.txt"
if not requirements_path.is_file():
    raise FileNotFoundError(
        f"{REPO_REF!r} does not contain {requirements_path.name}."
    )
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--requirement",
     str(requirements_path)],
    check=True,
)
sys.path.insert(0, str(REPO_DIR / "src"))
"""

VERIFY_INSTALL_MD = """\
Confirm that the Samudra 2 source tree is importable and PyTorch can use the
selected GPU. This check restores the source path explicitly, so it does not
depend on editable-install state from an earlier cell.
"""

VERIFY_INSTALL_PY = """\
import importlib
import sys
from pathlib import Path

source_dir = Path("/content/Samudra/src")
if not (source_dir / "samudra").is_dir():
    raise FileNotFoundError(
        "Samudra source was not found. Rerun the repository checkout cell."
    )
source_path = str(source_dir)
if source_path not in sys.path:
    sys.path.insert(0, source_path)
importlib.invalidate_caches()

import samudra
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot see a GPU. Select Runtime → Change runtime type → T4 GPU."
    )

print(f"Samudra import: {samudra.__file__}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

DATA_MD = """\
## Steps

### 3. Download a public OM4 slice

The repository's `clone_data.py` script defaults to the public, unfiltered 1°
OM4 release at
`https://nyu1.osn.mghpcc.org/m2lines-pubs/FOMO/v2025-11/om4_onedeg/`.
No S3 credentials are needed.

The following cell downloads indices 0–249: the minimum five-day window needed
to cover training and validation from January 1958 through June 1961. Before
transferring chunks, it selects only the five upper-ocean levels and variables
used by this tutorial. Ten-time-step output chunks reduce filesystem overhead.
The helper reuses a completed download and safely cleans up partial data before
a retry.
"""

DATA_PY = """\
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("/content/data_cache")
subprocess.run(
    [sys.executable, "scripts/download_quickstart_data.py", str(DATA_DIR)],
    check=True,
)
"""

DATA_CHECK_MD = """\
Verify that the download has the expected time range and variables:
"""

DATA_CHECK_PY = """\
import xarray as xr

from samudra.constants import build_om4_spec

ds = xr.open_zarr(DATA_DIR / "OM4.zarr")
dataset_spec = build_om4_spec("thermo_dynamic_5", "tau_hfds")
expected_variables = set(
    dataset_spec.prognostic_var_names + dataset_spec.boundary_var_names
)

assert ds.sizes["time"] == 250
assert expected_variables.issubset(ds.data_vars)
print(f"Downloaded {ds.sizes['time']} time steps and {len(ds.data_vars)} variables.")
print(f"Time range: {ds.time.values[0]} → {ds.time.values[-1]}")
"""

TRAIN_MD = """\
### 4. Train a Samudra 2 model

`configs/quickstart/train.yaml` uses the production Samudra 2 channel widths,
ConvNeXt expansion factor, zonally periodic upsampling, and dynamic
variance-weighted loss. To keep the tutorial practical, it uses five depth
levels rather than nineteen, one-step training, `batch_size=1`, and five epochs.
Training and validation still go through the normal `Trainer`, including
normalization, masking, EMA, metrics, and checkpointing.
"""

TRAIN_PY = """\
from samudra.config import TrainConfig
from samudra.train import Trainer
from samudra.utils.multiton import MultitonScope

cfg = TrainConfig.from_yaml_and_cli(
    [
        "configs/quickstart/train.yaml",
        "--experiment.data_root",
        str(DATA_DIR),
    ]
)

with MultitonScope():
    trainer = Trainer(cfg)
    trainer.run()

print("Training complete.")
print(f"Latest checkpoint: {trainer.ckpt_paths.latest_checkpoint_path}")
"""

PREDICTION_MD = """\
## Checks

### 5. Compare a prediction with ground truth

The final cell uses the last-epoch model on one held-out validation batch and
plots normalized sea surface height (`zos`). It is deliberately small; the
regular evaluation CLI runs long autoregressive rollouts and writes full
prediction datasets.
"""

PREDICTION_PY = """\
import matplotlib.pyplot as plt
import torch

batch = next(iter(trainer.val_loader))
trainer.model.eval()
with torch.no_grad():
    prediction = trainer.model(batch)[0]
target = batch.get_label(0)

prognostic_names = cfg.data.dataset.build().prognostic_var_names
zos_index = prognostic_names.index("zos")
predicted_zos = prediction[0, zos_index].cpu().numpy()
target_zos = target[0, zos_index].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, field, title in zip(
    axes,
    [target_zos, predicted_zos, predicted_zos - target_zos],
    ["Ground truth", "Prediction", "Prediction − truth"],
    strict=True,
):
    image = ax.imshow(field, origin="lower", cmap="RdBu_r")
    ax.set_title(f"{title} (normalized zos)")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(image, ax=ax, fraction=0.025)

plt.tight_layout()
plt.show()
"""

NEXT_MD = """\
## Next steps

- Use `configs/samudra_om4_v2/` for full-depth, multi-step Samudra 2 training.
- Use `python -m samudra.eval` for long autoregressive rollouts.
- Explore `configs/samudra_multi_om4/` for multi-resolution training.
- Read the [Samudra documentation](https://m2lines.github.io/Samudra/docs/).

If a setup or data step fails, please open a GitHub issue and include the
failing cell's output.
"""


def _markdown_cell(source: str, cell_id: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(source)
    cell["id"] = cell_id
    return cell


def _code_cell(
    source: str, cell_id: str, *, colab_form: bool = False
) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(source)
    cell["id"] = cell_id
    if colab_form:
        cell["metadata"]["cellView"] = "form"
    return cell


def build_notebook() -> nbf.NotebookNode:
    """Build and validate the quickstart notebook in memory."""
    notebook = nbf.v4.new_notebook()
    notebook.cells = [
        _markdown_cell(INTRO_MD, "intro"),
        _markdown_cell(RUNTIME_MD, "runtime"),
        _code_cell(RUNTIME_PY, "runtime-code"),
        _markdown_cell(INSTALL_MD, "install"),
        _code_cell(CHECKOUT_PY, "checkout-code", colab_form=True),
        _markdown_cell(INSTALL_DEPENDENCIES_MD, "install-dependencies"),
        _code_cell(INSTALL_PY, "install-code"),
        _markdown_cell(VERIFY_INSTALL_MD, "verify-install"),
        _code_cell(VERIFY_INSTALL_PY, "verify-install-code"),
        _markdown_cell(DATA_MD, "data"),
        _code_cell(DATA_PY, "data-code"),
        _markdown_cell(DATA_CHECK_MD, "data-check"),
        _code_cell(DATA_CHECK_PY, "data-check-code"),
        _markdown_cell(TRAIN_MD, "train"),
        _code_cell(TRAIN_PY, "train-code"),
        _markdown_cell(PREDICTION_MD, "prediction"),
        _code_cell(PREDICTION_PY, "prediction-code"),
        _markdown_cell(NEXT_MD, "next"),
    ]
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    }
    nbf.validate(notebook)
    return notebook


def main(*, check: bool = False) -> None:
    """Write the generated notebook, or check that the committed copy is current."""
    notebook = build_notebook()
    if check:
        if not OUT.exists() or nbf.read(OUT, as_version=4) != notebook:
            raise SystemExit(
                "notebooks/quickstart.ipynb is stale; run "
                "`uv run python scripts/build_quickstart_notebook.py`"
            )
        print(f"{OUT.relative_to(REPO)} is current")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as output:
        nbf.write(notebook, output)
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in notebook differs from the generated notebook",
    )
    main(check=parser.parse_args().check)
