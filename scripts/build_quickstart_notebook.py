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

Train a Samudra 2 ocean emulator on public 2° OM4 data using a free-tier Google
Colab GPU. This is an onboarding smoke test, not a production-quality or
scientifically useful model.

No HPC system, local installation, cloud credentials, or Weights & Biases
account is required. Samudra is installed from PyPI, and the data is streamed
directly from the public S3 store instead of being copied into Colab.

**Expected runtime:** depends on the Colab GPU and public S3 throughput. There
is no multi-gigabyte download step; training reads only the requested samples.

You will:

1. Check the Colab runtime.
2. Install the latest stable Samudra release from PyPI.
3. Inspect the public 2° data stream and configure a compact run.
4. Train a Samudra 2 model.
5. Compare one prediction with ground truth.
"""

RUNTIME_MD = """\
## Setup

### 1. Check the runtime

Choose **Runtime → Change runtime type → T4 GPU** before continuing. Samudra
currently supports Python 3.12; the explicit check below gives a useful error
before any project module is imported.
"""

RUNTIME_PY = """\
import sys

SUPPORTED_PYTHON = (3, 12)
if sys.version_info[:2] != SUPPORTED_PYTHON:
    raise RuntimeError(
        "The current Samudra package supports Python 3.12, but this runtime is "
        f"Python {sys.version.split()[0]}. Select Colab's latest runtime."
    )

print(f"Python: {sys.version.split()[0]}")
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"""

INSTALL_MD = """\
### 2. Install Samudra 2

This installs the latest stable `samudra` release from PyPI—there is
intentionally no version pin. The resolved version is printed below so each run
remains diagnosable.

The wheel is installed without its full infrastructure dependency set, then
the small runtime subset used by this notebook is added explicitly. This keeps
Colab's working NumPy and CUDA-enabled PyTorch stack intact.
"""

INSTALL_PY = """\
import subprocess
import sys

RUNTIME_DEPENDENCIES = (
    "cftime>=1.6.4.post1",
    "dacite>=1.9.2",
    "dask>=2025.2",
    "einops>=0.7",
    "jaxtyping>=0.3",
    "microsoft-aurora>=1.8",
    "perceiver-pytorch>=0.8.8",
    "pydantic-settings>=2.8.1",
    "s3fs>=2025.5.1",
    "torchinfo>=1.8",
    "wandb>=0.19.8",
    "xarray>=2025.1.2",
    "xarray-einstats>=0.8",
    "zarr<3",
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
     "--no-deps", "samudra"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--quiet", *RUNTIME_DEPENDENCIES],
    check=True,
)
"""

VERIFY_INSTALL_MD = """\
Confirm that Samudra came from the installed PyPI package and PyTorch can use
the selected GPU.
"""

VERIFY_INSTALL_PY = """\
from importlib.metadata import version

import samudra
import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot see a GPU. Select Runtime → Change runtime type → T4 GPU."
    )

print(f"Samudra: {version('samudra')} (PyPI)")
print(f"Samudra import: {samudra.__file__}")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

DATA_MD = """\
## Steps

### 3. Configure and inspect the public OM4 stream

PyPI ships the production Samudra 2 model preset and a zero-credential 2° OM4
data preset. The data preset points to `OM4.zarr`, `OM4_means.zarr`, and
`OM4_stds.zarr` in the public
`s3://m2lines-pubs/Samudra/v2026-07/om4_twodeg/` directory.

The cell below starts from those bundled presets, then narrows the tutorial to
five upper-ocean levels, 16 training times, seven validation times, one-step
prediction, and five epochs. It opens the remote Zarr metadata and reads one
value as a connectivity check; it does not copy the stores to Colab.
"""

DATA_PY = """\
from samudra.config import TrainConfig

base_cfg = TrainConfig.from_yaml_and_cli(
    ["samudra_om4/train.yaml", "--data", "@data/om4_demo.yaml"]
)

quickstart = base_cfg.model_dump(mode="json")
# The Location union currently serializes structured S3 locations through its
# string-path variant. Restore the validated location objects before rebuilding
# the config so the public endpoint, bucket, and anonymous-access flag survive.
for quickstart_source, base_source in zip(
    quickstart["data"]["sources"], base_cfg.data.sources, strict=True
):
    for location_field in (
        "data_location",
        "data_means_location",
        "data_stds_location",
    ):
        quickstart_source[location_field] = getattr(base_source, location_field)

quickstart.update(
    epochs=5,
    batch_size=4,
    backend="cuda",
    save_freq=5,
    steps=[1],
    step_transition=[],
    inference_epochs=[],
    preemptible=False,
)
quickstart["experiment"].update(
    name="samudra_quickstart",
    base_output_dir="/content/samudra_outputs",
)
quickstart["data"].update(
    hist=0,
    loading={"type": "cpu", "num_workers": 0, "persistent_workers": False},
)
quickstart["data"]["sources"][0].update(
    prognostic_vars_key="thermo_dynamic_5",
    boundary_vars_key="tau_hfds",
    train_time={"start": "1975-01-03", "end": "1975-03-19"},
    val_time={"start": "1975-03-24", "end": "1975-04-23"},
    inference_times=[],
)
cfg = TrainConfig.model_validate(quickstart)

# The Python model class keeps the historical name `Samudra`; the default
# architecture selected here is Samudra 2.
assert cfg.model.unet.ch_width == [280, 380, 480, 520]
assert cfg.model.unet.core_block.upscale_factor == 2
assert cfg.model.unet.up_sampling_block == "zonally_periodic_upsample"
print("Default model: Samudra 2")
"""

DATA_CHECK_PY = """\
source = cfg.data.sources[0]
ds = source.data_location.open()
window = ds.sel(
    time=slice(source.train_time.start.datetime, source.val_time.end.datetime)
)
dataset_spec = source.dataset_spec
expected_variables = set(
    dataset_spec.prognostic_var_names + dataset_spec.boundary_var_names
)
lat_dim = "lat" if "lat" in ds.dims else "y"
lon_dim = "lon" if "lon" in ds.dims else "x"

assert (ds.sizes[lat_dim], ds.sizes[lon_dim]) == (90, 180)
assert expected_variables.issubset(ds.data_vars)
assert window.sizes["time"] == 23
logical_bytes = sum(window[name].nbytes for name in expected_variables)
probe = window["zos"].isel(time=0, **{lat_dim: 0, lon_dim: 0}).load()

print(f"Streaming from: {source.data_location}")
print(f"Grid: {ds.sizes[lat_dim]} × {ds.sizes[lon_dim]} (2°)")
print(f"Quickstart window: {window.time.values[0]} → {window.time.values[-1]}")
print(f"Selected model fields: {logical_bytes / 2**20:.1f} MiB before compression")
print(f"Connectivity probe (zos): {probe.item()}")
print("No local data copy was created.")
ds.close()
"""

TRAIN_MD = """\
### 4. Train a Samudra 2 model

The bundled `samudra_om4` preset supplies the production Samudra 2 channel
widths, ConvNeXt expansion factor, zonally periodic upsampling, and dynamic
variance-weighted loss. Training and validation still go through the normal
`Trainer`, including normalization, masking, EMA, metrics, and checkpointing.

The cell keeps Colab output concise: it prints when training and validation
start for each epoch, while warnings and errors remain visible.
"""

TRAIN_PY = """\
import logging

from samudra.train import Trainer
from samudra.utils.multiton import MultitonScope

samudra_logger = logging.getLogger("samudra")
for handler in tuple(samudra_logger.handlers):
    if handler.get_name() == "samudra-colab-progress":
        samudra_logger.removeHandler(handler)
        handler.close()
samudra_logger.setLevel(logging.WARNING)


class ColabTrainer(Trainer):
    def train_one_epoch(self, epoch):
        print(f"Training epoch {epoch}/{self.epochs}")
        return super().train_one_epoch(epoch)

    def validate_one_epoch(self, epoch):
        print(f"Validating epoch {epoch}/{self.epochs}")
        return super().validate_one_epoch(epoch)


epoch_word = "epoch" if cfg.epochs == 1 else "epochs"
print(f"Preparing to train for {cfg.epochs} {epoch_word}.")
with MultitonScope():
    trainer = ColabTrainer(cfg)
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

prognostic_names = cfg.data.sources[0].dataset_spec.prognostic_var_names
zos_index = prognostic_names.index("zos")
predicted_zos = prediction[0, zos_index].cpu().numpy()
target_zos = target[0, zos_index].cpu().numpy()

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, field, title in zip(
    axes,
    [target_zos, predicted_zos, predicted_zos - target_zos],
    ["Ground truth", "Prediction", "Bias (prediction − truth)"],
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

- Use `samudra train samudra_om4/train.yaml --data @data/om4_demo.yaml` for the
  full-depth bundled demo.
- Use `samudra eval ...` for long autoregressive rollouts.
- Explore the bundled `samudra_multi_om4` preset for multi-resolution training.
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
        _code_cell(INSTALL_PY, "install-code"),
        _markdown_cell(VERIFY_INSTALL_MD, "verify-install"),
        _code_cell(VERIFY_INSTALL_PY, "verify-install-code"),
        _markdown_cell(DATA_MD, "data"),
        _code_cell(DATA_PY, "data-code"),
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
