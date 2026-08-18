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

Configure, train, validate, and visualize a Samudra 2 ocean emulator on public
2° OM4 data using a free-tier Google Colab GPU. The notebook demonstrates the
complete YAML-driven workflow and leaves you with an editable configuration
and checkpoint for follow-up experiments.

Samudra installs from PyPI and streams the selected samples directly from
public S3 storage. The workflow runs entirely in Colab, with experiment
controls visible in YAML and local checkpoint output ready for follow-up work.

**Expected runtime:** depends on the Colab GPU and public S3 throughput.
Training and validation use short, explicit date windows and stream only the
requested chunks.

By the end of this notebook, you will be able to:

1. Read and edit the YAML controls for a Samudra training run.
2. Select and inspect public 2° OM4 data without downloading a local copy.
3. Train and validate Samudra 2, then locate the saved checkpoint.
4. Compare a held-out prediction with ground truth and interpret its bias.
5. Identify the controls to extend the run with more data, epochs, or variables.
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

This tracks and installs the latest stable `samudra` release from PyPI.

Samudra's Perceiver uses PyTorch's native scaled dot product attention, so it
does not need separately compiled `flash-attn`, `flash-perceiver`, or
`perceiver-pytorch` packages. One resolver invocation can therefore install
Samudra and the bounded notebook runtime. Pip retains Colab's CUDA-enabled
PyTorch when it satisfies Samudra's supported version range; the following cell
verifies CUDA and PyTorch's compiled FlashAttention support before training.

Samudra declares NumPy `>=1.26.4,<2` as its supported range. The explicit
constraints also select Matplotlib `>=3.10.1` and a matching `s3fs`/`fsspec`
pair. Colab includes unrelated `datasets` and `gcsfs` packages that pin an older
`fsspec`; `--no-warn-conflicts` keeps setup output focused on this notebook's
runtime.

Colab may have already loaded NumPy 2 before this installation replaces it.
The following cell detects that one-time mismatch and restarts the Python
kernel automatically. After Colab reconnects, click **Runtime → Run all** once
more to continue automatically through the remaining cells.
"""

INSTALL_PY = """\
!pip install --quiet --upgrade --no-warn-conflicts --progress-bar off \\
    "numpy>=1.26.4,<2" "matplotlib>=3.10.1" \\
    "cftime>=1.6.4.post1" "dacite>=1.9.2" "dask>=2025.2,<2026" "einops>=0.7" \\
    "jaxtyping>=0.3" "microsoft-aurora>=1.8" \\
    "pydantic-settings>=2.8.1" "pyyaml>=6.0.2" "s3fs==2025.5.1" \\
    "torchinfo>=1.8" "tqdm>=4.67.1" "typing-extensions>=4.15" \\
    "wandb>=0.19.8" "xarray>=2025.1.2" \\
    "xarray-einstats>=0.8" "zarr<3" samudra
"""

RESTART_PY = """\
from importlib.metadata import version
import os
import signal
import sys
import time

from packaging.version import Version

installed_numpy_version = Version(version("numpy"))
loaded_numpy = sys.modules.get("numpy")
loaded_numpy_version = (
    Version(loaded_numpy.__version__) if loaded_numpy is not None else None
)

if (
    loaded_numpy_version is not None
    and loaded_numpy_version != installed_numpy_version
):
    print(
        "Setup installed NumPy "
        f"{installed_numpy_version}, replacing the preloaded "
        f"{loaded_numpy_version}. Restarting the Python kernel now. "
        "After Colab reconnects, click Runtime → Run all once more."
    )
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGKILL)
else:
    print(f"Python environment ready (NumPy {installed_numpy_version}).")

import torch

if not torch.cuda.is_available():
    raise RuntimeError(
        "The installed PyTorch cannot access CUDA. Select a Colab GPU runtime "
        "and rerun setup."
    )

print(f"PyTorch: {torch.__version__} (CUDA {torch.version.cuda})")
print(
    "PyTorch FlashAttention compiled: "
    f"{torch.backends.cuda.is_flash_attention_available()}"
)
"""

CONFIG_MD = """\
## Steps

### 3. Write the YAML configuration

Samudra is configured primarily through YAML and secondarily through CLI
overrides. The first cell below uses Colab's `%%writefile` magic to create a
complete, commented YAML file that you can edit directly. The next cell loads
that file with `TrainConfig.from_yaml_and_cli()` and summarizes what will run.
Useful controls are kept visible:

- Change `epochs` or the date ranges to run a longer or shorter tutorial.
- Change `prognostic_vars_key` to include more or fewer ocean levels.
- Change `batch_size` if the selected Colab GPU has more or less memory.
- Change model fields to prototype another Samudra 2 architecture.
"""


QUICKSTART_YAML = """\
# A compact Samudra 2 training run for a free-tier Colab GPU.
# Every training, data, and model control is visible and editable.
debug: false
disk_mode: true
pin_mem: true
save_freq: 1
# One epoch keeps this instructional run short. Samudra presets commonly use
# 70 epochs; also extend the date windows for a substantive experiment.
epochs: 1
# Batch 8 keeps this quickstart comfortable on a free-tier Colab GPU.
batch_size: 8
learning_rate: 0.0006
gradient_accumulation_steps: 1
scheduler: {type: cosine}
# Evaluate the directly trained weights after this one-epoch quickstart.
test_using_ema: false
loss:
  type: dynamic
  metric: mse
finetune: false
resume_ckpt_path: null
inference_epochs: []
# Temporal spacing between input and target frames; 1 uses consecutive frames.
data_stride: [1]
# Autoregressive forecast steps used at each training-curriculum stage.
steps: [1]
# Epochs at which training switches to the next value in `steps`.
# For example: steps: [1, 4] with step_transition: [20].
step_transition: []
preemptible: false
backend: cuda

experiment:
  name: samudra_quickstart
  rand_seed: 15
  base_output_dir: /content/samudra_outputs
  wandb:
    mode: disabled
    project: samudra_quickstart

data:
  # Number of additional past ocean states supplied with the current state.
  # Zero means that the model sees only the current timestep.
  hist: 0
  concurrent_compute: true
  loading:
    type: cpu
    num_workers: 0
    persistent_workers: false
  sources:
    - type: om4
      # This is the bundled data/om4_demo.yaml source narrowed for the quickstart.
      # Five upper-ocean levels keep the free-tier run compact.
      prognostic_vars_key: thermo_dynamic_5
      boundary_vars_key: tau_hfds
      train_time:
        start: "1975-01-03"
        # 17 frames produce 16 one-step samples: two full batches per epoch.
        end: "1975-03-24"
      val_time:
        start: "1975-03-29"
        end: "1975-04-28"
      inference_times: []
      # Public, anonymous 2° OM4 stores on OSN.
      data_location:
        type: s3
        endpoint_url: "https://nyu1.osn.mghpcc.org"
        anon: true
        bucket: m2lines-pubs
        path: Samudra/v2026-07/om4_twodeg/OM4.zarr
      data_means_location:
        type: s3
        endpoint_url: "https://nyu1.osn.mghpcc.org"
        anon: true
        bucket: m2lines-pubs
        path: Samudra/v2026-07/om4_twodeg/OM4_means.zarr
      data_stds_location:
        type: s3
        endpoint_url: "https://nyu1.osn.mghpcc.org"
        anon: true
        bucket: m2lines-pubs
        path: Samudra/v2026-07/om4_twodeg/OM4_stds.zarr

model:
  # Recompute U-Net layers during backpropagation to reduce GPU memory use.
  checkpointing: all
  pred_residuals: false
  last_kernel_size: 3
  # Longitude is periodic, so circular padding avoids a seam at 0°/360°.
  pad: circular
  unet:
    # Each list entry describes one U-Net resolution stage.
    # Channels per stage control model capacity.
    ch_width: [280, 380, 480, 520]
    # Wider sampling gaps give deeper stages broader spatial context.
    dilation: [1, 2, 4, 8]
    # ConvNeXt blocks per stage control depth.
    n_layers: [1, 1, 1, 1]
    core_block:
      block_type: conv_next_block
      kernel_size: 3
      activation: capped_gelu
      upscale_factor: 2
      norm: batch
    # Average pooling reduces the spatial grid between encoder stages.
    down_sampling_block: avg_pool
    # Periodic upsampling reconstructs the grid without a longitude seam.
    up_sampling_block: zonally_periodic_upsample
"""


def render_quickstart_yaml() -> str:
    """Return the self-contained YAML config used by the generated notebook."""
    return QUICKSTART_YAML


CONFIG_YAML_CELL = "%%writefile /content/samudra_quickstart.yaml\n" + QUICKSTART_YAML


CONFIG_LOAD_PY = """\
from pathlib import Path

from IPython.display import Markdown, display
from samudra.config import TrainConfig

CONFIG_PATH = Path("/content/samudra_quickstart.yaml")
cfg = TrainConfig.from_yaml_and_cli([str(CONFIG_PATH)])

display(Markdown(
    "#### Loaded training configuration\\n\\n"
    "| Config path | Epochs | Batch size | Backend |\\n"
    "|---|---:|---:|---|\\n"
    f"| `{CONFIG_PATH}` | {cfg.epochs} | {cfg.batch_size} | `{cfg.backend}` |"
))
"""


DATA_MD = """\
### 4. Inspect the selected public OM4 data

The YAML points to `OM4.zarr`, `OM4_means.zarr`, and `OM4_stds.zarr` in the
public `s3://m2lines-pubs/Samudra/v2026-07/om4_twodeg/` directory. The cell
opens remote metadata, selects the exact train and validation windows, and
loads one sea-surface-height value as a connectivity check. Only the selected
chunks stream into Colab.

These locations match Samudra's bundled
[`data/om4_demo.yaml`](https://github.com/m2lines/Samudra/blob/main/src/samudra/configs/data/om4_demo.yaml).
They remain explicit here because the quickstart selects shorter date ranges
and five upper-ocean levels.

The displayed table connects each observed value to the YAML control that
changes it, so you can turn this example into a new experiment.
"""

DATA_CHECK_PY = """\
source = cfg.data.sources[0]
ds = source.data_location.open()
train_window = ds.sel(time=source.train_time.time_slice)
val_window = ds.sel(time=source.val_time.time_slice)
dataset_spec = source.dataset_spec
expected_variables = set(
    dataset_spec.prognostic_var_names + dataset_spec.boundary_var_names
)
lat_dim = "lat" if "lat" in ds.dims else "y"
lon_dim = "lon" if "lon" in ds.dims else "x"

assert (ds.sizes[lat_dim], ds.sizes[lon_dim]) == (90, 180)
assert expected_variables.issubset(ds.data_vars)
assert train_window.sizes["time"] == 17
assert val_window.sizes["time"] == 7
probe = train_window["zos"].isel(time=0, **{lat_dim: 0, lon_dim: 0}).load()

display(Markdown(
    "| Observed experiment setting | Selected value | YAML control |\\n"
    "|---|---:|---|\\n"
    f"| Grid | {ds.sizes[lat_dim]} × {ds.sizes[lon_dim]} (2°) | "
    "`data_location` |\\n"
    f"| Training time frames | {train_window.sizes['time']} | "
    "`train_time` |\\n"
    f"| Validation time frames | {val_window.sizes['time']} | "
    "`val_time` |\\n"
    f"| Prognostic fields | {len(dataset_spec.prognostic_var_names)} | "
    "`prognostic_vars_key` |\\n"
    f"| Batch size | {cfg.batch_size} | `batch_size` |"
))
print(f"Anonymous stream: {source.data_location}")
print(f"Connectivity probe — first source zos value: {probe.item()}")
print("Requested chunks streamed directly from public storage.")
ds.close()
"""

TRAIN_MD = """\
### 5. Train and validate Samudra 2

The normal `Trainer` handles normalization, masking, dynamic
variance-weighted loss, validation, and checkpointing. The small wrapper below
adds one `tqdm` bar for the training batches and another for the validation
batches. Each completed bar shows its final loss and elapsed time while keeping
the output focused on progress, warnings, and errors.

Warnings and errors remain visible. The final table presents the run settings,
losses, and saved checkpoint separately.
"""

TRAIN_PY = """\
import logging

from IPython.display import Markdown, display
from samudra.train import Trainer
from samudra.utils.multiton import MultitonScope
from tqdm.auto import tqdm

samudra_logger = logging.getLogger("samudra")
for handler in tuple(samudra_logger.handlers):
    if handler.get_name() == "samudra-colab-progress":
        samudra_logger.removeHandler(handler)
        handler.close()
samudra_logger.setLevel(logging.WARNING)


class ColabTrainer(Trainer):
    def train_one_epoch(self, epoch):
        original_loader = self.train_loader
        try:
            with tqdm(
                original_loader,
                total=len(original_loader),
                desc=f"Training epoch {epoch}/{self.epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            ) as progress:
                self.train_loader = progress
                stats = super().train_one_epoch(epoch)
                self.quickstart_train_loss = float(stats["train/mean/loss"])
                progress.set_postfix(loss=f"{self.quickstart_train_loss:.4f}")
                return stats
        finally:
            self.train_loader = original_loader

    def validate_one_epoch(self, epoch):
        original_loader = self.val_loader
        try:
            with tqdm(
                original_loader,
                total=len(original_loader),
                desc=f"Validating epoch {epoch}/{self.epochs}",
                unit="batch",
                dynamic_ncols=True,
                leave=True,
            ) as progress:
                self.val_loader = progress
                stats = super().validate_one_epoch(epoch)
                self.quickstart_val_loss = float(stats["val/mean/loss"])
                progress.set_postfix(loss=f"{self.quickstart_val_loss:.4f}")
                return stats
        finally:
            self.val_loader = original_loader


with MultitonScope():
    trainer = ColabTrainer(cfg)
    trainer.run()

display(Markdown(
    "### Training complete\\n\\n"
    "| Epochs | Batch size | Train loss | Validation loss |\\n"
    "|---:|---:|---:|---:|\\n"
    f"| {cfg.epochs} | {cfg.batch_size} | "
    f"{trainer.quickstart_train_loss:.4f} | "
    f"{trainer.quickstart_val_loss:.4f} |\\n\\n"
    "#### Latest checkpoint\\n\\n"
    f"`{trainer.ckpt_paths.latest_checkpoint_path}`"
))
"""

PREDICTION_MD = """\
## Checks

### 6. Compare one prediction with ground truth

The final cell runs the trained model on one held-out validation batch and
plots normalized sea-surface height (`zos`). Ground truth, prediction, and
prediction-minus-truth bias share consistent color scales for direct visual
comparison.
"""

PREDICTION_PY = """\
import matplotlib.pyplot as plt
import numpy as np
import torch

batch = next(iter(trainer.val_loader))
trainer.model.eval()
with torch.no_grad():
    prediction = trainer.model(batch)[0]
target = batch.get_label(0)

prognostic_names = cfg.data.sources[0].dataset_spec.prognostic_var_names
zos_index = prognostic_names.index("zos")
zos_mask = trainer.primary_src.masks.prognostic[zos_index].cpu().numpy()
predicted_zos = np.where(
    zos_mask, prediction[0, zos_index].detach().cpu().numpy(), np.nan
)
target_zos = np.where(
    zos_mask, target[0, zos_index].detach().cpu().numpy(), np.nan
)
bias_zos = predicted_zos - target_zos

state_limit = max(
    float(np.nanmax(np.abs(target_zos))),
    float(np.nanmax(np.abs(predicted_zos))),
)
bias_limit = float(np.nanmax(np.abs(bias_zos)))

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), constrained_layout=True)
fields = [target_zos, predicted_zos, bias_zos]
titles = ["Ground truth", "Prediction", "Bias (prediction − truth)"]
for column, (field, title) in enumerate(zip(fields, titles, strict=True)):
    limit = bias_limit if column == 2 else state_limit
    image = axes[column].imshow(
        field,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )
    axes[column].set_title(f"{title}\\n(normalized zos)")
    axes[column].set_xticks([])
    axes[column].set_yticks([])
    fig.colorbar(image, ax=axes[column], fraction=0.025)

plt.show()
"""

NEXT_MD = """\
## Next steps

- Edit the generated YAML to train for more epochs, use a longer time window,
  or include more ocean levels.
- Use `samudra train samudra_om4/train.yaml --data @data/om4_demo.yaml` for
  the full-depth bundled demo.
- Use `samudra eval ...` for long autoregressive rollouts and physical-space
  metrics.
- Explore the bundled `samudra_multi_om4` preset for multi-resolution
  training.
- Browse [good first issues](https://github.com/m2lines/Samudra/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22)
  for approachable contributions.
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
        _code_cell(RESTART_PY, "restart-code"),
        _markdown_cell(CONFIG_MD, "config"),
        _code_cell(CONFIG_YAML_CELL, "config-yaml"),
        _code_cell(CONFIG_LOAD_PY, "config-load-code"),
        _markdown_cell(DATA_MD, "data"),
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
