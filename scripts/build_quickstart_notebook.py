"""Generate notebooks/quickstart.ipynb from inline cell sources.

Notebook JSON is painful to diff, so we author the cells here in plain Python
and let nbformat write the .ipynb. Run this any time the cell sources change:

    uv run python scripts/build_quickstart_notebook.py
"""

from pathlib import Path

import nbformat as nbf

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "notebooks" / "quickstart.ipynb"


# ---------------------------------------------------------------------------
# Cell sources
# ---------------------------------------------------------------------------

INTRO_MD = """\
# Ocean Emulator — Colab Quickstart

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Open-Athena/Ocean_Emulator/blob/main/notebooks/quickstart.ipynb)

Train a tiny Samudra ocean emulator on a slice of 1° OM4 data, on free-tier
Google Colab (T4 GPU, ~78 GB usable disk).

This is a smoke-test, not a production run. The point is to make the project
hackable from a browser — no HPC, no S3 credentials, no local environment
setup. Once this notebook works for you, you can scale up to the configs in
`configs/samudra_om4_v2/` on real hardware.

**Runtime:** ~15-25 minutes end-to-end (mostly the data download and the
training loop).

**What you'll do:**
1. Verify you have a GPU runtime
2. Install a slim subset of the project's dependencies
3. Download a ~7 GB time slice of public 1° OM4 data
4. Train a small Samudra model for a few epochs
5. Compare a model prediction to ground truth
"""

HW_CHECK_MD = """\
## 1. Hardware check

Free-tier Colab usually gives you a T4 (16 GB VRAM, Turing architecture).
Anything else is fine — A100 / L4 / V100 will all work — but if you're on
a CPU-only runtime, switch via *Runtime → Change runtime type → T4 GPU* now.
"""

HW_CHECK_PY = """\
import subprocess

import torch

print(f"torch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    raise RuntimeError(
        "No GPU detected. Switch to a GPU runtime: "
        "Runtime → Change runtime type → T4 GPU."
    )

# T4 (Turing, sm_75) does not support bfloat16; the quickstart config disables
# it. A100/L4/H100 do support it — feel free to flip use_bfloat16 back on.
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
"""

INSTALL_MD = """\
## 2. Slim install

Ocean Emulator's full dependency set includes a few packages that are heavy or
brittle on Colab (`xesmf` needs the ESMF system library; `skypilot`,
`coiled`, and `cartopy` are not needed to train Samudra). For the quickstart
we install just the packages on the training path — see
`requirements-quickstart.txt` for the curated list.

The repo is cloned into `/content/Ocean_Emulator` and added to `sys.path` so
we can import `ocean_emulators` without a full editable install.
"""

INSTALL_PY = """\
import os, sys, subprocess

REPO_DIR = "/content/Ocean_Emulator"
REPO_ZIP = "/content/Ocean_Emulator.zip"
REPO_URL = "https://github.com/Open-Athena/Ocean_Emulator.git"
MIN_PYTHON = (3, 12)

if sys.version_info < MIN_PYTHON:
    raise RuntimeError(
        "Ocean Emulator requires Python "
        f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, but this runtime is "
        f"Python {sys.version.split()[0]}. In Colab, switch to a Python 3.12 "
        "runtime, or run the quickstart locally with Python >=3.12."
    )

if os.path.isdir(REPO_DIR):
    print(f"Using existing repo at {REPO_DIR}")
elif os.path.isfile(REPO_ZIP):
    print(f"Unzipping {REPO_ZIP} → /content/")
    subprocess.run(["unzip", "-q", REPO_ZIP, "-d", "/content/"], check=True)
else:
    print(f"Cloning {REPO_URL}")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, REPO_DIR], check=True)

# Slim deps. Colab already has a working torch + CUDA build, so we let pip
# leave it alone if a compatible version is already installed.
subprocess.run(
    ["pip", "install", "-q", "-r",
     f"{REPO_DIR}/requirements-quickstart.txt"],
    check=True,
)

# Add the repo's src/ to the import path rather than `pip install -e .`, which
# would re-resolve the full pyproject dep set.
src = f"{REPO_DIR}/src"
if src not in sys.path:
    sys.path.insert(0, src)

import ocean_emulators  # noqa: F401  (smoke import)
print("ocean_emulators import OK")
"""

DATA_MD = """\
## 3. Download a data slice

`scripts/clone_data.py` pulls from the public OSN bucket
`https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/` — no credentials needed.

We slice ~290 5-day timesteps (about 4 years, 1958-01 → 1961-12). At 1°
resolution that's roughly 7 GB on disk; the quickstart config carves
training, validation, and inference windows out of this slice.

This cell is the slowest in the notebook — expect a few minutes depending on
the runtime's network throughput.
"""

DATA_PY = """\
import os
import subprocess
import shutil
from pathlib import Path

DATA_DIR = Path("/content/data_cache")
EXPECTED_STORES = ["OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr"]
DOWNLOAD_COMPLETE = DATA_DIR / ".quickstart_download_complete"

download_ready = DOWNLOAD_COMPLETE.exists() and all(
    (DATA_DIR / store).exists() for store in EXPECTED_STORES
)

if not download_ready:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_COMPLETE.unlink(missing_ok=True)
    for store in EXPECTED_STORES:
        shutil.rmtree(DATA_DIR / store, ignore_errors=True)

    # The subprocess invocation of clone_data.py does `from ocean_emulators...`,
    # so it needs the repo's src/ on PYTHONPATH (the parent kernel's sys.path
    # tweak doesn't propagate to subprocesses).
    subprocess.run(
        [
            "python",
            f"{REPO_DIR}/scripts/clone_data.py",
            str(DATA_DIR),
            "--time_start", "0",
            "--time_end",   "290",
            "--write_time_chunks", "1",
        ],
        env={**os.environ, "PYTHONPATH": f"{REPO_DIR}/src"},
        check=True,
    )
    DOWNLOAD_COMPLETE.write_text("complete\\n")

# Confirm what we got.
import xarray as xr
ds = xr.open_zarr(DATA_DIR / "OM4.zarr")
print(ds)
"""

TRAIN_MD = """\
## 4. Train

`configs/quickstart/train.yaml` is the same shape as the production v2 config
(`configs/samudra_om4_v2/`) but narrower (`ch_width: [128, 192, 256, 320]`),
single-step, batch_size=1, mse loss, fp32, 5 epochs. ~25M parameters.

We point `experiment.data_root` at the directory we just downloaded and let
the existing `Trainer` do the rest. The output (checkpoints, config snapshot,
metrics) lands under `.LOCAL/samudra_quickstart/`.
"""

TRAIN_PY = """\
import os
os.chdir(REPO_DIR)  # Trainer writes outputs relative to cwd by default.

from ocean_emulators.config import TrainConfig
from ocean_emulators.train import Trainer
from ocean_emulators.utils.multiton import MultitonScope

cfg = TrainConfig.from_yaml_and_cli(
    [
        "configs/quickstart/train.yaml",
        "--experiment.data_root", str(DATA_DIR),
    ]
)

with MultitonScope():
    trainer = Trainer(cfg)
    trainer.run()

print("\\nTraining complete.")
print(f"Checkpoints: {trainer.ckpt_paths.latest_checkpoint_path}")
"""

INFER_MD = """\
## 5. Compare a prediction to ground truth

The trainer leaves the model in-memory and a checkpoint on disk. Here we run a
single forward step on a held-out timestep and plot one prognostic channel
(sea-surface height, `zos`) side-by-side with the ground-truth target.

This is intentionally minimal — for full rollout evaluation see
`src/ocean_emulators/eval.py` and the `configs/samudra_om4_v2/eval.yaml`
config.
"""

INFER_PY = """\
import matplotlib.pyplot as plt
import torch

from ocean_emulators.constants import PROGNOSTIC_VARS

# Pull one batch from the validation loader (already moved to the model's device).
batch = next(iter(trainer.val_loader))

trainer.model.eval()
with torch.no_grad():
    pred = trainer.model(batch)[0]   # forward returns one tensor per step; we have 1 step.
label = batch.get_label(0)

# zos (sea-surface height) is the last prognostic in `thermo_dynamic_5`.
prog_names = PROGNOSTIC_VARS[cfg.experiment.prognostic_vars_key]
zos_idx = prog_names.index("zos")

pred_zos  = pred[0,  zos_idx].cpu().numpy()
label_zos = label[0, zos_idx].cpu().numpy()

# Both fields are normalized; the side-by-side comparison is still meaningful.
fig, axes = plt.subplots(1, 3, figsize=(14, 3.5))
for ax, arr, title in zip(
    axes,
    [label_zos, pred_zos, pred_zos - label_zos],
    ["Ground truth (zos, normalized)", "Prediction (zos, normalized)", "Prediction − truth"],
):
    im = ax.imshow(arr, origin="lower", cmap="RdBu_r")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.025)
plt.tight_layout()
plt.show()
"""

NEXT_MD = """\
## Where to go next

- **Scale up the model:** swap in `configs/samudra_om4_v2/{model,train}.yaml` —
  but you'll need beefier hardware than a T4 (the production config uses
  `batch_size=4`, `steps=[4]`, `ch_width=[280, 380, 480, 520]`).
- **Try a different resolution:** the half- and quarter-degree datasets live
  alongside the 1° data on the OSN bucket; point `clone_data.py`'s base URL at
  them to download.
- **Multi-scale FOMO:** see `configs/fomo_om4/`. Note FOMO needs `flash-attn`
  / `flash-perceiver` / `aurora` — the `[cuda]` extra in `pyproject.toml`.
- **Real evaluation:** `python -m ocean_emulators.eval configs/samudra_om4_v2/eval.yaml`
  runs full autoregressive rollouts and the aggregator metric stack.

If you ran into install or data issues, please open an issue against the repo —
the goal of this notebook is to keep the project's onboarding path frictionless,
and reports of what broke for you make that possible.
"""


# ---------------------------------------------------------------------------
# Build the notebook
# ---------------------------------------------------------------------------

def _markdown_cell(source: str, cell_id: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(source)
    cell["id"] = cell_id
    return cell


def _code_cell(source: str, cell_id: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(source)
    cell["id"] = cell_id
    return cell


def main() -> None:
    nb = nbf.v4.new_notebook()

    nb.cells = [
        _markdown_cell(INTRO_MD, "intro"),
        _markdown_cell(HW_CHECK_MD, "hardware-check"),
        _code_cell(HW_CHECK_PY, "hardware-check-code"),
        _markdown_cell(INSTALL_MD, "install"),
        _code_cell(INSTALL_PY, "install-code"),
        _markdown_cell(DATA_MD, "download-data"),
        _code_cell(DATA_PY, "download-data-code"),
        _markdown_cell(TRAIN_MD, "train"),
        _code_cell(TRAIN_PY, "train-code"),
        _markdown_cell(INFER_MD, "prediction-comparison"),
        _code_cell(INFER_PY, "prediction-comparison-code"),
        _markdown_cell(NEXT_MD, "next-steps"),
    ]
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        nbf.write(nb, f)
    print(f"Wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
