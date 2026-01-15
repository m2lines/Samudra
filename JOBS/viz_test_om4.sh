#!/bin/bash
#SBATCH -p pi_abodner
#SBATCH --job-name=viz_test_om4_11-10
#SBATCH -N 1
#SBATCH --mem=100GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00-23:00:00

# load Python platform with PyTorch and CUDA support preinstalled
module load miniforge/24.3.0-0

# activate uv environment for ocean_emulator
uv sync --dev

# Remove tkinter dependency within xarrayutils
uv run python - << 'EOF'
import site, glob, os
for sp in site.getsitepackages():
    fp = os.path.join(sp, "xarrayutils", "plotting.py")
    if os.path.exists(fp):
        with open(fp) as f:
            data = f.read()
        new = data.replace("from tkinter import Y", "from matplotlib.colors import Normalize as Y")
        if data != new:
            with open(fp, "w") as f:
                f.write(new)
            print(f"Patched: {fp}")
        else:
            print(f"No patch needed: {fp}")
EOF
export MPLBACKEND=Agg   # Avoid any GUI backends

echo "======== vizualize ocean_emulator samudra outputs w/ OM4 data ========"

uv run -m ocean_emulators.viz configs/viz_om4.yaml \