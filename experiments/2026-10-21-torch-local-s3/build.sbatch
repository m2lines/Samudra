#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
#SBATCH --job-name=s3-loader-build
#SBATCH --account=torch_pr_347_lzanna
#SBATCH --partition=cs
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --chdir=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021
#SBATCH --output=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/build-%j.out
#SBATCH --error=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/build-%j.err
set -euo pipefail
base=$PWD
export RUSTUP_HOME="$base/rustup" CARGO_HOME="$base/cargo" CARGO_TARGET_DIR="$base/target"
export UV_CACHE_DIR="$base/uv-cache" CARGO_BUILD_JOBS=8
mkdir -p "$CARGO_HOME" "$RUSTUP_HOME" "$base/wheels" "$base/extension"
if [[ ! -x "$CARGO_HOME/bin/rustc" ]]; then
  curl --fail --location --retry 3 https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init -o "$base/rustup-init"
  chmod +x "$base/rustup-init"
  "$base/rustup-init" -y --no-modify-path --profile minimal --default-toolchain 1.97.1
fi
export PATH="$CARGO_HOME/bin:$PATH"
apptainer exec /scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif bash -c '
  set -euo pipefail
  base=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021
  export PATH="$base/cargo/bin:$PATH"
  uv pip install --python /workspace/.venv/bin/python --target "$base/builddeps" maturin pytest
  export PYTHONPATH="$base/builddeps"
  cd "$base/repo"
  "$base/builddeps/bin/maturin" build --release --locked --manifest-path rust/loader/Cargo.toml --interpreter /workspace/.venv/bin/python --out "$base/wheels" --compatibility linux
  uv pip install --python /workspace/.venv/bin/python --no-deps --target "$base/extension" "$base"/wheels/*.whl
  export PYTHONPATH="$base/extension:$base/repo/src:$base/repo:$base/builddeps"
  export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
  /workspace/.venv/bin/python -m pytest -q -c rust/loader/pyproject.toml rust/loader/tests/test_extension.py
  /workspace/.venv/bin/python -m pytest -p jaxtyping._pytest_plugin -q tests/test_config.py tests/test_rust_data.py tests/test_native_loader.py -m "not cuda and not manual"
  /workspace/.venv/bin/python -c "import samudra_rust_loader; print(samudra_rust_loader.__file__)"
'
date -Is > "$base/BUILD_COMPLETE"
