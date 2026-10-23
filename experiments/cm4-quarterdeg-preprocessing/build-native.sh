#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
export ROOT=/scratch/jr7309/cm4-quarterdeg-20261023
export RUSTUP_HOME=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/rustup
export CARGO_HOME=/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/cargo
export CARGO_TARGET_DIR="$ROOT/native-target" CARGO_BUILD_JOBS=8
export UV_CACHE_DIR="$ROOT/uv-cache"
apptainer exec /scratch/jr7309/.apptainer-images/ghcr.io-m2lines-ocean-emulator-physicsnemo-26.05-9cd36b1bdcf4921fb027494afa4157e217818ccd.sif bash -c '
set -euo pipefail
export PATH="$CARGO_HOME/bin:$PATH"
cd "$ROOT/source"
/scratch/jr7309/benchmarks/quarterdeg-local-s3-20261021/builddeps/bin/maturin build --release --locked --manifest-path rust/loader/Cargo.toml --interpreter /workspace/.venv/bin/python --out "$ROOT/wheels" --compatibility linux
uv pip install --python /workspace/.venv/bin/python --no-deps --target "$ROOT/native" "$ROOT"/wheels/*.whl
'
