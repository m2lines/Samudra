#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

# Run inside the pinned ARM image with SOURCE read-only and BUILD writable.
# Only BUILD receives tooling, wheels, extension, caches, and audit artifacts.
# SOURCE is a reviewed git archive including .cargo/, rust-toolchain.toml,
# rust/crab_load/ and scripts/audit_quarter_compact.py. Never use a live checkout.
set -euo pipefail
source_root=${1:?Pass source archive directory}
build_root=${2:?Pass fresh runtime-build directory}
store=${3:?Pass averaged compact OM4.zarr directory}
python=/workspace/.venv/bin/python
test "$(uname -m)" = aarch64
test -f "$source_root/rust/crab_load/Cargo.lock"
test -f "$source_root/.cargo/config.toml"
for command in cc c++ cmake make curl sha256sum uv; do
  command -v "$command"
done
"$python" -c 'import sys,numpy,xarray,zarr; assert sys.version_info >= (3,12); assert numpy.__version__.split(".")[0] == "1"; assert zarr.__version__.split(".")[0] == "2"'
# A fresh output directory prevents accidental replacement of a qualified wheel.
mkdir "$build_root"
export CARGO_HOME="$build_root/cargo" RUSTUP_HOME="$build_root/rustup"
export CARGO_TARGET_DIR="$build_root/target" UV_CACHE_DIR="$build_root/uv-cache"
export TMPDIR="$build_root/tmp" PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CARGO_BUILD_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export RUSTUP_TOOLCHAIN=1.97.1
mkdir -p "$TMPDIR" "$build_root/tools" "$build_root/wheels" "$build_root/runtime"
exec > >(tee "$build_root/build.log") 2>&1
start=$SECONDS
trap 'code=$?; printf "exit_code=%s elapsed_seconds=%s\n" "$code" "$((SECONDS-start))" > "$build_root/status.txt"' EXIT
cd "$source_root"
"$python" - "$source_root" "$build_root/source-tree.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in sorted(root.rglob('*')) if path.is_file()}
Path(sys.argv[2]).write_text(json.dumps(files, indent=2) + '\n')
PY
sha256sum rust-toolchain.toml .cargo/config.toml rust/crab_load/Cargo.lock > "$build_root/source-inputs.sha256"
"$python" -c 'import tomllib; assert tomllib.load(open("rust-toolchain.toml","rb"))["toolchain"]["channel"] == "1.97.1"'
rustup_url=https://static.rust-lang.org/rustup/archive/1.28.2/aarch64-unknown-linux-gnu/rustup-init
curl --fail --location --retry 3 "$rustup_url" -o "$build_root/rustup-init"
curl --fail --location --retry 3 "$rustup_url.sha256" -o "$build_root/rustup-init.sha256"
(cd "$build_root" && sha256sum --check rustup-init.sha256)
chmod +x "$build_root/rustup-init"
"$build_root/rustup-init" -y --no-modify-path --profile minimal --default-toolchain 1.97.1
export PATH="$CARGO_HOME/bin:$PATH"
rustc --version --verbose > "$build_root/rustc.txt"
cargo --version > "$build_root/cargo.txt"
cc --version > "$build_root/cc.txt"
uv --version > "$build_root/uv.txt"
uv pip install --python "$python" --target "$build_root/tools" --only-binary=:all: --no-deps 'maturin==1.9.4'
export PATH="$build_root/tools/bin:$PATH"
export PYTHONPATH="$build_root/tools"
"$build_root/tools/bin/maturin" --version > "$build_root/maturin.txt"
printf 'bootstrap_seconds=%s\n' "$((SECONDS-start))" > "$build_root/timings.txt"
compile_start=$SECONDS
"$build_root/tools/bin/maturin" build --release --locked --manifest-path rust/crab_load/Cargo.toml \
  --features extension-module --interpreter "$python" --out "$build_root/wheels"
printf 'compile_seconds=%s\n' "$((SECONDS-compile_start))" >> "$build_root/timings.txt"
wheels=("$build_root"/wheels/*.whl)
test "${#wheels[@]}" = 1
sha256sum "${wheels[0]}" > "$build_root/wheel.sha256"
uv pip install --python "$python" --target "$build_root/runtime" --no-deps "${wheels[0]}"
export PYTHONPATH="$build_root/runtime"
audit_start=$SECONDS
"$python" scripts/audit_quarter_compact.py --store "$store" \
  --runtime "$build_root/runtime" --output "$build_root/compact-audit.json"
printf 'audit_seconds=%s\n' "$((SECONDS-audit_start))" >> "$build_root/timings.txt"
