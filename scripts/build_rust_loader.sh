#!/bin/bash
# Build the optional Rust LLC data loader and install it into the project venv.
#
# Nothing else in the project needs Rust. If this is not run, or if it fails,
# `data.loader_backend=cpu` (the default) is completely unaffected -- the
# extension is only imported when the rust backend is selected.
#
# To remove the feature entirely, see rust/llc_load/README.md.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE_DIR="${REPO_ROOT}/rust/llc_load"
VENV="${VENV:-${REPO_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV}/bin/python}"

export CARGO_HOME="${CARGO_HOME:-${HOME}/.cargo}"
export RUSTUP_HOME="${RUSTUP_HOME:-${HOME}/.rustup}"
CARGO="${CARGO:-${CARGO_HOME}/bin/cargo}"

if [[ ! -x "${CARGO}" ]]; then
  if command -v cargo >/dev/null 2>&1; then
    CARGO="$(command -v cargo)"
  else
    echo "Installing a minimal Rust toolchain into ${CARGO_HOME} ..."
    curl -sSf --proto '=https' --tlsv1.2 https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
    CARGO="${CARGO_HOME}/bin/cargo"
  fi
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "ERROR: no Python at ${PYTHON_BIN}; set PYTHON_BIN or VENV." >&2
  exit 1
fi

SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "cargo:         ${CARGO}"
echo "python:        ${PYTHON_BIN}"
echo "site-packages: ${SITE_PACKAGES}"

# `extension-module` stops pyo3 linking libpython, which a venv interpreter does
# not ship; PYO3_PYTHON pins the ABI to the interpreter we install into.
PYO3_PYTHON="${PYTHON_BIN}" "${CARGO}" build \
  --manifest-path "${CRATE_DIR}/Cargo.toml" \
  --release --features extension-module

install -m 0755 \
  "${CRATE_DIR}/target/release/libocean_llc_loader.so" \
  "${SITE_PACKAGES}/ocean_llc_loader.so"

"${PYTHON_BIN}" -c 'import ocean_llc_loader; print("ocean_llc_loader OK:", sorted(n for n in dir(ocean_llc_loader) if not n.startswith("_")))'
