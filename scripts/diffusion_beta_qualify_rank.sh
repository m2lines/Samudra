#!/bin/bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
roles=(baseline deterministic diffusion geometry)
exec /workspace/.venv/bin/python -m samudra.experiments.diffusion_beta_qualify \
  --root "$1" --role "${roles[$SLURM_LOCALID]}"
