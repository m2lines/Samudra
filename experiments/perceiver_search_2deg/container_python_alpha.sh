#!/usr/bin/env bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Run a search controller inside the same x86 container and source checkout as
# its Alpha GPU workers, while exposing Slurm submission commands.

set -euo pipefail
: "${CODE_DIR:?CODE_DIR is required}"
: "${CODE_COMMIT:?CODE_COMMIT is required}"
: "${SIF_PATH:?SIF_PATH is required}"
: "${SCRATCH_DIR:?SCRATCH_DIR is required}"

test "$(git -C "$CODE_DIR" rev-parse HEAD)" = "$CODE_COMMIT"
test -z "$(git -C "$CODE_DIR" status --porcelain)"

exec apptainer exec \
  --bind "$CODE_DIR:/opt/samudra-code:ro" \
  --bind "$SCRATCH_DIR:$SCRATCH_DIR" \
  --bind /cm:/cm:ro \
  --bind /usr/lib/x86_64-linux-gnu/libmunge.so.2:/usr/lib/x86_64-linux-gnu/libmunge.so.2:ro \
  --bind /usr/lib/x86_64-linux-gnu/libmunge.so.2.0.1:/usr/lib/x86_64-linux-gnu/libmunge.so.2.0.1:ro \
  --bind /run/munge:/run/munge:ro \
  --bind /etc/passwd:/etc/passwd:ro \
  --bind /etc/group:/etc/group:ro \
  --pwd /opt/samudra-code \
  --env PATH=/cm/shared/apps/slurm/current/bin:/workspace/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin \
  --env LD_LIBRARY_PATH=/cm/shared/apps/slurm/current/lib64/slurm:/usr/lib/x86_64-linux-gnu \
  --env PYTHONPATH="$SCRATCH_DIR/python-site:/opt/samudra-code/src" \
  --env "SAMUDRA_CODE_COMMIT=$CODE_COMMIT" \
  "$SIF_PATH" \
  /workspace/.venv/bin/python "$@"
