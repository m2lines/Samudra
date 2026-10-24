#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
app_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --project "$app_dir" python "$app_dir/server.py" "$@"
