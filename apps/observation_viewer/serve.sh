#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
viewer_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
viewer_port="${PORT:-61015}"
viewer_browser_port="${BROWSER_PORT:-$viewer_port}"
export PYTHONUNBUFFERED=1
exec uv run --frozen --project "$viewer_dir" panel serve "$viewer_dir/app.py" \
  --address 127.0.0.1 --port "$viewer_port" \
  --allow-websocket-origin="localhost:$viewer_port" \
  --allow-websocket-origin="127.0.0.1:$viewer_port" \
  --allow-websocket-origin="localhost:$viewer_browser_port" \
  --allow-websocket-origin="127.0.0.1:$viewer_browser_port"
