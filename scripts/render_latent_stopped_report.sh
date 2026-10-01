#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
# Read-only analysis of already completed evaluations; never submits GPU jobs.
set -euo pipefail
short=${1:?Short-run root}
stopped=${2:?Stopped-run root}
references=${3:?Verified reference bundle}
baseline=${4:?Historical deterministic report}
physical=${5:?Physical diffusion report root}
native=${6:?Physical diffusion native root}
output=${7:?Output directory}
export PYTHONUNBUFFERED=1
mkdir -p "$output"
uv run python -m samudra.experiments.diffusion_latent_summary \
  --baseline "$baseline" --physical-reports "$physical" --previous-latent-runs "$short" \
  --latent-runs "$stopped" --output "$output"
uv run python -m samudra.experiments.diffusion_latent_figures --summary "$output/summary.json.gz" --output "$output"
uv run python -m samudra.experiments.diffusion_validation_figures --short-runs "$short" --extended-runs "$stopped" --output "$output"
uv run python -m samudra.experiments.diffusion_temporal_figures --short-runs "$short" --extended-runs "$stopped" --output "$output"
uv run python -m samudra.experiments.diffusion_latent_native_figures --latent-runs "$stopped" --pretraining-runs "$short" --physical-native "$native" --output "$output"
uv run python -m samudra.experiments.diffusion_latent_maps --baseline "$baseline" --physical-reports "$physical" --latent-runs "$stopped" --pretraining-runs "$short" --output "$output"
for seed in 1729 1730; do
  uv run python -m samudra.experiments.diffusion_latent_diagnostics --maps "$stopped/D-$seed/report-v1" --output "$output/diagnostics-$seed.json"
  uv run python -m samudra.experiments.diffusion_member_spectra --maps "$stopped/D-$seed/report-v1" --output "$output/member-spectra-$seed.json"
done
uv run python scripts/plot_latent_stopped_annual.py --short-runs "$short" --stopped-runs "$stopped" \
  --baseline "$baseline" --references "$references/compact" --heat-context "$references/heat-context.npz" --output "$output"
uv run python scripts/plot_latent_stopped_initializer.py --short-runs "$short" --stopped-runs "$stopped" \
  --references "$references/native" --climatology "$references/december-climatology.npz" --output "$output"
uv run python scripts/plot_latent_stopped_spectra.py --stopped-runs "$stopped" --references "$references/compact" \
  --native "$references/native" --output "$output"
uv run python scripts/plot_latent_day30_year.py --stopped-runs "$stopped" --baseline "$baseline" \
  --references "$references/compact" --native "$references/native" --output "$output"
