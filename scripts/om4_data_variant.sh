#!/bin/bash

# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

# Resolve the raw source and output suffix shared by the preprocessing and
# normalization Slurm harnesses. Callers may set DATA_VARIANT before invoking
# this function.
resolve_om4_data_variant() {
  DATA_VARIANT="${DATA_VARIANT:-averaged}"
  case "${DATA_VARIANT}" in
    averaged)
      OM4_SOURCE_STORE="om4_5daily.zarr"
      OM4_OUTPUT_SUFFIX=""
      OM4_WFO_SOURCE_STORE="om4_5daily_snapshots.zarr"
      ;;
    snapshots)
      OM4_SOURCE_STORE="om4_5daily_snapshots.zarr"
      OM4_OUTPUT_SUFFIX="_snapshots"
      OM4_WFO_SOURCE_STORE=""
      ;;
    *)
      echo "ERROR: unknown DATA_VARIANT='${DATA_VARIANT}'." >&2
      echo "Expected one of: averaged | snapshots." >&2
      return 2
      ;;
  esac
}
