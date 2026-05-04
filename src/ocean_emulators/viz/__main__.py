# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from ocean_emulators.viz.config import VizConfig, main

main(VizConfig.from_yaml_and_cli())
