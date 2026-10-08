# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete file-level integrity checks for saved annual forecasts."""

from pathlib import Path

from samudra.experiments.observation_pilot import digest


def annual_output_hashes(output, complete):
    """Hash every required result; a completion marker alone is insufficient."""
    required = {"input.json", "COMPLETE.json"}
    for item in complete["origins"].values():
        if set(item) != {"metrics", "arrays", "persistence"}:
            raise ValueError("Missing annual result type")
        required.update(item.values())
    hashes = {}
    for filename in sorted(required):
        if Path(filename).name != filename or not (output / filename).is_file():
            raise ValueError("Missing or invalid annual output: " + filename)
        hashes[filename] = digest(output / filename)
    return hashes
