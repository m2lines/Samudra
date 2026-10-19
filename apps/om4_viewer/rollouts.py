# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Read-only access to the archived runs selected in the official workspace."""

import json
import subprocess

from native import PublicStore, SourceReadError

OFFICIAL_EVALS = (
    ("1°", "uw49dzuq", "2026-01-30-samudra_om4_onedeg_20_cap_eval", "om4_onedeg_v3"),
    (
        "½°",
        "20bfg825",
        "2026-02-05-samudra_om4_halfdeg_20_cap_fix_eval",
        "om4_halfdeg_v4",
    ),
    (
        "¼°",
        "9kl66qky",
        "2025-02-08-samudra_om4_quarterdeg_cap_20_eval_fix",
        "om4_quarterdeg_v2",
    ),
)
ARCHIVE_ROOT = "nyu-osn-emulators:emulators/jr7309/outputs"
TRUTH_ROOT = "nyu-osn-emulators:emulators/am16581/data/2025-11"


class ArchiveStore(PublicStore):
    """Use existing rclone authentication only on the server, with bounded caching."""

    def __init__(self, location, cache):
        super().__init__(location, cache)
        self.metadata = json.loads(self._read(".zmetadata"))["metadata"]

    def _read(self, key):
        location = self.url + "/" + key

        def download():
            result = subprocess.run(
                ["rclone", "cat", location, "--retries", "2"],
                check=False,
                capture_output=True,
                timeout=180,
            )
            if result.returncode:
                # Do not expose credential/config diagnostics in browser responses.
                raise SourceReadError(f"Could not read OSN object: {location}")
            return result.stdout

        return self.cache.get(location, read=download)

    def __getitem__(self, key):
        if key.startswith("/") or ".." in key.split("/"):
            raise KeyError(key)
        if key == ".zmetadata":
            return self._read(key)
        if key in self.metadata:
            return json.dumps(self.metadata[key]).encode()
        if key.rsplit("/", 1)[-1].startswith("."):
            raise KeyError(key)
        return self._read(key)
