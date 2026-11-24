# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Fingerprint the v2026-09 stores for a sampled cross-host identity check.

Read all metadata, static arrays and normalization chunks. Read first/middle/last
chunks of time-dependent ocean arrays. This does not verify every ocean chunk.
Usage: python fingerprint_data.py /path/to/om4_onedeg > fingerprint.json
"""

import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = {}
for store in ("OM4.zarr", "OM4_means.zarr", "OM4_stds.zarr"):
    path = root / store
    selected = list(path.glob(".*"))
    for array in sorted(path.iterdir()):
        if not array.is_dir():
            continue
        selected.extend(array.glob(".*"))
        meta = json.loads((array / ".zarray").read_text())
        if store != "OM4.zarr" or meta["shape"][0:1] != [4745]:
            selected.extend(
                p for p in array.iterdir() if p.is_file() and not p.name.startswith(".")
            )
        else:
            for step in (0, 2372, 4744):
                chunk = ".".join(
                    [str(step // meta["chunks"][0])] + ["0"] * (len(meta["shape"]) - 1)
                )
                selected.append(array / chunk)
    digest = hashlib.sha256()
    size = 0
    for f in sorted(set(selected)):
        if not f.is_file():
            raise FileNotFoundError(f)
        payload = f.read_bytes()
        digest.update(
            str(f.relative_to(path)).encode() + b"\0" + hashlib.sha256(payload).digest()
        )
        size += len(payload)
    records[store] = {
        "sample_sha256": digest.hexdigest(),
        "files": len(set(selected)),
        "bytes_read": size,
    }
print(json.dumps(records, indent=2))
