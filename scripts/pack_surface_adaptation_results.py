#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Pack collected result bytes below the repository file-size limit and verify them."""

import argparse
import gzip
import hashlib
from pathlib import Path

from samudra.experiments.surface_adaptation_analysis import read_bytes

LIMIT = 240 * 1024
LICENSE = "SPDX-FileCopyrightText: 2026 Samudra Authors\n\nSPDX-License-Identifier: CC-BY-4.0\n"


def licensed_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    Path(str(path) + ".license").write_text(LICENSE)


def pack(source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use an empty destination to prevent stale parts")
    sums = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix not in (
            ".csv",
            ".json",
            ".jsonl",
            ".log",
            ".npz",
        ):
            continue
        relative = path.relative_to(source)
        data = path.read_bytes()
        sums.append(hashlib.sha256(data).hexdigest() + "  " + str(relative))
        destination = output / relative
        if len(data) <= LIMIT:
            licensed_write(destination, data)
        else:
            compressed = gzip.compress(data, mtime=0)
            if len(compressed) <= LIMIT:
                licensed_write(Path(str(destination) + ".gz"), compressed)
            else:
                for index, start in enumerate(range(0, len(compressed), LIMIT)):
                    licensed_write(
                        Path(str(destination) + f".gz.part{index:03d}"),
                        compressed[start : start + LIMIT],
                    )
        if read_bytes(destination) != data:
            raise ValueError(f"Round-trip byte mismatch: {relative}")
    licensed_write(output / "source.sha256", ("\n".join(sums) + "\n").encode())
    return verify(output)


def verify(output):
    count = 0
    for line in (output / "source.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        if hashlib.sha256(read_bytes(output / relative)).hexdigest() != expected:
            raise ValueError(f"Checksum mismatch: {relative}")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        count = verify(args.output)
    else:
        if args.source is None:
            parser.error("--source is required when packing")
        count = pack(args.source, args.output)
    print(f"Verified {count} original result files byte-for-byte")


if __name__ == "__main__":
    main()
