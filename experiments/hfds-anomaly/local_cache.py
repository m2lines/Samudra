"""Verified immutable node-local mirror; no transforms or changed array values."""

import concurrent.futures
import fcntl
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

SOURCE = Path("/orcd/data/abodner/002/jrusak/om4_onedeg_v3/OM4.zarr")


def stage():
    metadata_hash = hashlib.sha256((SOURCE / ".zmetadata").read_bytes()).hexdigest()
    root = Path("/tmp") / f"hfds-input-{os.getuid()}-{metadata_hash[:16]}"
    root.mkdir(mode=0o700, exist_ok=True)
    assert root.stat().st_uid == os.getuid() and not root.is_symlink()
    destination = root / "OM4.zarr"
    with (root / "lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ready = root / "VERIFIED.json"
        if ready.exists():
            record = json.loads(ready.read_text())
            assert record["metadata_sha256"] == metadata_hash
            print("LOCAL_CACHE_REUSE", destination, flush=True)
            return destination
        entries = []
        # Mirror only arrays declared by consolidated metadata. Old, unreferenced
        # anomaly arrays are not part of the dataset opened by the native loader.
        meta = json.loads((SOURCE / ".zmetadata").read_text())["metadata"]
        arrays = sorted(k[:-8] for k in meta if k.endswith("/.zarray"))
        entries.extend(
            p for p in SOURCE.iterdir() if p.is_file() and p.name.startswith(".")
        )
        for name in arrays:
            entries.extend(p for p in (SOURCE / name).iterdir() if p.is_file())
        size = sum(p.stat().st_size for p in entries)
        if shutil.disk_usage(root).free < size + 8 * 1024**3:
            raise RuntimeError(f"Insufficient node-local storage for {size} bytes")
        print("LOCAL_CACHE_BEGIN", len(entries), size, destination, flush=True)
        start = time.monotonic()

        def copy_verify(src):
            dst = destination / src.relative_to(SOURCE)
            dst.parent.mkdir(parents=True, exist_ok=True)
            source_hash = hashlib.sha256()
            with src.open("rb") as source, dst.open("wb") as target:
                while block := source.read(1024 * 1024):
                    source_hash.update(block)
                    target.write(block)
            target_hash = hashlib.sha256()
            with dst.open("rb") as target:
                while block := target.read(1024 * 1024):
                    target_hash.update(block)
            assert source_hash.digest() == target_hash.digest(), str(src)
            return str(src.relative_to(SOURCE)), source_hash.hexdigest()

        manifest = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            for i, (name, digest) in enumerate(pool.map(copy_verify, entries), 1):
                manifest[name] = digest
                if i % 10000 == 0:
                    print(
                        "LOCAL_CACHE_PROGRESS",
                        i,
                        len(entries),
                        round(time.monotonic() - start, 1),
                        flush=True,
                    )
        encoded = json.dumps(manifest, sort_keys=True).encode()
        (root / "sha256.json").write_bytes(encoded)
        record = dict(
            source=str(SOURCE),
            destination=str(destination),
            files=len(entries),
            bytes=size,
            metadata_sha256=metadata_hash,
            manifest_sha256=hashlib.sha256(encoded).hexdigest(),
            seconds=time.monotonic() - start,
        )
        temporary = root / "VERIFIED.tmp"
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(ready)
        print("LOCAL_CACHE_VERIFIED", json.dumps(record), flush=True)
        return destination
