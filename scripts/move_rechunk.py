#!/usr/bin/env python
"""Move finished hours of the 1/48-degree rechunk store off scratch while the fill runs.

An hour is finished once its marker exists in `<coarse-root>/_progress`: its
rechunk tiles are written before the marker and never touched again. For each
finished hour the mover copies `<var>/<t>/` for all five variables (parallel
rsync via copy_tree_parallel.sh), compares every file's name and size, and only
then deletes the scratch copy. Moved hours are listed in `<dst-root>/moved_hours.txt`.

Store metadata, `hFacC_2d` and `time` are copied but never deleted: new fill
workers read `hFacC_2d` from the scratch store.

Repeats every --interval seconds while a `coarsen_LLC` job is queued, then makes
one final pass. Safe to kill and rerun.
"""

from __future__ import annotations

import argparse
import getpass
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

VARS_2D = ("Eta", "oceTAUX", "oceTAUY", "oceQnet", "oceFWflx")
STORE = "LLC4320_2d_720.zarr"
COPY = Path(__file__).resolve().parent / "copy_tree_parallel.sh"


def listing(root: Path) -> dict[str, int]:
    """Relative path -> size of every finished file under `root` (skips in-flight *.partial)."""
    out = {}
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith(".partial"):
                path = Path(dirpath) / name
                out[str(path.relative_to(root))] = path.stat().st_size
    return out


def copy(src: Path, dst: Path, files: list[str], streams: int) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".files", delete=False) as f:
        f.write("\n".join(files) + "\n")
    try:
        subprocess.run(["bash", str(COPY), str(src), str(dst), str(streams)], check=True,
                       env=os.environ | {"FILE_LIST": f.name, "PROGRESS_EVERY": "600"})
    finally:
        os.unlink(f.name)


def copy_metadata(src: Path, dst: Path, streams: int) -> None:
    """Store/array metadata plus the static hFacC_2d and time arrays; never deleted."""
    files = [n for n in (".zgroup", ".zattrs", ".zmetadata") if (src / n).exists()]
    files += [f"{v}/{n}" for v in VARS_2D for n in (".zarray", ".zattrs") if (src / v / n).exists()]
    for static in ("hFacC_2d", "time"):
        files += [f"{static}/{p}" for p in listing(src / static)]
    copy(src, dst, files, streams)


def fill_running() -> bool:
    out = subprocess.run(["squeue", "-h", "-u", getpass.getuser(), "-n", "coarsen_LLC"],
                         capture_output=True, text=True, check=True).stdout
    return bool(out.strip())


def one_pass(args, src: Path, dst: Path, ledger: Path) -> int:
    done = {int(n) for n in os.listdir(args.coarse_root / "_progress")}
    moved = {int(x) for x in ledger.read_text().split()} if ledger.exists() else set()
    # Hours rewritten after a repair reappear on scratch; move those again too.
    again = {t for t in moved & done if any((src / v / str(t)).exists() for v in VARS_2D)}
    todo = sorted((done - moved) | again)
    logger.info("%d finished hours, %d already moved, %d to move (%d again)", len(done), len(moved), len(todo), len(again))
    copy_metadata(src, dst, args.streams)
    for start in range(0, len(todo), args.batch):
        hours = todo[start:start + args.batch]
        listings = {(v, t): listing(src / v / str(t)) for t in hours for v in VARS_2D}
        files = [f"{v}/{t}/{p}" for (v, t), files in listings.items() for p in files]
        copy(src, dst, files, args.streams)
        for t in hours:
            if all(listing(dst / v / str(t)) == listings[(v, t)] for v in VARS_2D):
                for v in VARS_2D:
                    shutil.rmtree(src / v / str(t), ignore_errors=True)
                with ledger.open("a") as f:
                    f.write(f"{t}\n")
            else:
                logger.error("hour %d: copy does not match the source; left in place", t)
        logger.info("moved hours %d..%d (%d files)", hours[0], hours[-1], len(files))
    return len(todo)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--coarse-root", type=Path, default=Path("/orcd/scratch/seedfund/001/llc_emulator/coarse"))
    p.add_argument("--src-root", type=Path, default=Path("/orcd/scratch/seedfund/001/llc_emulator/rechunk"))
    p.add_argument("--dst-root", type=Path, default=Path("/orcd/data/abodner/002/cody/LLC_boundary_rechunk"))
    p.add_argument("--streams", type=int, default=16)
    p.add_argument("--batch", type=int, default=100, help="Hours per copy/verify/delete round")
    p.add_argument("--interval", type=int, default=1800, help="Seconds between passes")
    p.add_argument("--once", action="store_true", help="One pass, then exit")
    args = p.parse_args()
    src, dst = args.src_root / STORE, args.dst_root / STORE
    dst.mkdir(parents=True, exist_ok=True)
    ledger = args.dst_root / "moved_hours.txt"
    while True:
        running = fill_running()  # checked first, so the last pass sees every finished hour
        one_pass(args, src, dst, ledger)
        if args.once or not running:
            logger.info("no coarsen_LLC job left; final pass done")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
