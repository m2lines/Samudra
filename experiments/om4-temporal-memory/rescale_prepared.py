"""Re-express an isolated prepared copy using verified archive statistics."""

import argparse
import json
from pathlib import Path

import numpy as np
from common import BOUND, PROG


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--statistics", required=True)
    a = ap.parse_args()
    source = Path(a.source)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    assert (source / "COMPLETE").exists()
    if (out / "COMPLETE").exists():
        return
    meta = json.loads((source / "metadata.json").read_text())
    frozen = json.loads(Path(a.statistics).read_text())
    oldm = np.array(meta["means"], dtype="float32")
    olds = np.array(meta["stds"], dtype="float32")
    newm = np.array(
        [frozen["OM4_means.zarr"].get(k, oldm[i]) for i, k in enumerate(PROG + BOUND)],
        dtype="float32",
    )
    news = np.array(
        [frozen["OM4_stds.zarr"].get(k, olds[i]) for i, k in enumerate(PROG + BOUND)],
        dtype="float32",
    )
    mask = np.load(source / "mask.npy")
    np.save(out / "mask.npy", mask)
    for name, start, stop, wet in [
        ("prognostic", 0, 77, mask),
        ("boundary", 77, 81, mask[0]),
    ]:
        original = np.load(source / f"{name}.npy", mmap_mode="r")
        target = np.lib.format.open_memmap(
            out / f"{name}.npy", mode="w+", dtype="float32", shape=original.shape
        )
        for j in range(0, len(original), 8):
            raw = (
                original[j : j + 8] * olds[start:stop, None, None]
                + oldm[start:stop, None, None]
            )
            target[j : j + 8] = np.where(
                wet,
                (raw - newm[start:stop, None, None]) / news[start:stop, None, None],
                0,
            )
        target.flush()
    meta["previous_local_means"] = oldm.tolist()
    meta["previous_local_stds"] = olds.tolist()
    meta["means"] = newm.tolist()
    meta["stds"] = news.tolist()
    meta["statistics_file"] = a.statistics
    meta["normalization"] = (
        "frozen official archive statistics; hfds anomaly definition unchanged"
    )
    (out / "metadata.json").write_text(json.dumps(meta, indent=2))
    (out / "COMPLETE").write_text("complete\n")


if __name__ == "__main__":
    main()
