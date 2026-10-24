"""Cheap CPU-capable replay gate before allocating a diagnostic GPU."""

import argparse
import json
import time

import numpy as np
import torch
from common import OceanModel, Prepared, load_official


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    data = Prepared(args.data)
    m = OceanModel(torch.tensor(data.mask), checkpointing=False).eval()
    load_official(m, args.checkpoint)
    x, b, _ = data.window(data.meta["diagnostic_origins"][0], 1, "cpu")
    start = time.monotonic()
    with torch.no_grad():
        p = m(x, b[0])[0].reshape(2, 77, 180, 360).numpy()
    physical = p * data.std[None, :, None, None] + data.mean[None, :, None, None]
    ref = np.load(args.reference)["prediction"]
    d = (physical - ref) / data.std[None, :, None, None]
    wet = np.broadcast_to(data.mask[None], d.shape)
    result = {
        "seconds": time.monotonic() - start,
        "normalized_rmse": float(np.sqrt(np.mean(d[wet].astype("float64") ** 2))),
        "normalized_max_abs": float(np.max(abs(d[wet]))),
    }
    print(json.dumps(result), flush=True)
    assert (
        np.isfinite(result["normalized_rmse"]) and result["normalized_rmse"] < 0.001
    ), result


if __name__ == "__main__":
    main()
