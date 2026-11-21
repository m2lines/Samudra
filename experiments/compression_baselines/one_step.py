# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
"""One-step (true-history) forecasts from a Samudra checkpoint.

    python one_step.py CKPT DATA_ROOT OUT --split test [--preset samudra_om4_v2/eval.yaml]

Each forecast starts from the *true* input states, never from an earlier
prediction ("true-history refresh"), so it measures single-step skill only.
The model, data pipeline, normalization and unnormalization are Samudra's own:
the config is the bundled eval preset with the checkpoint, data root and time
window replaced, `samudra.eval.Eval` builds everything, and outputs are
unnormalized as `samudra.utils.writer` does.

Samudra 2 predicts two five-day steps per call. Only the first, t+1 from
(t-1, t), is the one-step forecast the ladder is compared with. Windows advance
by the model's output length, so forecasts exist for every other time step.

Writes OUT/<split>/pred.npy, float32 (n_windows, channels, y, x) in physical
units with NaN on land, and OUT/<split>/meta.json (channel names, target times,
checkpoint, config).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from samudra.config import EvalConfig
from samudra.eval import Eval
from samudra.utils.device import get_device

# The 1-degree preset's windows (src/samudra/configs/data/om4.yaml).
WINDOWS = {"val": ("2013-10-05", "2014-10-05"), "test": ("2014-10-10", "2022-12-24")}


OSN = "https://nyu1.osn.mghpcc.org"


def official(res: str, name: str) -> dict:
    """A structured S3 location in the public v2026-09 release (docs/data.md)."""
    return {
        "type": "s3",
        "endpoint_url": OSN,
        "anon": True,
        "bucket": "m2lines-pubs",
        "path": f"Samudra/v2026-09/{res}/{name}",
    }


def build_config(
    preset: str,
    ckpt: Path,
    data_root: Path | None,
    out: Path,
    split: str,
    backend: str,
    official_res: str | None = None,
) -> EvalConfig:
    d = EvalConfig._load_yaml(preset)  # the preset, includes resolved
    start, end = WINDOWS[split]
    d["backend"] = backend
    d["ckpt_path"] = str(ckpt)
    d["save_zarr"] = False
    if data_root is not None:
        d["experiment"]["data_root"] = str(data_root)
    d["experiment"]["base_output_dir"] = str(out)
    d["experiment"]["name"] = f"one_step_{split}"
    d["experiment"]["wandb"]["mode"] = "disabled"
    sources = d["data"]["sources"]
    if len(sources) != 1:
        raise ValueError(f"expected one data source, got {len(sources)}")
    sources[0]["inference_time"] = {"start": start, "end": end}
    if official_res is not None:
        # The official release lacks hfds_anomalies; Samudra derives it from hfds
        # (samudra.utils.data.compute_anomalies) when a preset asks for it.
        sources[0]["data_location"] = official(official_res, "OM4.zarr")
        sources[0]["data_means_location"] = official(official_res, "OM4_means.zarr")
        sources[0]["data_stds_location"] = official(official_res, "OM4_stds.zarr")
    return EvalConfig(**d)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("ckpt", type=Path)
    ap.add_argument(
        "data_root", help="local data directory, or 'official:<res>' for v2026-09"
    )
    ap.add_argument("out", type=Path)
    ap.add_argument("--split", choices=sorted(WINDOWS), required=True)
    ap.add_argument("--preset", default="samudra_om4_v2/eval.yaml")
    ap.add_argument("--backend", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--max-windows", type=int, default=0, help="smoke test only")
    a = ap.parse_args()

    res = (
        a.data_root.removeprefix("official:")
        if a.data_root.startswith("official:")
        else None
    )
    root = None if res else Path(a.data_root)
    cfg = build_config(a.preset, a.ckpt, root, a.out, a.split, a.backend, res)
    ev = Eval(cfg)
    model = ev.model.eval()
    ds = ev.inference_dataset
    ds.to(get_device())
    pre = ev.preprocessor
    names = list(ev.prognostic_var_names)
    steps = ds.output_steps
    n = len(ds) if a.max_windows <= 0 else min(a.max_windows, len(ds))

    split_dir = a.out / a.split
    split_dir.mkdir(parents=True, exist_ok=True)
    prog0, _, label0 = ds[0]
    h, w = label0.shape[-2:]
    pred = np.lib.format.open_memmap(
        split_dir / "pred.npy", mode="w+", dtype=np.float32, shape=(n, len(names), h, w)
    )
    times = []
    t0 = time.time()
    for k in range(n):
        prog, boundary, _ = ds[k]  # true input states and forcing for window k
        prog = prog.to(get_device())
        boundary = boundary.to(get_device())
        out = model._assemble_prediction(
            prog, model.forward_once(prog, boundary, ds.ctx)
        )
        out = rearrange(out, "n (hi c) h w -> (n hi) c h w", hi=steps)
        out = pre.unnormalize_tensor_prognostic(out, fill_value=float("nan"))
        pred[k] = out[0].float().cpu().numpy()  # first output step: t+1
        times.append(str(ds.get_target_time(k, 1).values[0]))
        if k % 25 == 0:
            print(
                f"[one_step] {a.split} {k + 1}/{n} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    pred.flush()
    meta = {
        "split": a.split,
        "window": WINDOWS[a.split],
        "channels": names,
        "target_times": times,
        "output_steps_per_call": steps,
        "checkpoint": str(a.ckpt),
        "data_root": str(a.data_root),
        "preset": a.preset,
        "config": cfg.model_dump(mode="json"),
        "complete": n == len(ds),
        "seconds": time.time() - t0,
    }
    (split_dir / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[one_step] wrote {split_dir} ({n} forecasts)")


if __name__ == "__main__":
    main()
