# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Controlled loss comparison on real 2-degree OM4 data.

Same small CNN, data, steps, seed and optimizer in every arm; only the loss
weighting and the output head change. Uses the PR's own TendencyLoss,
Samudra's DynamicLoss, and the PR's persistence_rms statistics function.

Run from the repository root (for ``samudra`` and ``scripts`` imports):
    python experiments/tendency_loss/run.py load            # cache 2-degree data
    python experiments/tendency_loss/run.py train ARM SEED  # one arm -> results/

Cached arrays and checkpoints go to $TENDENCY_EXP_WORK (default
/tmp/tendency_loss, about 2 GB); per-arm metrics go to results/ next to this
file. The original runs wrote everything to one scratch directory and saved the
physical stds for plotting in a separate step; otherwise the code is the one that
produced the checked-in results.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch import nn

from samudra.utils.ctx import BatchGrid
from samudra.utils.loss import (
    DynamicLoss,
    TendencyLoss,
    loss_fn_from_metric,
    tendency_loss_weights,
)
from scripts.compute_tendency_stats import persistence_rms

HERE = Path(__file__).parent
WORK = Path(os.environ.get("TENDENCY_EXP_WORK", "/tmp/tendency_loss"))
RESULTS = HERE / "results"
ROOT = "https://nyu1.osn.mghpcc.org/m2lines-pubs/Samudra/v2026-09/om4_twodeg/"
FIELDS = ["uo", "vo", "thetao", "so"]
PROG = [f"{f}_{z}" for f in FIELDS for z in range(19)] + ["zos"]
FORCING = ["tauuo", "tauvo", "hfds"]
TRAIN = ("1990-01-01", "1993-12-31")
VAL = ("2013-10-05", "2014-10-05")  # Samudra's validation window
STEPS = int(os.environ.get("STEPS", 2500))
BATCH, LR, WIDTH = 4, 1e-3, 48

ARMS = {
    # name: (loss, head, output channels)
    "equal_abs": ("equal", "abs", "all"),
    "dynamic_abs": ("dynamic", "abs", "all"),  # Samudra production
    "tendency_abs": ("tendency", "abs", "all"),
    "equal_res": ("equal", "res", "all"),
    "dynamic_res": ("dynamic", "res", "all"),
    "tendency_res": ("tendency", "res", "all"),  # proposed
    "tonly_res": ("equal", "res", "thetao"),  # no-mixing reference
    # GraphCast's full parameterization: network output in units of each
    # channel's one-step change (added after the first seven arms had run;
    # it does not change their code paths).
    "tendency_res_scaled": ("tendency", "res_scaled", "all"),
    # Same as tendency_res with a 20x cap, the ratio DynamicLoss uses in production.
    "tendency20_res": ("tendency20", "res", "all"),
    # Output scaling with Samudra's existing DynamicLoss: is the loss still needed?
    "dynamic_res_scaled": ("dynamic", "res_scaled", "all"),
}


def level(name):
    return int(name.rsplit("_", 1)[1]) if name != "zos" else 0


def load():
    WORK.mkdir(parents=True, exist_ok=True)
    ds = xr.open_zarr(ROOT + "OM4.zarr", chunks=None)
    means = xr.open_zarr(ROOT + "OM4_means.zarr")
    stds = xr.open_zarr(ROOT + "OM4_stds.zarr")
    cal = ds.time.dt.calendar
    import cftime

    def sl(a, b):
        return slice(
            cftime.datetime.strptime(a, "%Y-%m-%d", calendar=cal),
            cftime.datetime.strptime(b, "%Y-%m-%d", calendar=cal),
        )

    masks = np.stack([ds[f"mask_{z}"].values for z in range(19)]).astype(bool)
    out = {"masks": masks}
    for split, (a, b) in {"train": TRAIN, "val": VAL}.items():
        sub = ds.sel(time=sl(a, b))
        for kind, names in {"prog": PROG, "forc": FORCING}.items():
            # Preallocate on disk and fill one variable at a time: peak memory
            # stays at one field, not the whole stack.
            nt, ny, nx = sub.sizes["time"], sub.sizes["y"], sub.sizes["x"]
            arr = np.lib.format.open_memmap(
                WORK / f"{split}_{kind}.npy",
                mode="w+",
                dtype=np.float32,
                shape=(nt, len(names), ny, nx),
            )
            for c, n in enumerate(names):
                x = sub[n].values.astype(np.float32)
                x = (x - float(means[n])) / float(stds[n])
                wet = masks[level(n)] if kind == "prog" else masks[0]
                arr[:, c] = np.where(wet[None], np.nan_to_num(x), 0.0)
            arr.flush()
            del arr
            print(split, kind, (nt, len(names), ny, nx), flush=True)
    # Persistence RMS in *physical* units over the training subset, via the
    # PR's own function, then converted to normalized units.
    raw = ds.sel(time=sl(*TRAIN))
    phys = persistence_rms(raw, PROG, time_stride=1)
    out["rms_norm"] = np.array([float(phys[n]) / float(stds[n]) for n in PROG])
    out["phys_stds"] = np.array([float(stds[n]) for n in PROG])  # for °C plots
    for k, v in out.items():
        np.save(WORK / f"{k}.npy", v)


class Net(nn.Module):
    """Small CNN, zonally periodic, zero-padded in latitude."""

    def __init__(self, cin, cout, width=WIDTH, depth=4):
        super().__init__()
        self.inp = nn.Conv2d(cin, width, 3)
        self.hidden = nn.ModuleList(nn.Conv2d(width, width, 3) for _ in range(depth))
        self.out = nn.Conv2d(width, cout, 3)

    @staticmethod
    def pad(x):
        return F.pad(F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1))

    def forward(self, x):
        h = F.gelu(self.inp(self.pad(x)))
        for conv in self.hidden:
            h = h + F.gelu(conv(self.pad(h)))
        return self.out(self.pad(h))


def train(arm, seed):
    loss_kind, head, outs = ARMS[arm]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(int(os.environ.get("THREADS", 1)))
    d = {
        k: np.load(WORK / f"{k}.npy", mmap_mode="r")
        for k in (
            "masks",
            "train_prog",
            "train_forc",
            "val_prog",
            "val_forc",
            "rms_norm",
        )
    }
    masks = torch.from_numpy(np.array(d["masks"]))
    out_idx = (
        list(range(77))
        if outs == "all"
        else [PROG.index(f"thetao_{z}") for z in range(19)]
    )
    wet = torch.stack([masks[level(PROG[i])] for i in out_idx])
    ctx = BatchGrid(
        wet, (torch.zeros(1), torch.zeros(1)), (torch.zeros(1), torch.zeros(1))
    )
    static = masks.float()[None]  # wet masks as extra input channels

    tp, tf = d["train_prog"], d["train_forc"]  # memory-mapped
    vp = torch.from_numpy(np.array(d["val_prog"]))
    vf = torch.from_numpy(np.array(d["val_forc"]))
    cin = 77 + 3 + 19
    net = Net(cin, len(out_idx))
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=0.0)
    mse = loss_fn_from_metric("mse")
    n = len(out_idx)
    if loss_kind == "equal":
        loss_fn = mse
    elif loss_kind == "dynamic":
        loss_fn = DynamicLoss(
            mse, limit=20.0, device=torch.device("cpu"), num_channels=n
        )
    else:
        rms = torch.from_numpy(np.array(d["rms_norm"])[out_idx])
        cap = 20.0 if loss_kind == "tendency20" else 1000.0
        loss_fn = TendencyLoss(
            mse,
            weights=tendency_loss_weights(rms, max_weight_ratio=cap),
            device=torch.device("cpu"),
        )

    change_scale = (
        torch.from_numpy(np.array(d["rms_norm"])[out_idx]).float().view(1, -1, 1, 1)
    )

    def predict(prog, forc):
        x = torch.cat([prog, forc, static.expand(len(prog), -1, -1, -1)], 1)
        y = net(x)
        if head == "res":
            y = prog[:, out_idx] + y
        elif head == "res_scaled":
            y = prog[:, out_idx] + y * change_scale
        return y * wet[None]

    t0 = time.time()
    for step in range(1, STEPS + 1):
        lr = LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
        for g in opt.param_groups:
            g["lr"] = lr
        i = rng.integers(0, len(tp) - 1, BATCH)
        i.sort()
        target = torch.from_numpy(np.array(tp[i + 1]))[:, out_idx]
        pred = predict(
            torch.from_numpy(np.array(tp[i])), torch.from_numpy(np.array(tf[i]))
        )
        per_channel = loss_fn(pred, target, ctx)
        opt.zero_grad()
        per_channel.mean().backward()
        opt.step()
        if isinstance(loss_fn, DynamicLoss):  # same condition as loss_kind == "dynamic"
            loss_fn.update(mse(pred.detach(), target, ctx))
        if step % 250 == 0:
            print(
                f"{arm} s{seed} step {step} loss {per_channel.mean().item():.4g} "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )

    # Evaluate one-step forecasts on every validation origin.
    net.eval()
    L = torch.zeros(n)
    P = torch.zeros(n)
    rough_num = torch.zeros(n)
    rough_den = torch.zeros(n)
    with torch.no_grad():
        for s in range(0, len(vp) - 1, 8):
            e = min(s + 8, len(vp) - 1)
            prog, forc = vp[s:e], vf[s:e]
            target = vp[s + 1 : e + 1][:, out_idx]
            pred = predict(prog, forc)
            L += mse(pred, target, ctx) * (e - s)
            P += mse(prog[:, out_idx], target, ctx) * (e - s)
            # Small-scale roughness: squared differences between adjacent wet
            # cells, prediction vs truth (x and y neighbours).
            for dim in (2, 3):
                pair = wet & torch.roll(wet, -1, dim - 1)
                if dim == 2:
                    pair[:, -1, :] = False
                dp = (pred - torch.roll(pred, -1, dim)) * pair
                dt = (target - torch.roll(target, -1, dim)) * pair
                rough_num += dp.square().sum((0, 2, 3))
                rough_den += dt.square().sum((0, 2, 3))
    count = len(vp) - 1
    names = [PROG[i] for i in out_idx]
    result = {
        "arm": arm,
        "seed": seed,
        "seconds": time.time() - t0,
        "channels": names,
        "model_mse": (L / count).tolist(),
        "persistence_mse": (P / count).tolist(),
        "roughness_ratio": (rough_num / rough_den).tolist(),
    }
    RESULTS.mkdir(exist_ok=True)
    torch.save(net.state_dict(), WORK / f"{arm}_s{seed}.pt")
    (RESULTS / f"{arm}_s{seed}.json").write_text(json.dumps(result))
    print(f"done {arm} s{seed} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "load":
        load()
    else:
        train(sys.argv[2], int(sys.argv[3]))
