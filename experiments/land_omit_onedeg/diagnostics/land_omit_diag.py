# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0
#
# Ran as Torch cs job 23771547 inside code layer 9250edf8; output in
# land-omit-diag-23771547.json.
"""Diagnose land: omit vs mask on 1-step forecasts, split coastal vs interior.

Usage: python land_omit_diag.py OUT.json
"""
import json
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from samudra.config import TrainConfig
from samudra.datasets import InferenceDataset

torch.set_num_threads(16)
RUNS = "/scratch/am16581/runs"
CASES = [  # (label, run, checkpoint, batch-stat BN)
    ("mask_e40", "2026-11-15-land-mask-onedeg", "ckpt_40.pt", False),
    ("omit_e40", "2026-11-15-land-omit-onedeg", "ckpt_40.pt", False),
    ("omit_e40_batchstat", "2026-11-15-land-omit-onedeg", "ckpt_40.pt", True),
    ("omit_e70", "2026-11-15-land-omit-onedeg", "ckpt.pt", False),
]
N_TRAIN_WINDOWS = 32
BATCH = 4


def load(run, ckpt_name):
    cfg = TrainConfig.from_yaml(f"{RUNS}/{run}/config.yaml")
    bundle = cfg.data.build(
        data_root=cfg.experiment.resolved_data_root, include_inference=False
    )
    layout = bundle.data_layout
    ni, no = cfg.data.input_steps, cfg.data.output_steps
    model = cfg.model.build(
        prog_channels=ni * len(layout.prognostic_var_names),
        boundary_channels=ni * len(layout.boundary_var_names),
        out_channels=no * len(layout.prognostic_var_names),
        input_steps=ni,
        grid_sizes=[s.grid_size for s in bundle.train_sources],
    )
    ck = torch.load(f"{RUNS}/{run}/saved_nets/{ckpt_name}", map_location="cpu")
    sd = OrderedDict((k.removeprefix("module."), v) for k, v in ck["model"].items())
    model.load_state_dict(sd)
    ema = ck["ema"]["ema_params"]
    for name, p in model.named_parameters():
        p.data.copy_(ema[name.replace(".", "")])
    model.eval()
    return cfg, bundle, layout, model, int(ck["epoch"])


def datasets(cfg, bundle, layout):
    out = {}
    for split, src in (("val", bundle.val_sources[0]), ("train", bundle.train_sources[0])):
        out[split] = InferenceDataset(
            source=src,
            prognostic_var_names=layout.prognostic_var_names,
            boundary_var_names=layout.boundary_var_names,
            input_steps=cfg.data.input_steps,
            output_steps=cfg.data.output_steps,
            normalize_before_mask=cfg.data.normalize_before_mask,
            masked_fill_value=cfg.data.masked_fill_value,
            long_rollout=False,
        )
    return out, bundle.val_sources[0].spherical_area_weights


def coastal(mask):  # (C, H, W) bool -> ocean cells with a land neighbor at their level
    land = (~mask).float().unsqueeze(1)
    land = F.pad(land, (1, 1, 0, 0), mode="circular")
    land = F.pad(land, (0, 0, 1, 1), mode="constant", value=1.0)
    near = F.max_pool2d(land, 3, stride=1).squeeze(1) > 0
    return mask & near


def main(out_path):
    results = {}
    for label, run, ckpt, batchstat in CASES:
        cfg, bundle, layout, model, epoch = load(run, ckpt)
        if batchstat:
            for m in model.modules():
                if isinstance(m, _BatchNorm):
                    m.track_running_stats = False
        dsets, area = datasets(cfg, bundle, layout)
        names = list(layout.prognostic_var_names) * cfg.data.output_steps
        groups = sorted({n.rsplit("_", 1)[0] if n.rsplit("_", 1)[-1].isdigit() else n for n in names})
        gidx = {g: [i for i, n in enumerate(names) if (n.rsplit("_", 1)[0] if n.rsplit("_", 1)[-1].isdigit() else n) == g] for g in groups}
        res = {"epoch": epoch}
        for split, ds in dsets.items():
            wet = ds.ctx.label_mask
            coast = coastal(wet)
            interior = wet & ~coast
            w = area.float()
            n = len(ds) if split == "val" else min(N_TRAIN_WINDOWS, len(ds))
            idx = list(range(len(ds))) if split == "val" else [round(i * (len(ds) - 1) / (n - 1)) for i in range(n)]
            se = {"coast": torch.zeros(len(names)), "interior": torch.zeros(len(names))}
            for b in range(0, len(idx), BATCH):
                items = [ds[i] for i in idx[b : b + BATCH]]
                prog = torch.cat([x[0] for x in items]); bnd = torch.cat([x[1] for x in items]); lab = torch.cat([x[2] for x in items])
                with torch.no_grad():
                    pred = model._assemble_prediction(prog, model.forward_once(prog, bnd, ds.ctx))
                err2 = ((pred - lab) ** 2).sum(0)  # (C, H, W)
                for k, m in (("coast", coast), ("interior", interior)):
                    se[k] += (err2 * m * w).sum((-2, -1))
                print(label, split, b, flush=True)
            out = {}
            for k, m in (("coast", coast), ("interior", interior)):
                denom = (m * w).sum((-2, -1)) * len(idx)
                mse = se[k] / denom.clamp_min(1e-12)
                out[k] = {g: float(mse[gidx[g]][denom[gidx[g]] > 0].mean()) for g in groups}
                out[k]["all"] = float(mse[denom > 0].mean())
                out[k + "_cell_fraction"] = float(m.sum() / wet.sum())
            out["n_windows"] = len(idx)
            res[split] = out
        results[label] = res
        json.dump(results, open(out_path, "w"), indent=1)
    print(json.dumps(results, indent=1))


main(sys.argv[1])
