# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Is land: omit's deficit a batch-norm train/eval mismatch?

Scores each arm's epoch-40 checkpoint four ways:

* ``ema_running``: EMA weights with the stored running statistics, which
  the live weights collected (what training validation and eval use);
* ``live_running``: the live weights with their own running statistics;
* ``ema_recal``: EMA weights, with every batch norm's running statistics
  recomputed for them (a cumulative average over training windows);
* ``ema_batchstat``: EMA weights, each batch normalized by its own
  statistics.

For each, it reports 1-step error on the validation windows, split into
coastal and interior cells, and the 360-day rollout score, computed by the
trainer's own rollout validation (``validate_rollout`` with
``RolloutValidationAggregator``), so ``ema_running`` should reproduce the
W&B ``rollout_val/360d`` values at epoch 40.

Usage: python bn_diag.py OUT.json
"""

import copy
import json
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from samudra.aggregator.validate.rollout import RolloutValidationAggregator
from samudra.config import TrainConfig
from samudra.datasets import InferenceDataset
from samudra.stepper import validate_rollout
from samudra.utils.data import BatchPreprocessor
from samudra.utils.rollout_validation import RolloutValidationSpec

torch.set_num_threads(16)
RUNS = "/scratch/am16581/runs"
ARMS = {"mask": "2026-11-15-land-mask-onedeg", "omit": "2026-11-15-land-omit-onedeg"}
CKPT = "ckpt_40.pt"
N_RECAL_WINDOWS = 128
BATCH = 4


def coastal(mask):
    land = (~mask).float().unsqueeze(1)
    land = F.pad(land, (1, 1, 0, 0), mode="circular")
    land = F.pad(land, (0, 0, 1, 1), mode="constant", value=1.0)
    return mask & (F.max_pool2d(land, 3, stride=1).squeeze(1) > 0)


def group(name):
    base, _, depth = name.rpartition("_")
    return base if depth.isdigit() else name


def dataset(cfg, layout, src, long_rollout):
    return InferenceDataset(
        source=src,
        prognostic_var_names=layout.prognostic_var_names,
        boundary_var_names=layout.boundary_var_names,
        input_steps=cfg.data.input_steps,
        output_steps=cfg.data.output_steps,
        normalize_before_mask=cfg.data.normalize_before_mask,
        masked_fill_value=cfg.data.masked_fill_value,
        long_rollout=long_rollout,
    )


def set_params(model, params):
    for name, p in model.named_parameters():
        p.data.copy_(params[name])


def batches(ds, idx):
    for b in range(0, len(idx), BATCH):
        items = [ds[i] for i in idx[b : b + BATCH]]
        yield tuple(torch.cat([x[k] for x in items]) for k in range(3))


def recalibrate(model, ds):
    """Recompute every batch norm's running statistics for the current weights."""
    bns = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.eval()
    for m in bns:
        m.train()
    n = len(ds)
    idx = [round(i * (n - 1) / (N_RECAL_WINDOWS - 1)) for i in range(N_RECAL_WINDOWS)]
    with torch.no_grad():
        for prog, bnd, _ in batches(ds, idx):
            model.forward_once(prog, bnd, ds.ctx)
    model.eval()


def use_batch_stats(model):
    for m in model.modules():
        if isinstance(m, _BatchNorm):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
            m.num_batches_tracked = None


def one_step(model, ds, area, names):
    wet = ds.ctx.label_mask
    coast = coastal(wet)
    regions = {"coast": coast, "interior": wet & ~coast}
    se = {k: torch.zeros(len(names)) for k in regions}
    idx = list(range(len(ds)))
    with torch.no_grad():
        for prog, bnd, lab in batches(ds, idx):
            pred = model._assemble_prediction(prog, model.forward_once(prog, bnd, ds.ctx))
            err2 = ((pred - lab) ** 2).sum(0)
            for k, m in regions.items():
                se[k] += (err2 * m * area).sum((-2, -1))
    out = {}
    for k, m in regions.items():
        denom = (m * area).sum((-2, -1)) * len(idx)
        mse = se[k] / denom.clamp_min(1e-12)
        ok = denom > 0
        out[k] = {g: float(mse[[i for i, n in enumerate(names) if group(n) == g and ok[i]]].mean()) for g in sorted({group(n) for n in names})}
        out[k]["all"] = float(mse[ok].mean())
    return out


def rollout_360(model, cfg, layout, bundle):
    src = bundle.val_sources[0]
    ds = dataset(cfg, layout, src, long_rollout=True)
    spec = RolloutValidationSpec.from_day_horizon(
        days=360,
        start_time=src.time.values[0],
        target_times=ds.get_target_time(0, len(ds)).values,
        output_steps=cfg.data.output_steps,
    )
    agg = RolloutValidationAggregator(
        output_steps=cfg.data.output_steps,
        area_weights=bundle.train_sources[0].spherical_area_weights,
        preprocessor=BatchPreprocessor(
            bundle.train_sources[0],
            prognostic_var_names=layout.prognostic_var_names,
            boundary_var_names=layout.boundary_var_names,
        ),
        data_layout=layout,
        prognostic_var_names=layout.prognostic_var_names,
        distributed_reduce=False,
    )
    with torch.no_grad():
        validate_rollout(
            model=model,
            dataset=ds,
            aggregators_by_step={spec.model_steps: agg},
            epoch=40,
            num_model_steps=spec.model_steps,
            num_model_steps_forward=1,
        )
    logs = agg.get_logs(label="rollout_val/360d")
    keep = ["normalized_rmse/channel_mean"] + [
        f"weighted_rmse/{v}/depth_band/{b}" for v in ("thetao", "so", "uo", "vo") for b in ("upper", "middle", "deep")
    ] + ["weighted_rmse/zos"]
    return {k: float(logs[f"rollout_val/360d/{k}"]) for k in keep}


def main(out_path):
    results = {}
    for arm, run in ARMS.items():
        cfg = TrainConfig.from_yaml(f"{RUNS}/{run}/config.yaml")
        bundle = cfg.data.build(data_root=cfg.experiment.resolved_data_root, include_inference=False)
        layout = bundle.data_layout
        ni, no = cfg.data.input_steps, cfg.data.output_steps
        base = cfg.model.build(
            prog_channels=ni * len(layout.prognostic_var_names),
            boundary_channels=ni * len(layout.boundary_var_names),
            out_channels=no * len(layout.prognostic_var_names),
            input_steps=ni,
            grid_sizes=[s.grid_size for s in bundle.train_sources],
        )
        ck = torch.load(f"{RUNS}/{run}/saved_nets/{CKPT}", map_location="cpu")
        base.load_state_dict(OrderedDict((k.removeprefix("module."), v) for k, v in ck["model"].items()))
        live = {n: p.detach().clone() for n, p in base.named_parameters()}
        ema = {n: ck["ema"]["ema_params"][n.replace(".", "")] for n, _ in base.named_parameters()}
        val = dataset(cfg, layout, bundle.val_sources[0], long_rollout=False)
        train = dataset(cfg, layout, bundle.train_sources[0], long_rollout=False)
        area = bundle.val_sources[0].spherical_area_weights.float()
        names = list(layout.prognostic_var_names) * no

        variants = {
            "ema_running": lambda m: set_params(m, ema),
            "live_running": lambda m: set_params(m, live),
            "ema_recal": lambda m: (set_params(m, ema), recalibrate(m, train)),
            "ema_batchstat": lambda m: (set_params(m, ema), use_batch_stats(m)),
        }
        results[arm] = {"epoch": int(ck["epoch"])}
        for label, prepare in variants.items():
            model = copy.deepcopy(base).eval()
            prepare(model)
            model.eval()
            results[arm][label] = {
                "one_step_val": one_step(model, val, area, names),
                "rollout_360d": rollout_360(model, cfg, layout, bundle),
            }
            print(arm, label, json.dumps(results[arm][label]["rollout_360d"]), flush=True)
            json.dump(results, open(out_path, "w"), indent=1)
    print(json.dumps(results, indent=1))


main(sys.argv[1])
