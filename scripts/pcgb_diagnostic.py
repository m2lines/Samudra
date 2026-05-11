#!/usr/bin/env python
# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Free PCGB diagnostic — gate the 12 GPU-hour training run on per-mask val MSE.

Loads a pretrained Samudra checkpoint, evaluates every skip-mask in the
PCGB pool on the validation period, and reports the per-mask MSE table.
The spread `(max - min) / mean` across masks tells you whether PCGB has
any slack to redistribute:

  spread < 5%   → lattice saturated, PCGB unlikely to recover anything
  spread 5–20%  → ambiguous; PCGB may help marginally
  spread > 20%  → non-trivial spread; PCGB has signal to fit

Usage:
    python scripts/pcgb_diagnostic.py \
        configs/samudra_om4_v2/boosted_pcgb.yaml \
        --resume_ckpt_path=/scratch/.../samudra_om4_v2_drop_path_new_data/saved_nets/ckpt.pt

The config supplies model architecture, data sources, and val_time. The
ckpt path can also be set inside the YAML; CLI override wins.

Single-GPU, ~30 min on 1° data with the default 16-mask enumeration.
"""

from __future__ import annotations

import json
import logging
import sys
from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

from ocean_emulators.backend import init_eval_backend
from ocean_emulators.config import PCGBConfig
from ocean_emulators.constants import BoundaryVarNames, PrognosticVarNames, TensorMap
from ocean_emulators.datasets import TorchTrainDataset, TrainDataLoader
from ocean_emulators.models import Samudra
from ocean_emulators.pcgb import EnumerateSearcher, PathMask, _per_sample_masked_mse
from ocean_emulators.utils.data import Normalize
from ocean_emulators.utils.distributed import set_seed
from ocean_emulators.utils.logging import handle_logging, handle_warnings
from ocean_emulators.utils.train import collate_raw_train_data

logger = logging.getLogger(__name__)


SATURATED_THRESHOLD = 0.05
SLACK_THRESHOLD = 0.20


def build_val_loader(cfg: PCGBConfig, dataset_spec, prog_names, bound_names, device):
    """Single-step val loader over cfg.val_time. Sequential — no shuffle."""
    src = cfg.data.build(cfg.experiment.resolved_data_root).primary_source.slice(
        cfg.val_time
    )
    ds = TorchTrainDataset(
        src=src,
        dst=None,
        prognostic_var_names=prog_names,
        boundary_var_names=bound_names,
        hist=cfg.data.hist,
        steps=1,
        normalize_before_mask=cfg.data.normalize_before_mask,
        masked_fill_value=cfg.data.masked_fill_value,
        stride=1,
        concurrent_compute_=cfg.data.concurrent_compute,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,  # single-pass diagnostic; not worth the worker overhead
        pin_memory=False,
        collate_fn=collate_raw_train_data,
        drop_last=False,
    )
    return TrainDataLoader(loader, [ds], device), len(ds)


def load_model_and_ckpt(cfg: PCGBConfig, device, data_container):
    """Build Samudra from config and load weights from cfg.resume_ckpt_path."""
    if cfg.resume_ckpt_path is None:
        raise ValueError(
            "Diagnostic requires resume_ckpt_path to be set (in the YAML or "
            "via --resume_ckpt_path on the CLI)."
        )

    n_prog = len(data_container.dataset_spec.prognostic_var_names)
    n_bound = len(data_container.dataset_spec.boundary_var_names)
    num_prog_in = (cfg.data.hist + 1) * n_prog
    num_boundary_in = (cfg.data.hist + 1) * n_bound

    tensor_map = TensorMap(dataset_spec=data_container.dataset_spec).to(device)
    normalize = Normalize(
        data_container.primary_source,
        prognostic_var_names=data_container.dataset_spec.prognostic_var_names,
        boundary_var_names=data_container.dataset_spec.boundary_var_names,
    )
    model = cfg.model.build(
        prog_channels=num_prog_in,
        boundary_channels=num_boundary_in,
        out_channels=num_prog_in,
        hist=cfg.data.hist,
        static_data_for_corrector=data_container.static_data,
        srcs=data_container.sources,
        tensor_map=tensor_map,
        normalize=normalize,
        dataset_spec=data_container.dataset_spec,
    ).to(device)
    if not isinstance(model, Samudra):
        raise TypeError(
            f"Diagnostic requires a Samudra model; got {type(model).__name__}"
        )

    logger.info(f"Loading checkpoint from {cfg.resume_ckpt_path}")
    ckpt = torch.load(cfg.resume_ckpt_path, map_location=device)
    state_dict = OrderedDict(
        (k.removeprefix("module."), v) for k, v in ckpt["model"].items()
    )
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def score_mask(
    model: Samudra,
    val_loader: TrainDataLoader,
    mask: PathMask,
    device: torch.device,
) -> float:
    """Mean per-sample MSE on val under the given path mask."""
    backbone = model.unet
    skip_drops = mask.skip_drops
    block_drops = mask.block_drops if mask.block_drops else None

    sq_sum = 0.0
    count = 0
    with backbone.with_path_mask(skip_drops=skip_drops, block_drops=block_drops):
        for data in val_loader:
            prog, boundary = data.get_initial_input()
            label = data.get_label(0)
            decodings = model.forward_once(prog, boundary, data.ctx)
            pred = prog + decodings if model.pred_residuals else decodings
            per_sample = _per_sample_masked_mse(pred, label, data.ctx.label_mask)
            sq_sum += float(per_sample.sum(dtype=torch.float64).item())
            count += per_sample.shape[0]
    return sq_sum / max(count, 1)


def format_table(rows: list[dict]) -> str:
    """Markdown table sorted by val MSE (best first)."""
    sorted_rows = sorted(rows, key=lambda r: r["val_mse"])
    lines = [
        "| Mask | Val MSE | (MSE - min) / min |",
        "| --- | --- | --- |",
    ]
    best = sorted_rows[0]["val_mse"]
    for r in sorted_rows:
        rel = (r["val_mse"] - best) / max(best, 1e-12)
        lines.append(f"| {r['mask']} | {r['val_mse']:.6g} | {rel:+.2%} |")
    return "\n".join(lines)


def verdict(rows: list[dict]) -> tuple[str, float]:
    mses = [r["val_mse"] for r in rows]
    spread = (max(mses) - min(mses)) / max(sum(mses) / len(mses), 1e-12)
    if spread < SATURATED_THRESHOLD:
        recommendation = (
            f"LATTICE SATURATED (spread {spread:.1%} < {SATURATED_THRESHOLD:.0%}). "
            f"PCGB unlikely to recover slack that isn't there — recommend "
            f"pivoting before committing the 12 GPU-hour training run."
        )
    elif spread < SLACK_THRESHOLD:
        recommendation = (
            f"AMBIGUOUS (spread {spread:.1%}, between {SATURATED_THRESHOLD:.0%} "
            f"and {SLACK_THRESHOLD:.0%}). PCGB may help marginally; consider "
            f"running a short ablation first (e.g. 10 rounds) before the full T=70 run."
        )
    else:
        recommendation = (
            f"NON-TRIVIAL SLACK (spread {spread:.1%} > {SLACK_THRESHOLD:.0%}). "
            f"PCGB has signal to fit. Green-light the full training run."
        )
    return recommendation, spread


def main() -> None:
    cfg = PCGBConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    device = init_eval_backend(cfg.backend)
    set_seed(cfg.experiment.rand_seed)

    data_container = cfg.data.build(cfg.experiment.resolved_data_root)
    prog_names: PrognosticVarNames = data_container.dataset_spec.prognostic_var_names
    bound_names: BoundaryVarNames = data_container.dataset_spec.boundary_var_names

    model = load_model_and_ckpt(cfg, device, data_container)
    val_loader, n_val = build_val_loader(
        cfg, data_container.dataset_spec, prog_names, bound_names, device
    )
    logger.info(f"Diagnostic: {n_val} val samples over {cfg.val_time}")

    # The diagnostic uses the same searcher the training run will use — so
    # we test exactly the pool PCGB would cycle through.
    searcher = EnumerateSearcher(num_skips=cfg.mask_searcher.num_skips)
    masks = searcher.candidates_for_round(0)
    logger.info(f"Diagnostic: scoring {len(masks)} masks")

    rows: list[dict] = []
    for i, mask in enumerate(masks):
        mse = score_mask(model, val_loader, mask, device)
        rows.append({"mask": str(mask), "val_mse": mse})
        logger.info(f"  [{i + 1}/{len(masks)}] {mask}: val_mse={mse:.6g}")

    table = format_table(rows)
    recommendation, spread = verdict(rows)

    summary = {
        "ckpt_path": cfg.resume_ckpt_path,
        "val_time": {"start": str(cfg.val_time.start), "end": str(cfg.val_time.end)},
        "n_val_samples": n_val,
        "rows": rows,
        "spread": spread,
        "recommendation": recommendation,
    }

    out_path = cfg.experiment.output_dir / "pcgb_diagnostic.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print("\n" + table + "\n")
    print(f"Recommendation: {recommendation}\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Diagnostic failed")
        sys.exit(1)
