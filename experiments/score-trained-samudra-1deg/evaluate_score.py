# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Common held-out forecast scores for reused A and new B/C/D checkpoints.

The table contains per-origin, per-lead, per-channel metrics; it does not rank
models by their incomparable training losses. All stochastic histories evolve
independently. Long-trajectory and observational diagnostics are separate tasks.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from multitask_model import MultitaskModelConfig
from native_pointwise import configure_native_pointwise
from score_model import crps, reduce_field, weights
from train_multitask import MultitaskTrainConfig
from train_score import ScoreTrainConfig

from samudra.backend import init_train_backend
from samudra.datasets import InferenceDataset


def field_metrics(pred, target, area):
    """Physical-unit conversion is applied to this normalized metric table."""
    mean = pred.mean(0)
    m = pred.shape[0]
    # Quantile intervals and the member range have distinct finite-M behavior.
    intervals = torch.quantile(
        pred, torch.tensor([0.05, 0.95], device=pred.device), dim=0
    )
    result = {
        "rmse": reduce_field((mean - target).square(), area).sqrt(),
        "bias": reduce_field(mean - target, area),
        "mae": reduce_field((mean - target).abs(), area),
        "crps_empirical": reduce_field(crps(pred, target, 0), area),
        "crps_fair": reduce_field(crps(pred, target, 1), area),
        "spread": reduce_field(pred.var(0, unbiased=m > 1), area).sqrt(),
        "member_rmse": torch.stack(
            [reduce_field((p - target).square(), area).sqrt() for p in pred]
        ).mean(0),
        "coverage_q05_q95": reduce_field(
            ((target >= intervals[0]) & (target <= intervals[1])).float(), area
        ),
        "coverage_member_range": reduce_field(
            ((target >= pred.min(0).values) & (target <= pred.max(0).values)).float(),
            area,
        ),
    }
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", choices=["A", "B", "C", "D", "F", "M"], required=True)
    parser.add_argument("--members", type=int, default=16)
    parser.add_argument("--origin-limit", type=int)
    parser.add_argument(
        "--checkpoint-role",
        choices=["final-ema", "final-raw", "validation-selected"],
        default="final-ema",
    )
    parser.add_argument("--forecast-cooldown", action="store_true")
    parser.add_argument(
        "--validation-epoch",
        type=int,
        help="Score this checkpoint epoch on validation only",
    )
    parser.add_argument("--forecast-adaptation", action="store_true")
    args = parser.parse_args()
    if args.forecast_adaptation and args.forecast_cooldown:
        parser.error("Choose cooldown or adaptation, not both")
    args.forecast_cooldown = args.forecast_cooldown or args.forecast_adaptation
    expected_epoch = (
        30 if args.forecast_adaptation else (10 if args.forecast_cooldown else 70)
    )
    if args.validation_epoch is not None:
        if args.validation_epoch < 1 or args.arm not in ("F", "M"):
            parser.error("validation-epoch requires a positive epoch and F/M arm")
        expected_epoch = args.validation_epoch
    if args.arm in ("C", "D", "F", "M") and args.members < 2:
        parser.error("Stochastic evaluation requires at least two members")
    if args.origin_limit is not None and args.origin_limit < 1:
        parser.error("--origin-limit must be positive")
    config_type: type[ScoreTrainConfig] | type[MultitaskTrainConfig] = (
        MultitaskTrainConfig if args.arm in ("F", "M") else ScoreTrainConfig
    )
    cfg = config_type.from_yaml(args.config)
    if args.arm in ("F", "M"):
        expected_mode = (
            "forecast"
            if args.forecast_cooldown or args.validation_epoch is not None
            else {"F": "forecast", "M": "multitask"}[args.arm]
        )
        assert isinstance(cfg.model, MultitaskModelConfig)
        if cfg.model.mode != expected_mode:
            raise ValueError(f"Arm {args.arm} requires mode={expected_mode}")
        assert isinstance(cfg, MultitaskTrainConfig)
        configure_native_pointwise(cfg.native_pointwise)
    if args.forecast_cooldown:
        if (
            args.arm not in ("F", "M")
            or not cfg.finetune
            or cfg.epochs != expected_epoch
        ):
            raise ValueError(
                f"Forecast adaptation evaluation requires an epoch-{expected_epoch} F/M finetune config"
            )
    raw = cfg.model_dump(mode="json")
    raw["data"]["sources"][0]["inference_time"] = {
        "start": "2014-10-10",
        "end": "2022-12-24",
    }
    if args.validation_epoch is not None:
        raw["data"]["sources"][0]["inference_time"] = raw["data"]["sources"][0][
            "val_time"
        ]
    cfg = config_type(**raw)
    device, dist = init_train_backend("auto")
    rank = (dist.rank or 0) if dist else 0
    world = (dist.world_size or 1) if dist else 1
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob("metrics-rank*.csv")):
        raise FileExistsError(
            "Use a fresh output directory to avoid mixing evaluations"
        )
    bundle = cfg.data.build(
        data_root=cfg.experiment.resolved_data_root, include_inference=True
    )
    source = bundle.inference_source
    if source is None:
        raise ValueError("Missing held-out source")
    names = list(bundle.data_layout.prognostic_var_names)
    boundary_names = bundle.data_layout.boundary_var_names
    dataset = InferenceDataset(
        source,
        names,
        boundary_names,
        2,
        2,
        cfg.data.normalize_before_mask,
        cfg.data.masked_fill_value,
        False,
    ).to(device)
    net = (
        cfg.model.build(
            prog_channels=154,
            boundary_channels=6,
            out_channels=154,
            input_steps=2,
            grid_sizes=[source.grid_size],
        )
        .to(device)
        .eval()
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    checkpoint_epoch = checkpoint["epoch"]
    if (
        args.checkpoint_role != "validation-selected"
        and checkpoint_epoch != expected_epoch
    ):
        raise ValueError(
            f"Expected final epoch {expected_epoch}, got {checkpoint_epoch}"
        )
    if not 1 <= checkpoint_epoch <= expected_epoch:
        raise ValueError(f"Unexpected checkpoint epoch {checkpoint_epoch}")
    expected_file = {
        "final-ema": "ema_ckpt.pt",
        "final-raw": f"ckpt_{expected_epoch}.pt",
        "validation-selected": "best_validation_ckpt.pt",
    }[args.checkpoint_role]
    if Path(args.checkpoint).name != expected_file:
        raise ValueError(f"{args.checkpoint_role} requires {expected_file}")
    selection_identity = checkpoint.get("validation_checkpoint_identity")
    if args.checkpoint_role == "validation-selected" and not selection_identity:
        raise ValueError("Validation-selected checkpoint lacks selection provenance")
    state = {k.removeprefix("module."): v for k, v in checkpoint["model"].items()}
    if args.arm == "A":
        net.base.load_state_dict(state, strict=True)
    else:
        net.load_state_dict(state, strict=True)
    del checkpoint, state
    m = 1 if args.arm in ("A", "B") else args.members
    std = source.statistics(names).std.reshape(-1)
    origins = [
        i
        for i in range(0, len(dataset) - 36 + 1, 9)
        if str(source.time.values[2 * i + 1])[:4] <= "2021"
    ]
    if args.validation_epoch is not None:
        origins = [0] if len(dataset) >= 36 else []
    if args.origin_limit:
        origins = origins[: args.origin_limit]
    if not origins:
        raise ValueError("No complete 360-day forecast origins")
    run = None
    if rank == 0:
        with open(args.checkpoint, "rb") as f:
            checkpoint_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
        cfg.save_yaml(args.output / "config.yaml")
        run = wandb.init(
            entity=cfg.experiment.wandb.entity,
            project=cfg.experiment.wandb.project,
            name=f"score-{args.arm}-heldout-{args.checkpoint_role}",
            mode="online",
            config={
                "arm": args.arm,
                "members": m,
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": checkpoint_epoch,
                "checkpoint_role": args.checkpoint_role,
                "validation_selection_identity": selection_identity,
                "checkpoint_sha256": checkpoint_sha256,
                "source_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
                "origins": origins,
                "seed": 26000000 if args.validation_epoch is not None else 35000000,
                "evaluation_split": "validation"
                if args.validation_epoch is not None
                else "heldout",
                "test_period": ["2014-10-10", "2022-12-24"],
            },
        )
        print(f"Online evaluation: {run.url}", flush=True)
        (args.output / "manifest.json").write_text(
            json.dumps(
                {
                    "wandb_url": run.url,
                    "arm": args.arm,
                    "checkpoint": str(args.checkpoint),
                    "checkpoint_epoch": checkpoint_epoch,
                    "checkpoint_role": args.checkpoint_role,
                    "validation_selection_identity": selection_identity,
                    "checkpoint_sha256": checkpoint_sha256,
                    "source_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
                    "container": os.environ.get("SAMUDRA_CONTAINER_IMAGE_REF"),
                    "slurm_job": os.environ.get("SLURM_JOB_ID"),
                    "data_root": str(cfg.experiment.resolved_data_root),
                    "variables": names,
                    "forcing": list(boundary_names),
                    "normalization_std": std.tolist(),
                    "members": m,
                    "origin_indices": origins,
                    "origin_times": [
                        str(source.time.values[2 * i + 1]) for i in origins
                    ],
                    "leads_days": [5, 10, 20, 40, 90, 180, 360],
                    "finite_member_range_reference": (m - 1) / (m + 1),
                    "rank_ties": "uniform random among equal member values",
                },
                indent=2,
            )
            + "\n"
        )
    if dist:
        torch.distributed.barrier()  # Require online W&B before any rank forecasts.
    rows = []
    histogram = torch.zeros(7, 77, m + 1, device=device)
    leads = [5, 10, 20, 40, 90, 180, 360]
    area = weights(dataset.ctx, 77, area=True)
    area = area / area.sum((-2, -1), keepdim=True)
    for origin in origins[rank::world]:
        torch.manual_seed(
            (26000000 if args.validation_epoch is not None else 35000000) + origin
        )
        initial = dataset[origin][0].to(device)
        history = initial[None].expand(m, *initial.shape)
        for step in range(36):
            boundary = dataset.get_boundary(origin + step).to(device)
            pred = net.sample_step(history, boundary, dataset.ctx)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(
                    f"Nonfinite prediction: origin={origin} step={step}"
                )
            history = pred
            for frame in (0, 1):
                lead = 10 * step + 5 * (frame + 1)
                if lead not in leads:
                    continue
                target = dataset.inference_target(origin + step).to(device)[
                    :, frame * 77 : (frame + 1) * 77
                ]
                p = pred[:, :, frame * 77 : (frame + 1) * 77]
                metrics = {
                    k: v.cpu().numpy()
                    for k, v in field_metrics(p, target, area).items()
                }
                less = (p < target).sum(0)
                equal = (p == target).sum(0)
                # Separate tie RNG: diagnostic bookkeeping must not change forecasts.
                generator = torch.Generator(device=device).manual_seed(
                    45000000 + origin * 100 + lead
                )
                tie = torch.floor(
                    torch.rand(target.shape, device=device, generator=generator)
                    * (equal + 1)
                ).long()
                ranks = less + tie
                for r in range(m + 1):
                    histogram[leads.index(lead), :, r] += reduce_field(
                        (ranks == r).float(), area
                    )
                target_time = str(source.time.values[2 * (origin + step) + 2 + frame])
                for channel, name in enumerate(names):
                    row = {
                        "arm": args.arm,
                        "origin": str(source.time.values[2 * origin + 1]),
                        "target": target_time,
                        "lead_days": lead,
                        "variable": name,
                        "members": m,
                    }
                    for key, values in metrics.items():
                        row[
                            f"{key}_normalized"
                            if not key.startswith("coverage")
                            else key
                        ] = float(values[channel])
                        if not key.startswith("coverage"):
                            row[f"{key}_physical"] = float(
                                values[channel] * std[channel]
                            )
                    rows.append(row)
                if origin == origins[0] and lead in (5, 40, 360):
                    # Prespecified channels/date/members; 180x360 native grid retained.
                    selected = [
                        names.index(v) for v in ("zos", "thetao_0", "so_9", "uo_9")
                    ]
                    np.savez_compressed(
                        args.output / f"fields-{lead:03d}d.npz",
                        variables=np.array([names[i] for i in selected]),
                        target=target[0, selected].cpu().numpy(),
                        mean=p.mean(0)[0, selected].cpu().numpy(),
                        member0=p[0, 0, selected].cpu().numpy(),
                        member1=p[min(1, m - 1), 0, selected].cpu().numpy(),
                        spread=p.std(0, unbiased=m > 1)[0, selected].cpu().numpy(),
                        mask=dataset.ctx.label_mask[selected].cpu().numpy(),
                    )
        print(f"rank={rank} completed origin={origin} members={m}", flush=True)
        # A per-rank atomic replacement preserves completed origins after failure.
        path = args.output / f"metrics-rank{rank}.csv"
        temporary = path.with_suffix(".tmp")
        with temporary.open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    if dist:
        torch.distributed.all_reduce(histogram)
        torch.distributed.barrier()
    if rank == 0:
        np.savez_compressed(
            args.output / "rank-histograms.npz",
            counts=histogram.cpu().numpy(),
            variables=np.array(names),
            leads=np.array(leads),
            origins=len(origins),
        )
        assert run is not None
        all_rows: list[dict[str, str]] = []
        for path in sorted(args.output.glob("metrics-rank*.csv")):
            with path.open() as f:
                all_rows.extend(csv.DictReader(f))
        expected_rows = len(origins) * len(leads) * len(names)
        if len(all_rows) != expected_rows:
            raise ValueError(
                f"Expected {expected_rows} metric rows, got {len(all_rows)}"
            )
        for lead in leads:
            selected_rows = [r for r in all_rows if int(r["lead_days"]) == lead]
            run.log(
                {
                    "lead_days": lead,
                    **{
                        name: float(np.mean([float(r[name]) for r in selected_rows]))
                        for name in (
                            "rmse_normalized",
                            "crps_empirical_normalized",
                            "crps_fair_normalized",
                            "spread_normalized",
                            "coverage_q05_q95",
                            "coverage_member_range",
                        )
                    },
                }
            )
        run.finish()
        (args.output / "COMPLETE.json").write_text(
            json.dumps({"rows": len(all_rows), "origins": len(origins), "members": m})
            + "\n"
        )
    if dist:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
