# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Fixed held-out two-time completion; score requested entries only."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from evaluate_score import field_metrics
from multitask_model import task_masks
from native_pointwise import configure_native_pointwise
from score_model import reduce_field, weights
from train_multitask import MultitaskTrainConfig

from samudra.backend import init_train_backend
from samudra.datasets import InferenceDataset


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arm", choices=["F", "M"], required=True)
    parser.add_argument("--origin-limit", type=int)
    parser.add_argument("--forecast-cooldown", action="store_true")
    args = parser.parse_args()
    expected_epoch = 10 if args.forecast_cooldown else 70
    if args.origin_limit is not None and args.origin_limit < 1:
        parser.error("origin-limit must be positive")
    cfg = MultitaskTrainConfig.from_yaml(args.config)
    assert cfg.model.mode == (
        "forecast"
        if args.forecast_cooldown
        else {"F": "forecast", "M": "multitask"}[args.arm]
    )
    configure_native_pointwise(cfg.native_pointwise)
    if args.forecast_cooldown:
        if args.arm not in ("F", "M") or not cfg.finetune or cfg.epochs != 10:
            raise ValueError(
                "Cooldown evaluation requires a ten-epoch F/M finetune config"
            )
    raw = cfg.model_dump(mode="json")
    raw["data"]["sources"][0]["inference_time"] = {
        "start": "2014-10-10",
        "end": "2022-12-24",
    }
    cfg = MultitaskTrainConfig(**raw)
    device, dist = init_train_backend("auto")
    if dist is not None:
        raise ValueError("Use a direct single process")
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.csv").exists():
        raise FileExistsError("Use a fresh output directory")
    bundle = cfg.data.build(
        data_root=cfg.experiment.resolved_data_root, include_inference=True
    )
    source = bundle.inference_source
    assert source is not None
    names = list(bundle.data_layout.prognostic_var_names)
    dataset = InferenceDataset(
        source,
        names,
        bundle.data_layout.boundary_var_names,
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
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    assert ckpt["epoch"] == expected_epoch and args.checkpoint.name == "ema_ckpt.pt"
    net.load_state_dict(
        {k.removeprefix("module."): v for k, v in ckpt["model"].items()}, strict=True
    )
    del ckpt
    with args.checkpoint.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    origins = [
        i
        for i in range(0, len(dataset) - 36 + 1, 9)
        if str(source.time.values[2 * i + 1])[:4] <= "2021"
    ]
    if args.origin_limit:
        origins = origins[: args.origin_limit]
    assert origins
    cfg.save_yaml(args.output / "config.yaml")
    manifest = dict(
        arm=args.arm,
        origins=origins,
        members=16,
        checkpoint=str(args.checkpoint),
        checkpoint_epoch=expected_epoch,
        checkpoint_sha256=digest,
        source_commit=os.environ["SAMUDRA_CODE_COMMIT"],
        variables=names,
        tasks=["surface", "variable", "gap"],
        mask_seed=61000000,
        sample_seed=62000000,
        protocol="Same thirty forecast origins; one two-date completion per task; no history; score requested wet entries only",
    )
    run = wandb.init(
        entity=cfg.experiment.wandb.entity,
        project=cfg.experiment.wandb.project,
        name=f"score-{args.arm}-heldout-completion",
        mode="online",
        dir=str(args.output),
        config=manifest,
    )
    print(f"Online completion evaluation: {run.url}", flush=True)
    manifest["wandb_url"] = run.url
    std = source.statistics(names).std.reshape(-1)
    wet = dataset.ctx.label_mask[:154].bool()
    area = weights(dataset.ctx, 154, area=True)
    rows, audit = [], []
    histogram = torch.zeros(3, 154, 17, device=device)
    active_counts = torch.zeros(3, 154, device=device)
    for origin in origins:
        initial = dataset[origin][0].to(device)
        history = torch.zeros_like(initial)[None].expand(16, *initial.shape)
        target = dataset.inference_target(origin).to(device)
        boundary = dataset.get_boundary(origin).to(device)
        for task_index, task in enumerate(manifest["tasks"]):
            seed = 61000000 + origin * 10 + task_index
            history_known, known, request = task_masks(
                task, wet, torch.Generator().manual_seed(seed)
            )
            torch.manual_seed(62000000 + origin * 10 + task_index)
            pred = net.complete(
                history, boundary, dataset.ctx, target, history_known, known, request
            )
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"Nonfinite {origin=} {task=}")
            assert torch.equal(
                pred.masked_select(known[None, None]),
                target[None].expand_as(pred).masked_select(known[None, None]),
            )
            requested_area = area * request
            active = requested_area.sum((-2, -1)) > 0
            assert active.any()
            scores = {
                k: v.cpu().numpy()
                for k, v in field_metrics(pred, target, requested_area).items()
            }
            less, equal = (pred < target).sum(0), (pred == target).sum(0)
            generator = torch.Generator(device=device).manual_seed(
                63000000 + origin * 10 + task_index
            )
            ranks = (
                less
                + torch.floor(
                    torch.rand(target.shape, device=device, generator=generator)
                    * (equal + 1)
                ).long()
            )
            for rank in range(17):
                histogram[task_index, :, rank] += reduce_field(
                    (ranks == rank).float(), requested_area
                )
            active_counts[task_index] += active
            channels = active.nonzero().flatten().tolist()
            audit.append(
                dict(origin=origin, task=task, seed=seed, active_channels=channels)
            )
            for ch in channels:
                frame, channel = divmod(ch, 77)
                row = dict(
                    arm=args.arm,
                    origin=str(source.time.values[2 * origin + 1]),
                    target=str(source.time.values[2 * origin + 2 + frame]),
                    task=task,
                    frame=frame,
                    variable=names[channel],
                    members=16,
                    requested_cells=int(request[ch].sum()),
                )
                for key, value in scores.items():
                    row[key if key.startswith("coverage") else key + "_normalized"] = (
                        float(value[ch])
                    )
                    if not key.startswith("coverage"):
                        row[key + "_physical"] = float(value[ch] * std[channel])
                rows.append(row)
            np.savez_compressed(
                args.output / f"mask-{origin:03d}-{task}.npz",
                request=request.cpu().numpy(),
                known=known.cpu().numpy(),
            )
            if origin == origins[0]:
                selected = [76, 38, 66, 9, 153, 115, 143, 86]
                np.savez_compressed(
                    args.output / f"fields-{task}.npz",
                    channels=selected,
                    target=target[0, selected].cpu().numpy(),
                    mean=pred.mean(0)[0, selected].cpu().numpy(),
                    member0=pred[0, 0, selected].cpu().numpy(),
                    member1=pred[1, 0, selected].cpu().numpy(),
                    spread=pred.std(0)[0, selected].cpu().numpy(),
                    request=request[selected].cpu().numpy(),
                )
        with (args.output / "metrics.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Completed origin {origin}", flush=True)
    assert len(rows) == int(active_counts.sum())
    assert all(
        np.isfinite(v)
        for row in rows
        for v in row.values()
        if isinstance(v, (float, int))
    )
    assert torch.allclose(histogram.sum(-1), active_counts, atol=1e-4)
    np.savez_compressed(
        args.output / "rank-histograms.npz",
        counts=histogram.cpu().numpy(),
        active_counts=active_counts.cpu().numpy(),
    )
    manifest["mask_audit"] = audit
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for task in manifest["tasks"]:
        selected_rows = [r for r in rows if r["task"] == task]
        run.log(
            {
                f"{task}/{key}": float(np.mean([r[key] for r in selected_rows]))
                for key in (
                    "rmse_normalized",
                    "crps_empirical_normalized",
                    "coverage_q05_q95",
                )
            }
        )
    run.finish()
    (args.output / "COMPLETE.json").write_text(
        json.dumps(dict(rows=len(rows), origins=len(origins), members=16, tasks=3))
        + "\n"
    )


if __name__ == "__main__":
    main()
