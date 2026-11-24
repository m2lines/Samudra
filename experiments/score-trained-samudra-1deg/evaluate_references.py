# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Persistence and training-only five-day seasonal climatology on annual origins."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from evaluate_score import field_metrics
from score_model import weights
from train_score import ScoreTrainConfig

from samudra.backend import init_train_backend
from samudra.datasets import InferenceDataset


def seasonal_bin(time):
    """The source has 73 five-day centers per Julian-calendar year.

    Centers are day 3, 8, ..., 363, including leap years. The additional leap
    day appears as a six-day last-to-first-center interval, not an extra bin.
    """
    day = time.dayofyr
    if day < 3 or day > 363 or (day - 3) % 5:
        raise ValueError(f"Unexpected OM4 seasonal center: {time}")
    return (day - 3) // 5


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    device, dist = init_train_backend("nccl")
    if dist is None or dist.rank is None or dist.world_size != 4:
        raise ValueError("Use four ranks for distributed data aggregation")
    rank, world = dist.rank, dist.world_size
    cfg = ScoreTrainConfig.from_yaml(args.campaign / "configs/train-B.yaml")
    raw = cfg.model_dump(mode="json")
    raw["data"]["sources"][0]["inference_time"] = {
        "start": "2014-10-10",
        "end": "2022-12-24",
    }
    cfg = ScoreTrainConfig(**raw)
    bundle = cfg.data.build(
        data_root=cfg.experiment.resolved_data_root, include_inference=True
    )
    train, source = bundle.train_sources[0], bundle.inference_source
    if source is None:
        raise ValueError("No held-out source")
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
    contract = json.loads(
        (args.campaign / "outputs/eval-full-A-212403/manifest.json").read_text()
    )
    origins, leads = contract["origin_indices"], contract["leads_days"]
    assert names == contract["variables"]
    assert [str(source.time.values[2 * i + 1]) for i in origins] == contract[
        "origin_times"
    ]
    assert str(train.time.values[0]) == "1975-01-03 12:00:00"
    assert str(train.time.values[-1]) == "2013-10-05 12:00:00"
    bins = np.array([seasonal_bin(t) for t in train.time.values])
    run = None
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        run = wandb.init(
            entity=cfg.experiment.wandb.entity,
            project=cfg.experiment.wandb.project,
            name="score-reference-annual",
            mode="online",
            dir=str(args.output),
            config={
                "train_start": str(train.time.values[0]),
                "train_end": str(train.time.values[-1]),
                "source_commit": os.environ["SAMUDRA_CODE_COMMIT"],
                "origins": origins,
            },
        )
        print(f"Online references: {run.url}", flush=True)
    torch.distributed.barrier()
    wet = dataset.ctx.label_mask[:77].bool()
    assert torch.equal(wet.cpu(), train.masks.prognostic.bool().cpu())
    sums = torch.zeros(73, 77, *source.grid_size, device=device, dtype=torch.float64)
    counts = torch.zeros(73, device=device, dtype=torch.int64)
    indices = np.arange(rank, len(train.time), world)
    for offset in range(0, len(indices), 16):
        ids = indices[offset : offset + 16]
        values = torch.from_numpy(train.read(ids, names)).to(device)
        values = torch.where(wet, values, 0).double()
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite wet training value")
        phases = torch.from_numpy(bins[ids]).to(device)
        sums.index_add_(0, phases, values)
        counts.index_add_(
            0, phases, torch.ones(len(ids), device=device, dtype=torch.int64)
        )
        if rank == 0 and offset % 160 == 0:
            print(
                f"Climatology: local samples {min(offset + 16, len(indices))}/{len(indices)}",
                flush=True,
            )
    torch.distributed.all_reduce(sums)
    torch.distributed.all_reduce(counts)
    assert counts.min() > 0 and int(counts.sum()) == len(train.time)
    climatology = (sums / counts[:, None, None, None]).float()
    del sums
    if rank == 0:
        np.save(args.output / "climatology-physical.npy", climatology.cpu().numpy())
    stats = source.statistics(names)
    mean = torch.as_tensor(stats.mean, device=device).reshape(77, 1, 1)
    std = torch.as_tensor(stats.std, device=device).reshape(77, 1, 1)
    np.testing.assert_array_equal(
        std.flatten().cpu().numpy(), contract["normalization_std"]
    )
    normalized = (climatology - mean) / std * wet
    area = weights(dataset.ctx, 77, area=True)
    rows = []
    for origin in origins[rank::world]:
        persistence = dataset[origin][0].to(device)[:, -77:]
        for lead in leads:
            step, frame = divmod(lead // 5 - 1, 2)
            time = source.time.values[2 * (origin + step) + 2 + frame]
            target = dataset.inference_target(origin + step).to(device)[
                :, frame * 77 : (frame + 1) * 77
            ]
            season = normalized[seasonal_bin(time)][None]
            for arm, prediction in (("P", persistence), ("S", season)):
                metrics = {
                    k: v.cpu().numpy()
                    for k, v in field_metrics(prediction[None], target, area).items()
                }
                for channel, name in enumerate(names):
                    row = {
                        "arm": arm,
                        "origin": str(source.time.values[2 * origin + 1]),
                        "target": str(time),
                        "lead_days": lead,
                        "variable": name,
                        "members": 1,
                    }
                    for key, values in metrics.items():
                        row[
                            key if key.startswith("coverage") else f"{key}_normalized"
                        ] = float(values[channel])
                        if not key.startswith("coverage"):
                            row[f"{key}_physical"] = float(values[channel]) * float(
                                std[channel]
                            )
                    rows.append(row)
        print(f"rank={rank} references completed origin={origin}", flush=True)
    with (args.output / f"metrics-rank{rank}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    torch.distributed.barrier()
    if rank == 0:
        assert run is not None
        all_rows: list[dict[str, str]] = []
        for file in args.output.glob("metrics-rank*.csv"):
            with file.open() as stream:
                all_rows.extend(csv.DictReader(stream))
        assert len(all_rows) == 2 * len(origins) * len(leads) * len(names)
        manifest = {
            "wandb_url": run.url,
            "source_commit": os.environ["SAMUDRA_CODE_COMMIT"],
            "slurm_job": os.environ["SLURM_JOB_ID"],
            "container": os.environ["SAMUDRA_CONTAINER_IMAGE_REF"],
            "data_root": str(cfg.experiment.resolved_data_root),
            "train_first": str(train.time.values[0]),
            "train_last": str(train.time.values[-1]),
            "train_count": len(train.time),
            "seasonal_bin_counts": counts.cpu().tolist(),
            "seasonal_definition": "(Julian day of year - 3) // 5",
            "variables": names,
            "normalization_mean": mean.flatten().cpu().tolist(),
            "normalization_std": std.flatten().cpu().tolist(),
            "origin_times": contract["origin_times"],
            "origin_indices": origins,
            "leads_days": leads,
            "arms": {
                "P": "latest-input persistence",
                "S": "training-only seasonal climatology",
            },
        }
        (args.output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        for arm in ("P", "S"):
            for lead in leads:
                selected = [
                    r
                    for r in all_rows
                    if r["arm"] == arm and int(r["lead_days"]) == lead
                ]
                run.log(
                    {
                        "lead_days": lead,
                        f"{arm}/rmse": np.mean(
                            [float(r["rmse_normalized"]) for r in selected]
                        ),
                    }
                )
        run.finish()
        (args.output / "COMPLETE.json").write_text(
            json.dumps({"rows": len(all_rows), "origins": len(origins)}) + "\n"
        )
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
