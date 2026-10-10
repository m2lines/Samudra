# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Matched streaming forecast scores; no DynamicLoss weights or checkpoint selection."""

import argparse
import datetime
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from samudra.config import EvalConfig, Om4TimeConfig, SamudraConfig
from samudra.datasets import InferenceDataset
from samudra.eval import Eval
from samudra.utils.writer import ZarrWriter

LEADS = [5, 30, 90, 180, 360]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def score(evaluator, dataset, output, *, continuous):
    """Return per-date/channel normalized MSE and normalized signed errors at leads."""
    dataset.to(evaluator.device)
    nchan = evaluator.N_prog
    wet = evaluator.source.masks.prognostic.to(evaluator.device).bool()
    weights = evaluator.area_weights * wet
    weights = weights / weights.sum((-2, -1), keepdim=True)
    total_sq = torch.zeros_like(weights, dtype=torch.float64)
    total_error = torch.zeros_like(total_sq)
    scalar_mse = []
    times: list[str] = []
    lead_errors = {}
    prognostic = dataset.initial_prognostic
    writer = None
    if continuous:
        writer = ZarrWriter(
            output,
            dataset.get_coords_dict(),
            evaluator.output_steps,
            evaluator.model_path,
            10,
            evaluator.preprocessor,
            evaluator.data_layout,
        )
    count = 0
    for step in range(0, len(dataset), 5):
        result = evaluator.model.inference(
            dataset,
            prognostic,
            steps_completed=step,
            num_steps=min(5, len(dataset) - step),
            epoch=0,
        )
        prognostic = result.final_prognostic.clone()
        error = (result.prediction - result.target).reshape(-1, nchan, *wet.shape[-2:])
        if not torch.isfinite(error[:, wet]).all():
            raise ValueError("Non-finite forecast on wet support")
        total_sq += error.double().square().sum(0)
        total_error += error.double().sum(0)
        scalar_mse.append(
            (error.double().square() * weights).sum((-2, -1)).cpu().numpy()
        )
        times.extend(str(t) for t in result.time.values)
        for local in range(error.shape[0]):
            lead = (count + local + 1) * 5
            if lead in LEADS:
                lead_errors[lead] = error[local].cpu().numpy()
        count += error.shape[0]
        if writer:
            writer.record_batch(result)
            writer.write()
        del result, error
    np.savez_compressed(
        output / "scores.npz",
        mse=np.concatenate(scalar_mse),
        times=np.asarray(times),
        mean_squared_error=(total_sq / count).cpu().numpy().astype("float32"),
        mean_error=(total_error / count).cpu().numpy().astype("float32"),
    )
    if continuous:
        return np.empty(0), times
    return np.stack([lead_errors[lead] for lead in LEADS]), times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = EvalConfig.from_yaml(args.config)
    cfg.ckpt_path = args.checkpoint
    cfg.experiment.base_output_dir = str(output / "runtime")
    cfg.experiment.name = "matched-evaluation"
    cfg.experiment.wandb.mode = "disabled"
    assert isinstance(cfg.model, SamudraConfig)
    cfg.model.auxiliary_ke = not args.control
    evaluator = Eval(cfg)
    evaluator.model.eval()
    assert evaluator.input_steps == evaluator.output_steps == 2
    source = evaluator.source
    wet = source.masks.prognostic.cpu().numpy().astype(bool)
    statistics = source.statistics(evaluator.prognostic_var_names)
    lat, lon = source.resolution
    np.savez_compressed(
        output / "grid.npz",
        wet=wet,
        area=evaluator.area_weights.cpu().numpy(),
        std=statistics.std,
        lat=lat.cpu().numpy(),
        lon=lon.cpu().numpy(),
        channels=np.asarray(evaluator.prognostic_var_names),
        depths=np.asarray(evaluator.data_layout.depth_levels),
    )
    cont = output / "continuous"
    cont.mkdir()
    if args.smoke:
        evaluator.inference_dataset.size = 2
    score(evaluator, evaluator.inference_dataset, cont, continuous=True)
    print("Continuous rollout complete", flush=True)
    # One common initialization at the first available date on/after each quarter.
    dates = source.time.values
    date_strings = np.asarray([str(t)[:10] for t in dates])
    starts = [
        f"{year}-{month:02d}-01"
        for year in range(2015, 2022)
        for month in (1, 4, 7, 10)
    ]
    if args.smoke:
        starts = starts[:1]
    maps = np.lib.format.open_memmap(
        output / "quarterly_errors.npy",
        mode="w+",
        dtype="float32",
        shape=(len(starts), len(LEADS), *wet.shape),
    )
    initializations = []
    for number, nominal in enumerate(starts):
        index = int(np.searchsorted(date_strings, nominal))
        # Two context frames and 72 target frames; end remains before the next sample.
        end = dates[index + 72] + datetime.timedelta(days=1)
        sliced = source.slice_time(
            Om4TimeConfig.model_validate(
                dict(start=str(dates[index - 1])[:10], end=str(end)[:10])
            )
        )
        dataset = InferenceDataset(
            sliced,
            evaluator.prognostic_var_names,
            evaluator.boundary_var_names,
            2,
            2,
            cfg.data.normalize_before_mask,
            cfg.data.masked_fill_value,
            True,
        )
        assert len(dataset) == 36, (nominal, len(dataset))
        actual_leads = [
            (dates[index + lead // 5] - dates[index]).total_seconds() / 86400
            for lead in LEADS
        ]
        if any(
            actual not in (nominal, nominal + 1)
            for actual, nominal in zip(actual_leads, LEADS, strict=True)
        ):
            raise ValueError(f"Unexpected physical forecast leads: {actual_leads}")
        quarter = output / f"quarter-{number:02d}"
        quarter.mkdir()
        errors, target_dates = score(evaluator, dataset, quarter, continuous=False)
        maps[number] = errors
        maps.flush()
        initializations.append(
            dict(
                nominal=nominal,
                initial_date=str(dates[index]),
                actual_lead_days=actual_leads,
                target_dates=target_dates,
            )
        )
        print(f"Quarter {number + 1}/28 complete: {dates[index]}", flush=True)
    receipt = dict(
        checkpoint=args.checkpoint,
        checkpoint_sha256=sha256(args.checkpoint),
        evaluation_script_sha256=sha256(__file__),
        config_sha256=sha256(args.config),
        control=args.control,
        smoke=args.smoke,
        leads=LEADS,
        initializations=initializations,
        continuous_initial_dates=[str(t) for t in dates[:2]],
        metric="normalized errors; equal channel weight; spherical area within each channel wet mask",
    )
    (output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    evaluator.finish()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
