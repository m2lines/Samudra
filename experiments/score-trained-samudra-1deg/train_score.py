# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Use Samudra's trainer, Rust loader, DDP, optimizer and EMA for B/C/D."""

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import torch
from score_model import (
    ScoreModelConfig,
    crps,
    depth_difference,
    depth_weights,
    pool_field,
    reduce_field,
    time_difference,
    weights,
)

from samudra.config import PostTrainEvalConfig, TrainConfig
from samudra.datasets import InferenceDataset
from samudra.train import Trainer
from samudra.utils.distributed import is_main_process
from samudra.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


class ScoreTrainConfig(TrainConfig):
    model: ScoreModelConfig
    scales_path: str | None = None
    calibrate_auxiliary: bool = False


ScoreTrainConfig.model_rebuild(
    _types_namespace={"PostTrainEvalConfig": PostTrainEvalConfig}
)


class ScoreTrainer(Trainer):
    def __init__(self, cfg):
        self.arm = cfg.model.arm
        super().__init__(cfg)
        self.cfg = cfg
        self.rank = (self.distributed.rank or 0) if self.distributed else 0
        expected = [
            f"{v}_{z}" for v in ("uo", "vo", "thetao", "so") for z in range(19)
        ] + ["zos"]
        assert list(self.prognostic_var_names) == expected, (
            "Structured differences require the pinned 77-field ordering"
        )
        assert list(self.boundary_var_names) == ["tauuo", "tauvo", "hfds"]
        if cfg.scales_path:
            scales = json.loads(Path(cfg.scales_path).read_text())
            for name in ("spatial_rms", "depth_rms", "time_rms"):
                getattr(self.score_model, name).copy_(
                    torch.tensor(scales[name], device=self.device)
                )
        if is_main_process():
            self.write_manifest()
            logger.info(
                "Model parameters: %s", sum(p.numel() for p in self.model.parameters())
            )

    @property
    def score_model(self):
        return self.model.module if self.distributed else self.model

    def write_manifest(self):
        """Record every permitted fitting/validation window and assert separation."""
        target_sets = []
        summaries: dict[str, Any] = {}
        for split, src, calls in (
            ("train", self.data_bundle.train_sources[0], 4),
            ("validation", self.data_bundle.val_sources[0], 1),
        ):
            times = [str(t) for t in src.time.values]
            targets = set()
            path = self.output_dir / f"{split}-origins.csv.gz"
            with gzip.open(path, "wt") as f:
                writer = csv.writer(f)
                writer.writerow(["window", "input_0", "input_1", "target_times"])
                for start in range(len(times) - 2 - 2 * calls + 1):
                    output_times = times[start + 2 : start + 2 + 2 * calls]
                    targets.update(output_times)
                    writer.writerow(
                        [start, times[start], times[start + 1], "|".join(output_times)]
                    )
            target_sets.append(targets)
            summaries[split] = {
                "first": times[0],
                "last": times[-1],
                "times": len(times),
                "target_first": min(targets),
                "target_last": max(targets),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        assert not target_sets[0].intersection(target_sets[1]), (
            "Training and validation targets overlap"
        )
        assert max(target_sets[1]) < "2014-10-10", (
            "Validation touches the held-out test interval"
        )
        # The reused control's home path now resolves to this project store.
        summaries["data_root"] = str(self.cfg.experiment.resolved_data_root)
        summaries["variables"] = list(self.prognostic_var_names)
        summaries["forcing"] = list(self.boundary_var_names)
        stats = self.primary_source.statistics(self.prognostic_var_names)
        summaries["normalization_mean"] = stats.mean.reshape(-1).tolist()
        summaries["normalization_std"] = stats.std.reshape(-1).tolist()
        summaries["mask_sha256"] = hashlib.sha256(
            self.primary_source.masks.prognostic.cpu().numpy().tobytes()
        ).hexdigest()
        summaries["coordinates"] = [
            v.cpu().tolist() for v in self.primary_source.resolution
        ]
        summaries["upstream_attributes"] = {
            k: v
            for k, v in self.primary_source.attrs.items()
            if k.startswith("m2lines/")
        }
        summaries["rng"] = (
            "epoch-boundary restart: training seed=15000000+1000*epoch+rank; validation seed=25000000+rank; initialization seed=15; noise projection initialization=150001; sampler seed=15"
        )
        (self.output_dir / "data-manifest.json").write_text(
            json.dumps(summaries, indent=2) + "\n"
        )

    @torch.no_grad()
    def compute_scales(self):
        """Freeze target RMS on 64 training origins, 4 batches per rank (4 GPUs)."""
        self.init_data_loaders(4)
        self.train_sampler.set_epoch(0)
        sums = None
        count = 0
        origin_indices = list(itertools.islice(iter(self.train_sampler), 4))
        for batch in itertools.islice(self.train_loader, 4):
            area = weights(batch.ctx, 154, area=True)
            previous = batch.get_initial_input()[0][:, -77:]
            for step in range(4):
                target = batch.get_label(step)
                spatial = []
                for size in (4, 16):
                    value, w = pool_field(target, area, size)
                    spatial.append(reduce_field(value.square(), w))
                depth = reduce_field(
                    depth_difference(target).square(), depth_weights(area)
                )
                temporal = reduce_field(
                    time_difference(target, previous).square(), area
                )
                current = torch.cat([*spatial, depth, temporal])
                sums = current if sums is None else sums + current
                count += 1
                previous = target[:, -77:]
        if self.distributed:
            torch.distributed.all_reduce(sums)
        values = (sums / (count * self.world_size)).sqrt()
        a, b, c = torch.split(values, [308, 144, 154])
        self.score_model.spatial_rms.copy_(a.reshape(2, 154))
        self.score_model.depth_rms.copy_(b)
        self.score_model.time_rms.copy_(c)
        all_indices: list[Any] = [None for _ in range(self.world_size)]
        if self.distributed:
            torch.distributed.all_gather_object(all_indices, origin_indices)
        else:
            all_indices = [origin_indices]
        if is_main_process():
            payload = {
                name: getattr(self.score_model, name).cpu().tolist()
                for name in ("spatial_rms", "depth_rms", "time_rms")
            }
            payload["training_window_indices_by_rank"] = all_indices
            payload["sampler_epoch"] = 0
            (self.output_dir / "structured-scales.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            logger.info(
                "Frozen structured target scales from %s contexts",
                count // 4 * self.batch_size * self.world_size,
            )

    def train_one_epoch(self, epoch, **kwargs):
        # Per-epoch/per-rank counter is the RNG checkpoint: only epoch boundaries
        # are restartable. Validation, model summaries and scale preparation cannot
        # alter later epochs. Checkpointing preserves the RNG inside each block.
        with torch.random.fork_rng(
            devices=[self.device] if self.device.type == "cuda" else []
        ):
            torch.manual_seed(15000000 + epoch * 1000 + self.rank)
            return super().train_one_epoch(epoch, **kwargs)

    def calibrate_auxiliary_weights(self):
        """Cap each auxiliary's initial output-gradient norm at 10% of grid CRPS.

        One batch per rank, from training sampler epoch zero, before any updates.
        Frozen coefficients use no validation/test targets or scores.
        """
        assert self.arm == "D"
        self.init_data_loaders(4)
        self.train_sampler.set_epoch(0)
        batch = next(iter(self.train_loader))
        initial, boundary = batch.get_initial_input()
        history = initial[None].expand(4, *initial.shape)
        with torch.random.fork_rng(
            devices=[self.device] if self.device.type == "cuda" else []
        ):
            torch.manual_seed(55000000 + self.rank)
            with torch.no_grad():
                pred = self.score_model.sample_step(history, boundary, batch.ctx)
        pred = pred.detach().requires_grad_(True)
        terms = self.score_model.score_components(
            pred, batch.get_label(0), history[:, :, -77:], initial[:, -77:], batch.ctx
        )
        names = ["grid", "spatial", "differences"]
        squares = torch.stack(
            [
                torch.autograd.grad(terms[name].mean(), pred, retain_graph=True)[0]
                .square()
                .sum()
                for name in names
            ]
        )
        values = torch.stack([terms[name].mean().detach() for name in names])
        if self.distributed:
            torch.distributed.all_reduce(squares)
            torch.distributed.all_reduce(values)
        norms = squares.sqrt()
        values /= self.world_size
        assert torch.isfinite(norms).all() and (norms > 0).all()
        spatial = min(0.1, float(0.1 * norms[0] / norms[1]))
        differences = min(0.1, float(0.1 * norms[0] / norms[2]))
        self.score_model.spatial_weight = self.cfg.model.spatial_weight = spatial
        self.score_model.difference_weight = self.cfg.model.difference_weight = (
            differences
        )
        if is_main_process():
            payload = {
                "training_contexts": self.batch_size * self.world_size,
                "sampler_epoch": 0,
                "sampler_batch": 0,
                "model_call": 0,
                "noise_seed": 55000000,
                "unweighted_values": dict(
                    zip(names, values.cpu().tolist(), strict=True)
                ),
                "output_gradient_norms": dict(
                    zip(names, norms.cpu().tolist(), strict=True)
                ),
                "spatial_weight": spatial,
                "difference_weight": differences,
            }
            (self.output_dir / "auxiliary-calibration.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )
            self.cfg.save_yaml(self.output_dir / "config.yaml")
            logger.info("FROZEN AUXILIARY CALIBRATION: %s", payload)

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        self.model.eval()
        total = torch.zeros(4, device=self.device)
        with (
            self._test_context(),
            torch.random.fork_rng(
                devices=[self.device] if self.device.type == "cuda" else []
            ),
        ):
            torch.manual_seed(25000000 + self.rank)
            for batch in self.val_loader:
                initial, boundary = batch.get_initial_input()
                m = 1 if self.arm == "B" else 8
                history = initial.unsqueeze(0).expand(m, *initial.shape)
                pred = self.score_model.sample_step(history, boundary, batch.ctx)
                target = batch.get_label(0)
                area = weights(batch.ctx, 154, area=True)
                metrics = torch.stack(
                    [
                        reduce_field(crps(pred, target, alpha=0), area).mean(),
                        reduce_field((pred.mean(0) - target).square(), area).mean(),
                        reduce_field(pred.var(0, unbiased=False), area).mean(),
                    ]
                )
                total[:3] += metrics * target.shape[0]
                total[3] += target.shape[0]
        if self.distributed:
            torch.distributed.all_reduce(total)
        values = total[:3] / total[3]
        result = {
            "val/mean/loss": float(values[1] if self.arm == "B" else values[0]),
            "val/empirical_crps": float(values[0]),
            "val/ensemble_mean_rmse": float(values[1].sqrt()),
            "val/spread_rms": float(values[2].sqrt()),
        }
        logger.info("Validation epoch %s: %s", epoch, result)
        return result

    @torch.no_grad()
    def validate_rollout_one_epoch(self, epoch):
        with self._rank0_rollout_validation_context() as active:
            if not active:
                return {}
            source = self.data_bundle.val_sources[0]
            dataset = InferenceDataset(
                source,
                self.prognostic_var_names,
                self.boundary_var_names,
                2,
                2,
                self.normalize_before_mask,
                self.masked_fill_value,
                False,
            ).to(self.device)
            self.model.eval()
            metrics = {}
            scores = []
            with (
                self._test_context(),
                torch.random.fork_rng(
                    devices=[self.device] if self.device.type == "cuda" else []
                ),
            ):
                # Two validation-only origins, each containing all 90-day targets.
                for origin in (0, 18):
                    torch.manual_seed(26000000 + origin)
                    initial = dataset[origin][0].to(self.device)
                    m = 1 if self.arm == "B" else 8
                    history = initial.unsqueeze(0).expand(m, *initial.shape)
                    for step in range(9):
                        boundary = dataset.get_boundary(origin + step).to(self.device)
                        pred = self.score_model.sample_step(
                            history, boundary, dataset.ctx
                        )
                        history = pred
                        lead = (step + 1) * 10
                        if lead in (10, 20, 40, 90):
                            target = dataset.inference_target(origin + step).to(
                                self.device
                            )
                            area = weights(dataset.ctx, 154, area=True)[77:]
                            px, py = pred[:, :, 77:], target[:, 77:]
                            score = reduce_field(crps(px, py, alpha=0), area).mean()
                            rmse = (
                                reduce_field((px.mean(0) - py).square(), area)
                                .sqrt()
                                .mean()
                            )
                            metrics[f"rollout_val/origin{origin}/{lead}d/crps"] = float(
                                score
                            )
                            metrics[f"rollout_val/origin{origin}/{lead}d/rmse"] = float(
                                rmse
                            )
                            scores.append(rmse if self.arm == "B" else score)
            metrics["rollout_val/selection_score"] = float(torch.stack(scores).mean())
            logger.info("Rollout validation epoch %s: %s", epoch, metrics)
            return metrics

    def validation_checkpoint_score(self, epoch, val_stats, rollout_val_stats):
        return rollout_val_stats.get("rollout_val/selection_score")

    def validation_checkpoint_identity(self):
        return {
            "metric": "ensemble_mean_rmse" if self.arm == "B" else "empirical_crps",
            "leads_days": [10, 20, 40, 90],
            "validation_origin_indices": [0, 18],
            "members": 1 if self.arm == "B" else 8,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--compute-scales", action="store_true")
    args = parser.parse_args()
    cfg = ScoreTrainConfig.from_yaml(args.config)
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    trainer = ScoreTrainer(cfg)
    try:
        if args.compute_scales:
            trainer.compute_scales()
        if cfg.calibrate_auxiliary:
            if not args.qualify:
                raise ValueError(
                    "Calibrate only in qualification, then freeze coefficients for training"
                )
            trainer.calibrate_auxiliary_weights()
        if args.qualify:
            trainer.init_data_loaders(4)
            trainer.train_sampler.set_epoch(1)
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            trainer.train_one_epoch(1, max_optimizer_steps=3)
            elapsed = time.monotonic() - start
            validation = trainer.validate_one_epoch(1)
            rollout = trainer.validate_rollout_one_epoch(1)
            # A probe is explicitly NOT a completed epoch; never resume from it.
            if is_main_process():
                trainer.save_checkpoint(0, trainer.output_dir / "qualification-only.pt")
                result = {
                    "arm": trainer.arm,
                    "optimizer_steps": trainer.train_progress.optimizer_steps,
                    "train_seconds": elapsed,
                    "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "seconds_per_microbatch_including_startup": elapsed
                    / trainer.num_batches_seen,
                    "source_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
                    **validation,
                    **rollout,
                }
                (trainer.output_dir / "qualification.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                logger.info("QUALIFICATION COMPLETE: %s", result)
        else:
            if trainer.arm == "D" and not cfg.scales_path:
                raise ValueError(
                    "D requires frozen training-only scales from qualification"
                )
            trainer.run()
    finally:
        for name in ("train_loader", "val_loader"):
            if hasattr(trainer, name):
                getattr(trainer, name).close()
        trainer.finish()
        if trainer.distributed:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
