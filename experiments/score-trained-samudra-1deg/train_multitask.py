# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0

"""Matched forecast/completion training with annual validation and task audits."""

import argparse
import itertools
import json
import logging
import os
import time

import torch
from multitask_model import TASK_PROBABILITIES, TASKS, MultitaskModelConfig, task_masks
from native_pointwise import configure_native_pointwise
from score_model import crps, reduce_field, weights
from train_score import ScoreTrainConfig, ScoreTrainer

from samudra.config import PostTrainEvalConfig
from samudra.datasets import InferenceDataset
from samudra.utils.distributed import is_main_process
from samudra.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


class MultitaskTrainConfig(ScoreTrainConfig):
    model: MultitaskModelConfig
    native_pointwise: bool = False


MultitaskTrainConfig.model_rebuild(
    _types_namespace={"PostTrainEvalConfig": PostTrainEvalConfig}
)


class MultitaskTrainer(ScoreTrainer):
    def write_manifest(self):
        super().write_manifest()
        path = self.output_dir / "data-manifest.json"
        record = json.loads(path.read_text())
        record["completion"] = {
            "mode": self.cfg.model.mode,
            "tasks": TASKS,
            "probabilities": TASK_PROBABILITIES
            if self.cfg.model.mode == "multitask"
            else [1, 0, 0, 0],
            "task_seed": "31000000 + 1000*epoch + rank; separate CPU generator",
            "mask_scope": "one task and spatial/variable mask per rank minibatch; four model calls",
            "surface_fields": ["thetao_0", "zos"],
            "reconstruction_history": "unavailable; input values masked inside model",
            "gap": "quarter height by quarter width, wet-cell-centered, periodic longitude",
            "nominal_time_slots_days": [-5, 0, 5, 10],
            "note": "same source steps as Samudra; occasional six-day calendar intervals retained in origin manifest",
        }
        record["validation_selection"] = self.validation_checkpoint_identity()
        record["native_pointwise"] = self.cfg.native_pointwise
        path.write_text(json.dumps(record, indent=2) + "\n")

    def train_one_epoch(self, epoch, **kwargs):
        self.score_model.reset_tasks(epoch, self.rank, self.device)
        stats = super().train_one_epoch(epoch, **kwargs)
        counts = self.score_model.task_counts.clone()
        sums = self.score_model.task_loss_sums.clone()
        if self.distributed:
            torch.distributed.all_reduce(counts)
            torch.distributed.all_reduce(sums)
        for i, task in enumerate(TASKS):
            stats[f"train/task/{task}/contexts"] = float(counts[i])
            stats[f"train/task/{task}/fraction"] = float(counts[i] / counts.sum())
            if counts[i] > 0:
                stats[f"train/task/{task}/loss"] = float(sums[i] / counts[i])
        if is_main_process():
            with (self.output_dir / "task-exposure.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "contexts": counts.tolist(),
                            "loss_sums": sums.tolist(),
                        }
                    )
                    + "\n"
                )
        logger.info("Task exposure epoch %s: %s", epoch, counts.tolist())
        return stats

    @torch.no_grad()
    def validate_one_epoch(self, epoch):
        result = super().validate_one_epoch(epoch)
        if epoch != 1 and epoch % 5:
            return result
        # Identical held-out validation masks in both arms, independent of training.
        total = torch.zeros(3, 4, device=self.device)
        rng = torch.Generator().manual_seed(51000000 + self.rank)
        with (
            self._test_context(),
            torch.random.fork_rng(
                devices=[self.device] if self.device.type == "cuda" else []
            ),
        ):
            torch.manual_seed(52000000 + self.rank)
            for batch in itertools.islice(self.val_loader, 4):
                initial, boundary = batch.get_initial_input()
                history = initial[None].expand(8, *initial.shape)
                target = batch.get_label(0)
                for i, task in enumerate(TASKS[1:]):
                    history_known, known, request = task_masks(
                        task, batch.ctx.label_mask[:154], rng
                    )
                    pred = self.score_model.complete(
                        history,
                        boundary,
                        batch.ctx,
                        target,
                        history_known,
                        known,
                        request,
                    )
                    area = weights(batch.ctx, 154, area=True) * request
                    active = area.sum((-2, -1)) > 0
                    scores = torch.stack(
                        [
                            reduce_field(crps(pred, target, 0), area)[active].mean(),
                            reduce_field((pred.mean(0) - target).square(), area)[
                                active
                            ].mean(),
                            reduce_field(pred.var(0, unbiased=True), area)[
                                active
                            ].mean(),
                        ]
                    )
                    total[i, :3] += scores * len(target)
                    total[i, 3] += len(target)
        if self.distributed:
            torch.distributed.all_reduce(total)
        for i, task in enumerate(TASKS[1:]):
            value = total[i, :3] / total[i, 3]
            result.update(
                {
                    f"completion_val/{task}/crps": float(value[0]),
                    f"completion_val/{task}/mean_rmse": float(value[1].sqrt()),
                    f"completion_val/{task}/spread_rms": float(value[2].sqrt()),
                    f"completion_val/{task}/contexts": float(total[i, 3]),
                }
            )
        logger.info("Completion validation epoch %s: %s", epoch, result)
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
            if len(dataset) < 36:
                raise ValueError("Validation interval cannot hold a 360-day rollout")
            self.model.eval()
            metrics, selection = {}, []
            with (
                self._test_context(),
                torch.random.fork_rng(
                    devices=[self.device] if self.device.type == "cuda" else []
                ),
            ):
                torch.manual_seed(26000000)
                initial = dataset[0][0].to(self.device)
                history = initial[None].expand(8, *initial.shape)
                for step in range(36):
                    pred = self.score_model.sample_step(
                        history, dataset.get_boundary(step).to(self.device), dataset.ctx
                    )
                    if not torch.isfinite(pred).all():
                        raise FloatingPointError(
                            f"Nonfinite annual validation epoch={epoch}, step={step}"
                        )
                    history = pred
                    lead = (step + 1) * 10
                    if lead in self.validation_checkpoint_identity()["leads_days"]:
                        target = dataset.inference_target(step).to(self.device)[:, 77:]
                        area = weights(dataset.ctx, 154, area=True)[77:]
                        p = pred[:, :, 77:]
                        score = reduce_field(crps(p, target, 0), area).mean()
                        rmse = (
                            reduce_field((p.mean(0) - target).square(), area)
                            .sqrt()
                            .mean()
                        )
                        metrics[f"rollout_val/origin0/{lead}d/crps"] = float(score)
                        metrics[f"rollout_val/origin0/{lead}d/rmse"] = float(rmse)
                        selection.append(score)
            metrics["rollout_val/selection_score"] = float(
                torch.stack(selection).mean()
            )
            logger.info("Annual validation epoch %s: %s", epoch, metrics)
            return metrics

    def validation_checkpoint_identity(self):
        return {
            "metric": "empirical_crps",
            "leads_days": [10, 20, 40, 90, 180, 360],
            "validation_origin_indices": [0],
            "members": 8,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--qualify", action="store_true")
    args = parser.parse_args()
    cfg = MultitaskTrainConfig.from_yaml(args.config)
    configure_native_pointwise(cfg.native_pointwise)
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    logger.info(
        "1x1 convolution backend: %s",
        "native PyTorch" if cfg.native_pointwise else "PyTorch default",
    )
    trainer = MultitaskTrainer(cfg)
    try:
        run = trainer.wandb_logger.run
        if not args.qualify:
            online = torch.tensor(
                int(not is_main_process() or (run is not None and not run.offline)),
                device=trainer.device,
            )
            if trainer.distributed:
                torch.distributed.broadcast(online, src=0)
            if not online:
                raise RuntimeError("Production requires an online W&B run")
        if run is not None:
            logger.info("Online multitask training: %s", run.url)
        if args.qualify:
            trainer.init_data_loaders(4)
            trainer.train_sampler.set_epoch(1)
            trainer.score_model.reset_tasks(1, trainer.rank, trainer.device)
            batch = next(iter(trainer.train_loader))
            # Exercise backward through every task on real fields before trusting
            # random sampling to cover rare task branches in a brief smoke test.
            task_probe = {}
            trainer.model.train()
            for task in TASKS:
                trainer.optimizer.zero_grad()
                value = trainer.score_model.forward_task(batch, task, True).mean()
                value.backward()
                assert torch.isfinite(value)
                assert all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in trainer.model.parameters()
                )
                task_probe[task] = float(value.detach())
            trainer.optimizer.zero_grad()
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            trainer.train_one_epoch(1, max_optimizer_steps=3)
            elapsed = time.monotonic() - start
            metrics = trainer.validate_one_epoch(
                1
            ) | trainer.validate_rollout_one_epoch(1)
            if is_main_process():
                trainer.save_checkpoint(0, trainer.output_dir / "qualification-only.pt")
                record = {
                    "mode": cfg.model.mode,
                    "optimizer_steps": trainer.train_progress.optimizer_steps,
                    "train_seconds": elapsed,
                    "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "source_commit": os.environ.get("SAMUDRA_CODE_COMMIT"),
                    "native_pointwise": cfg.native_pointwise,
                    "task_backward_probe": task_probe,
                    "model_parameters": sum(
                        p.numel() for p in trainer.model.parameters()
                    ),
                    **metrics,
                }
                (trainer.output_dir / "qualification.json").write_text(
                    json.dumps(record, indent=2) + "\n"
                )
                logger.info("QUALIFICATION COMPLETE: %s", record)
        else:
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
