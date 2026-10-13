# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Training-only calibration and real-batch qualification; no test data selection."""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from samudra.config import TrainConfig
from samudra.models.samudra import Samudra
from samudra.stepper import train_batch
from samudra.train import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = TrainConfig.from_yaml(args.config)
    cfg.backend = "cuda"
    cfg.post_train_eval = None
    cfg.experiment.wandb.mode = "disabled"
    cfg.experiment.name += "-qualification"
    cfg.experiment.base_output_dir = str(Path(args.output).parent / "qualification")
    trainer = Trainer(cfg)
    assert cfg.auxiliary_ke is not None
    assert trainer.auxiliary_targets is not None
    assert isinstance(trainer.model, Samudra)
    assert trainer.model.auxiliary_head is not None
    trainer.init_data_loaders(4)
    trainer.model.train()
    params = list(trainer.model.unet.parameters())
    rows: list[dict[str, Any]] = []
    # Fixed first four seed-15 training batches. No parameter updates or held-out selection.
    iterator = iter(trainer.train_loader)
    for index in range(4):
        batch = next(iterator)
        trainer.auxiliary_targets.attach(batch, trainer.device)
        result = train_batch(trainer.model, batch, trainer.loss_fn)
        assert result.auxiliary_loss is not None
        physical = torch.autograd.grad(result.loss, params, retain_graph=True)
        accessory = torch.autograd.grad(
            result.auxiliary_loss, params, retain_graph=True
        )
        pnorm = torch.sqrt(
            sum(
                (g.double().square().sum() for g in physical),
                torch.zeros((), device=trainer.device, dtype=torch.float64),
            )
        ).item()
        anorm = torch.sqrt(
            sum(
                (g.double().square().sum() for g in accessory),
                torch.zeros((), device=trainer.device, dtype=torch.float64),
            )
        ).item()
        cosine = sum(
            (
                (a.double() * b.double()).sum()
                for a, b in zip(physical, accessory, strict=True)
            ),
            torch.zeros((), device=trainer.device, dtype=torch.float64),
        ).item() / (pnorm * anorm)
        if not np.isfinite([pnorm, anorm, cosine]).all() or min(pnorm, anorm) <= 0:
            raise ValueError("Invalid calibration gradients")
        rows.append(
            dict(
                batch=index,
                physical_loss=result.loss.item(),
                auxiliary_mse=result.auxiliary_loss.item(),
                physical_gradient_norm=pnorm,
                auxiliary_gradient_norm=anorm,
                cosine=cosine,
                first_label_dates=[str(t) for t in batch.label_times[0][0]],
                last_label_dates=[str(t) for t in batch.label_times[-1][0]],
            )
        )
        result.optimization_loss.backward()
        assert trainer.model.auxiliary_head.weight.grad is not None
        assert trainer.model.auxiliary_head.weight.grad.abs().sum() > 0
        trainer.optimizer.zero_grad()
        print(json.dumps(rows[-1]), flush=True)
        del result, batch, physical, accessory
    trainer.train_loader.close()
    # Common coefficient is derived from aligned supervision only and later copied to seasonal.
    coefficient = float(
        0.1
        * np.median(
            [r["physical_gradient_norm"] / r["auxiliary_gradient_norm"] for r in rows]
        )
    )
    trainer.model.eval()
    trainer.init_data_loaders(4)
    batch = next(iter(trainer.train_loader))
    with torch.no_grad():
        before = trainer.model(batch)
        state = trainer.model.state_dict()
        trainer.model.load_state_dict(state, strict=True)
        after = trainer.model(batch)
        for a, b in zip(before, after, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    receipt = dict(
        mode=cfg.auxiliary_ke.mode,
        suggested_coefficient=coefficient,
        target=trainer.auxiliary_targets.provenance,
        batches=rows,
        status="qualified",
        cuda=torch.version.cuda,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    )
    Path(args.output).write_text(json.dumps(receipt, indent=2) + "\n")
    trainer.finish()


if __name__ == "__main__":
    main()
