# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production model and loader for one optimizer update."""

import sys

import torch

from samudra.config import TrainConfig
from samudra.train import Trainer
from samudra.utils.logging import handle_logging

cfg = TrainConfig.from_yaml(sys.argv[1])
cfg.experiment.name += "-probe"
cfg.experiment.wandb.mode = "disabled"
cfg.prepare_output_dirs()
handle_logging(False, cfg.experiment.output_dir)
trainer = Trainer(cfg)
trainer.probe_optimizer_step()
if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()
