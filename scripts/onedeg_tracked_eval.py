# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the native evaluator, requiring online W&B and recording weight provenance."""

import hashlib
import os
from pathlib import Path

import torch

from samudra.config import EvalConfig
from samudra.eval import Eval
from samudra.utils.logging import handle_logging, handle_warnings


def main():
    cfg = EvalConfig.from_yaml_and_cli()
    if cfg.experiment.wandb.mode != "online":
        raise ValueError("This experiment requires online W&B tracking")
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    evaluator = Eval(cfg)
    run = evaluator.wandb_logger.run
    if run is None or run.offline:
        raise RuntimeError(
            "W&B did not initialize online; refusing an untracked rollout"
        )
    assert cfg.ckpt_path is not None
    with open(cfg.ckpt_path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    checkpoint = torch.load(
        cfg.ckpt_path, map_location="cpu", mmap=True, weights_only=True
    )
    run.config.update(
        {
            "weight_edit": checkpoint.get(
                "edit_provenance", {"status": "original reference"}
            ),
            "checkpoint_sha256": digest,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
    )
    del checkpoint
    (Path(cfg.experiment.output_dir) / "wandb-url.txt").write_text(run.url + "\n")
    evaluator.run()


if __name__ == "__main__":
    main()
