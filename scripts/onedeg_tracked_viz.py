# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Run the standard visualization pipeline with online W&B progress and artifacts."""

import os
import time

import wandb

from samudra.utils.location import LocalLocation
from samudra.utils.logging import handle_logging
from samudra.viz.config import VizConfig, run_steps


def main():
    cfg = VizConfig.from_yaml_and_cli()
    cfg.output_path.mkdir(parents=True, exist_ok=False)
    handle_logging(cfg.debug, cfg.output_path)
    cfg.save_yaml(cfg.output_path / "config.yaml")
    with wandb.init(
        entity="ocean_emulators",
        project="default",
        group=os.environ["WANDB_RUN_GROUP"],
        job_type="visualization",
        name=cfg.name,
        mode="online",
        dir=str(cfg.output_path),
        config={
            "viz": cfg.model_dump(mode="json"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "code_commit": os.environ["WANDB_GIT_COMMIT"],
        },
        tags=["experiment", "direct-weight-edit", "onedeg", "comparison"],
    ) as run:
        if run.offline:
            raise RuntimeError("Online W&B tracking is required for this experiment")
        (cfg.output_path / "wandb-url.txt").write_text(run.url + "\n")
        run.log({"viz/stages_complete": 0})
        viz = cfg.build(LocalLocation(path=cfg.output_path))
        logged_images = set()
        for index, step in enumerate(cfg.selected_steps, 1):
            start = time.monotonic()
            run_steps(viz, [step])
            run.log(
                {
                    "viz/stages_complete": index,
                    f"viz/seconds/{step}": time.monotonic() - start,
                }
            )
            for path in sorted(cfg.output_path.rglob("*.png")):
                relative = path.relative_to(cfg.output_path).as_posix()
                if relative not in logged_images:
                    run.log({f"figures/{relative}": wandb.Image(str(path))})
                    logged_images.add(relative)
        artifact = wandb.Artifact(cfg.name, type="evaluation-figures")
        for path in sorted(cfg.output_path.rglob("*")):
            if path.is_file() and path.suffix.lower() in {
                ".png",
                ".pdf",
                ".csv",
                ".yaml",
                ".txt",
            }:
                artifact_path = path.relative_to(cfg.output_path)
                if "wandb" not in artifact_path.parts:
                    artifact.add_file(str(path), name=artifact_path.as_posix())
        run.log_artifact(artifact)
        run.summary["viz/figure_count"] = len(logged_images)


if __name__ == "__main__":
    main()
