# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

r"""Score an existing rollout against observations, without rerunning it.

    python -m samudra.metrics configs/samudra_om4_v2/eval.yaml \\
        --experiment.name=my_past_eval --observations.enabled=true

Takes the same `EvalConfig` as `samudra.eval` and reads the `predictions.zarr`
that job already wrote, so iterating on a metric costs minutes rather than the
hours a fresh rollout would. No GPU and no checkpoint required.
"""

import logging

from samudra.config import EvalConfig
from samudra.metrics.run import open_predictions, run_observation_metrics
from samudra.utils.data import DataContainer
from samudra.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


def main() -> None:
    cfg = EvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    if not cfg.observations.enabled:
        raise ValueError(
            "Observation metrics are disabled for this config; rerun with "
            "--observations.enabled=true"
        )

    data_container: DataContainer = cfg.data.build(cfg.experiment.resolved_data_root)
    src = data_container.inference_source
    if src is None:
        raise ValueError("Inference time is not configured for the first data source")

    baselines = {"om4": src.data} if "om4" in cfg.observations.baselines else {}

    frame, scalars = run_observation_metrics(
        cfg.observations,
        predictions=open_predictions(cfg.experiment.output_dir),
        dataset_spec=data_container.dataset_spec,
        data_root=cfg.experiment.resolved_data_root,
        model_label=cfg.experiment.name,
        baselines=baselines,
        output_dir=cfg.experiment.output_dir,
    )

    logger.info("Observation metrics:")
    for key, value in sorted(scalars.items()):
        logger.info("  %s = %s", key, value)
    logger.info("Wrote %d rows of metric detail alongside the rollout", len(frame))


if __name__ == "__main__":
    main()
