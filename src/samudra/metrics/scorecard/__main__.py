# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

r"""Score an existing rollout, without rerunning it.

    python -m samudra.metrics.scorecard samudra_om4_v2/eval.yaml \\
        --experiment.name=my_past_eval \\
        --scorecard.track=om4-v2026-09-onedeg-thermo_dynamic_all \\
        --scorecard.climatology_location=/path/to/climatology.zarr \\
        --scorecard.champion=https://.../scorecard/<track>/champion

Takes the same `EvalConfig` as `samudra.eval` and reads the `predictions.zarr`
that job wrote. Writes `<output_dir>/scorecard/` and exits with status 1 when
the gate fails, so it can stand in a pipeline. No GPU or checkpoint needed.
"""

import logging
import sys

from samudra.config import EvalConfig
from samudra.metrics.run import open_predictions
from samudra.metrics.scorecard import run
from samudra.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


def main() -> int:
    cfg = EvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()
    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()
    if cfg.scorecard is None:
        raise ValueError("This config has no `scorecard` block; nothing to score.")

    root = cfg.experiment.resolved_data_root
    bundle = cfg.data.build(root, include_inference=True)
    source = bundle.inference_source
    if source is None:
        raise ValueError("Inference time is not configured for the first data source")
    climatology = root.resolve(cfg.scorecard.climatology_location).open({})
    tables, scored = run.score(
        open_predictions(cfg.experiment.output_dir),
        source.to_xarray_dataset(),
        climatology,
        cfg.scorecard,
        bundle.data_layout,
        label=cfg.experiment.name,
        truth_name=str(root.resolve(cfg.data.sources[0].data_location)),
    )
    run.write(cfg.experiment.output_dir / "scorecard", tables, scored)
    for check in scored.checks:
        logger.info(
            "  %-28s %s  %s",
            check.name,
            "pass" if check.passed else "FAIL",
            check.detail,
        )
    return 0 if scored.passed else 1


if __name__ == "__main__":
    sys.exit(main())
