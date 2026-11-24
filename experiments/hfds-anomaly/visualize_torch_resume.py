"""Continue native visualization from successful step records in the prior job log."""

import json
import re
from pathlib import Path

import dask

from samudra.viz.config import VizConfig, main

if __name__ == "__main__":
    root = Path("/scratch/jr7309/om4-hfds-ablation-20261021")
    cfg = VizConfig.from_yaml_and_cli()
    prior = root / "logs/torch-viz-20861102_1.out"
    completed = set(
        re.findall(r" - config - Step (\w+) took [\d.]+ seconds", prior.read_text())
    )
    required = cfg.selected_steps
    remaining = [step for step in required if step not in completed]
    print("RESUME_STEPS", remaining, flush=True)
    if remaining:
        cfg.steps = remaining
        with dask.config.set(scheduler="synchronous"):
            main(cfg)
    receipt = dict(
        prior_log=str(prior),
        prior_completed=sorted(completed),
        resumed=remaining,
        required=required,
    )
    (cfg.output_path / "RESUME_COMPLETE.json").write_text(json.dumps(receipt, indent=2))
    (cfg.output_path / "COMPLETE.txt").write_text(
        "All required native steps completed across recorded jobs\n"
    )
