# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Console entry point for Samudra's training, evaluation, and search tasks.

Installed as the ``samudra`` command (see ``[project.scripts]`` in
pyproject.toml), so a user who ``pip install samudra`` can run
``samudra train path/to/config.yaml`` without cloning the repo. Each
subcommand forwards to the same ``main`` the module entry points use
(``python -m samudra.train`` etc.), so behaviour is identical.
"""

import sys

_COMMANDS = ("train", "eval", "viz", "search")

_HELP = """\
samudra — train and evaluate emulators of ocean physics

Samudra is a PyTorch package for models that auto-regressively predict future
ocean states — temperature, salinity, horizontal currents, sea-surface height,
and surface heat flux — learned from the OM4 ocean model at 1°, 1/2°, and 1/4°
resolution. See https://arxiv.org/abs/2412.03795 for the method.

Usage:
  samudra <command> [ARGS ...]

Commands:
  train   Train a model from a config (checkpointing, W&B logging, multi-GPU).
  eval    Roll a trained model out autoregressively and collect metrics.
  viz     Render maps, time series, and PDFs from evaluation outputs.
  search  Plan and run successive-halving architecture searches on Slurm.

For train, eval, and viz, CONFIG is a YAML path or bundled preset such as
`samudra_om4/train.yaml`, with inline config overrides such as `--epochs 100`.
Search has its own `plan`, `start`, `run-task`, and `advance` subcommands. Run
`samudra <command> --help` for details.

Examples:
  samudra train samudra_om4/train.yaml --experiment.data_root ./data
  samudra eval  samudra_om4/eval.yaml  --ckpt_path ./checkpoint.pt
  samudra search plan search.yaml

Docs: https://m2lines.github.io/Samudra/docs/
"""


def main() -> None:
    """Dispatch ``samudra <command> ...`` to the matching task entry point."""
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        sys.stdout.write(_HELP)
        raise SystemExit(0)
    if argv[0] not in _COMMANDS:
        sys.stderr.write(f"samudra: unknown command {argv[0]!r}\n\n{_HELP}")
        raise SystemExit(2)

    command, rest = argv[0], argv[1:]
    # The config loaders parse sys.argv directly, so strip the subcommand token
    # and leave them an argv of `CONFIG [OVERRIDES...]`.
    sys.argv = [f"samudra {command}", *rest]

    if command == "train":
        from samudra.train import main as train_main

        train_main()
    elif command == "eval":
        from samudra.eval import main as eval_main

        eval_main()
    elif command == "viz":
        from samudra.viz.config import VizConfig
        from samudra.viz.config import main as viz_main

        viz_main(VizConfig.from_yaml_and_cli())
    else:  # search
        from samudra.search import main as search_main

        search_main()


if __name__ == "__main__":
    main()
