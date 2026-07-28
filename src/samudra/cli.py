# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Console-script entry point: ``samudra <train|eval|viz> CONFIG [OVERRIDES...]``.

Installed as the ``samudra`` command (see ``[project.scripts]`` in
pyproject.toml), so a user who ``pip install samudra`` can run
``samudra train path/to/config.yaml`` without cloning the repo. Each
subcommand forwards to the same ``main`` the module entry points use
(``python -m samudra.train`` etc.), so behaviour is identical.
"""

import sys

_COMMANDS = ("train", "eval", "viz")


def main() -> None:
    """Dispatch ``samudra <command> ...`` to the matching task entry point."""
    argv = sys.argv[1:]
    if not argv or argv[0] not in _COMMANDS:
        sys.stderr.write(
            f"usage: samudra {{{'|'.join(_COMMANDS)}}} CONFIG [OVERRIDES...]\n"
        )
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
    else:  # viz
        from samudra.viz.config import VizConfig
        from samudra.viz.config import main as viz_main

        viz_main(VizConfig.from_yaml_and_cli())


if __name__ == "__main__":
    main()
