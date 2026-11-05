# SPDX-FileCopyrightText: 2026 Samudra Authors
# SPDX-License-Identifier: Apache-2.0
"""Generate reviewable experiment configs; this module never submits jobs."""

import argparse
from pathlib import Path

from samudra.observations.config import ObservationTrainConfig
from samudra.observations.qualify import QualificationConfig

ARMS = {
    "A-obs-small": ("obs", False, 4),
    "B-mixed-small": ("om4-obs", False, 4),
    "C-obs-large": ("obs", True, 4),
    "D-mixed-large": ("om4-obs", True, 4),
    "E-mixed-large-no-replay": ("om4-obs", True, 8),
}


def screen_configs(base, *, qualification=False):
    result = {}
    for name, (mix, large, fresh) in ARMS.items():
        values = base.model_dump(mode="json")
        values.update(pretrain_mix=mix, fresh_examples=fresh)
        values["experiment"]["name"] = name
        if large:
            backbone = (
                base.model.evolution_backbone or base.model.backbone
            ).model_dump(mode="json")
            values["model"]["initializer_backbone"] = backbone
        config_type = ObservationTrainConfig
        if qualification:
            values.update(
                main_updates=4,
                post_training_updates=2,
                milestone_updates=[4],
                validation_freq=6,
                save_freq=1,
                measure_timing=True,
                qualify_chains=name == "D-mixed-large",
                qualify_data=name == "D-mixed-large",
            )
            config_type = QualificationConfig
        result[name] = config_type.model_validate(values)
    return result


def winner_configs(winner):
    """Continue the uncooled 4k state; branch isolated cooldowns at 8k and 12k.

    The screen already supplies the 4k cooldown. The main continuation finishes
    with its 16k cooldown, for 20k total updates including all four cooldowns.
    """
    root = winner.experiment.base_output_dir
    name = winner.experiment.name
    values = winner.model_dump(mode="json")
    values.update(
        main_updates=16000,
        milestone_updates=[8000, 12000, 16000],
        resume_ckpt_path=None,
        fork_ckpt_path=str(winner.experiment.nets_dir / "main.pt"),
    )
    values["experiment"]["name"] = name + "-main16k"
    main = ObservationTrainConfig.model_validate(values)
    result = {main.experiment.name: main}
    for update in (8000, 12000):
        branch = main.model_dump(mode="json")
        branch.update(
            main_updates=update,
            milestone_updates=[update],
            fork_ckpt_path=str(
                Path(root) / main.experiment.name / "saved_nets" / f"main-{update}.pt"
            ),
        )
        branch["experiment"]["name"] = name + f"-cooldown{update}"
        result[branch["experiment"]["name"]] = ObservationTrainConfig.model_validate(
            branch
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--winner", action="store_true")
    args = parser.parse_args()
    base = ObservationTrainConfig.from_yaml(args.config)
    configs = (
        winner_configs(base)
        if args.winner
        else screen_configs(base, qualification=args.qualification)
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs.items():
        cfg.save_yaml(args.output / (name + ".yaml"))


if __name__ == "__main__":
    main()
