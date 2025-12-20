from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import optuna

from ocean_emulators.config import TrainConfig
from ocean_emulators.sweep_config import (
    CategoricalParamConfig,
    FloatParamConfig,
    IntParamConfig,
    SweepConfig,
)
from ocean_emulators.train import Trainer
from ocean_emulators.utils.logging import handle_logging, handle_warnings

logger = logging.getLogger(__name__)


def _get_sweep_config_path(args: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("config", type=str)
    parsed, _ = parser.parse_known_args(args)
    return Path(parsed.config).expanduser().resolve()


def _deep_update(target: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if (
            isinstance(value, dict)
            and isinstance(target.get(key), dict)
            and target.get(key) is not None
        ):
            _deep_update(target[key], value)
        else:
            target[key] = value


def _set_path_value(target: dict[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _build_train_config(
    base_cfg: TrainConfig,
    base_overrides: dict[str, Any],
    trial_overrides: dict[str, Any],
) -> TrainConfig:
    cfg_dict = base_cfg.model_dump()
    _deep_update(cfg_dict, base_overrides)
    for path, value in trial_overrides.items():
        _set_path_value(cfg_dict, path, value)
    return TrainConfig.model_validate(cfg_dict)


def _suggest_param(trial: optuna.Trial, param) -> Any:
    match param:
        case FloatParamConfig(
            name=name, low=low, high=high, log=log, step=step
        ):
            return trial.suggest_float(name, low, high, log=log, step=step)
        case IntParamConfig(
            name=name, low=low, high=high, step=step, log=log
        ):
            return trial.suggest_int(name, low, high, step=step, log=log)
        case CategoricalParamConfig(name=name, choices=choices):
            return trial.suggest_categorical(name, choices)
        case _:
            raise ValueError(f"Unsupported parameter config: {param}")


def _build_study(sweep_cfg: SweepConfig) -> optuna.Study:
    sampler: optuna.samplers.BaseSampler
    match sweep_cfg.study.sampler:
        case "tpe":
            sampler = optuna.samplers.TPESampler(seed=sweep_cfg.study.seed)
        case "random":
            sampler = optuna.samplers.RandomSampler(seed=sweep_cfg.study.seed)
        case _:
            raise ValueError(f"Unsupported sampler: {sweep_cfg.study.sampler}")

    return optuna.create_study(
        study_name=sweep_cfg.study.name,
        direction=sweep_cfg.study.direction,
        sampler=sampler,
        storage=sweep_cfg.study.storage,
        load_if_exists=sweep_cfg.study.load_if_exists,
    )


def _reset_logging_handlers() -> None:
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)


def main():
    sweep_config_path = _get_sweep_config_path()
    sweep_cfg = SweepConfig.from_yaml_and_cli()

    base_config_path = sweep_cfg.base_config
    if not base_config_path.is_absolute():
        base_config_path = (sweep_config_path.parent / base_config_path).resolve()

    base_cfg = TrainConfig.from_yaml_and_cli([str(base_config_path)])

    handle_warnings()

    study = _build_study(sweep_cfg)

    def objective(trial: optuna.Trial) -> float:
        trial_overrides = {
            param.path: _suggest_param(trial, param)
            for param in sweep_cfg.parameters
        }

        trial_name = sweep_cfg.trial_name_template.format(
            study_name=sweep_cfg.study.name,
            trial_number=trial.number,
            base_name=base_cfg.experiment.name,
        )

        trial_overrides.update(
            {
                "experiment.base_output_dir": sweep_cfg.output_dir,
                "experiment.name": trial_name,
            }
        )

        trial_cfg = _build_train_config(
            base_cfg, sweep_cfg.base_overrides, trial_overrides
        )

        trial_cfg.prepare_output_dirs()
        _reset_logging_handlers()
        handle_logging(trial_cfg.debug, trial_cfg.experiment.output_dir)

        trainer = Trainer(trial_cfg)
        trainer.run()

        if sweep_cfg.objective == "val":
            return trainer.best_val_loss
        if sweep_cfg.objective == "inference":
            if trainer.best_inf_loss >= 1e8:
                raise ValueError(
                    "Inference objective requested, but inference did not run."
                )
            return trainer.best_inf_loss
        raise ValueError(f"Unsupported objective: {sweep_cfg.objective}")

    study.optimize(
        objective,
        n_trials=sweep_cfg.study.n_trials,
        timeout=sweep_cfg.study.timeout_seconds,
        n_jobs=sweep_cfg.study.n_jobs,
        gc_after_trial=True,
    )

    logger.info("Best value: %.6f", study.best_value)
    logger.info("Best params: %s", study.best_params)


if __name__ == "__main__":
    main()
