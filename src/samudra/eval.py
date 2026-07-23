# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import torch

from samudra.aggregator import Aggregator, InferenceEvaluatorAggregator
from samudra.constants import TensorMap
from samudra.datasets import InferenceDataset
from samudra.models.base import BaseModel
from samudra.stepper import run_rollout
from samudra.utils.data import DataContainer, DataSource, Normalize
from samudra.utils.distributed import is_main_process
from samudra.utils.logging import handle_logging, handle_warnings
from samudra.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


def load_model_checkpoint(
    model: BaseModel,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(
        {
            name.removeprefix("module."): value
            for name, value in checkpoint["model"].items()
        }
    )


@dataclass(frozen=True)
class InferenceAggregatorFactory:
    src: DataSource
    num_time_steps: int
    hist: int
    num_out: int
    tensor_map: TensorMap
    normalize: Normalize
    prognostic_var_names: list[str]

    def __call__(self) -> InferenceEvaluatorAggregator:
        device = self.tensor_map.dz.device
        return Aggregator.get_standalone_inference_aggregator(
            self.num_time_steps,
            self.src.metadata,
            self.hist,
            self.src.spherical_area_weights.to(device),
            self.src.masks.prognostic.to(device),
            self.num_out,
            self.tensor_map,
            self.normalize,
            self.prognostic_var_names,
        )


class Eval:
    """Evaluation pipeline for ocean emulator models.

    Runs a long autoregressive rollout and computes metrics against ground-truth
    ocean states. All dependencies are built before they are handed to this class.
    """

    def __init__(
        self,
        *,
        model: BaseModel,
        inference_dataset: InferenceDataset,
        inference_aggregator_factory: InferenceAggregatorFactory,
        num_model_steps_forward: int,
        tensor_map: TensorMap,
        normalize: Normalize,
        data_container: DataContainer,
    ) -> None:
        self.model = model
        self.inference_dataset = inference_dataset
        self.inference_aggregator_factory = inference_aggregator_factory
        self.num_model_steps_forward = num_model_steps_forward
        self.tensor_map = tensor_map
        self.normalize = normalize
        self.data_container = data_container

    def to(self, device: torch.device) -> Self:
        self.model.to(device)
        self.tensor_map.to(device)
        return self

    @torch.no_grad()
    def standalone_inference(
        self,
        *,
        output_dir: Path,
        model_path: Path,
        save_zarr: bool,
    ):
        self.model.eval()
        inference_aggregator = self.inference_aggregator_factory()
        run_rollout(
            model=self.model,
            dataset=self.inference_dataset,
            inf_aggregator=inference_aggregator,
            epoch=0,
            output_dir=output_dir,
            model_path=model_path,
            num_model_steps_forward=self.num_model_steps_forward,
            save_zarr=save_zarr,
            tensor_map=self.tensor_map,
            normalize=self.normalize,
        )
        logs = inference_aggregator.get_summary_logs()
        return {f"inference/{k}": v for k, v in logs.items()}


@dataclass(frozen=True)
class StandaloneEval:
    evaluator: Eval
    output_dir: Path
    model_path: Path
    save_zarr: bool
    wandb_logger: WandBLogger

    def run(self) -> None:
        start_time = time.perf_counter()
        inf_stats = self.evaluator.standalone_inference(
            output_dir=self.output_dir,
            model_path=self.model_path,
            save_zarr=self.save_zarr,
        )
        time_elapsed = time.perf_counter() - start_time

        log_stats = {
            **inf_stats,
            "eval_total_seconds": time_elapsed,
        }
        if is_main_process():
            self.wandb_logger.log(log_stats, step=None)

        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Eval time (Including wandb logging) {total_time_str}")
        self.wandb_logger.finish()


def main():
    from samudra.config import StandaloneEvalConfig

    cfg = StandaloneEvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()  # we do this first so logging can use them

    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    standalone_eval = cfg.build()

    try:
        standalone_eval.run()
    except Exception as e:
        # Log the exception with traceback
        logger.exception("Evaluation failed with an exception")
        raise e


if __name__ == "__main__":
    main()
