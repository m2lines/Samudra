# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import logging
import time
from collections import OrderedDict

import torch

from ocean_emulators.aggregator import Aggregator
from ocean_emulators.backend import init_eval_backend
from ocean_emulators.config import EvalConfig
from ocean_emulators.constants import (
    BoundaryVarNames,
    Grid,
    PrognosticVarNames,
    TensorMap,
)
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.stepper import run_rollout
from ocean_emulators.utils.data import (
    Normalize,
    get_inference_steps,
    spherical_area_weights,
)
from ocean_emulators.utils.device import using_gpu
from ocean_emulators.utils.distributed import is_main_process, set_seed
from ocean_emulators.utils.logging import (
    get_model_summary,
    handle_logging,
    handle_warnings,
)
from ocean_emulators.utils.wandb import WandBLogger

logger = logging.getLogger(__name__)


class Eval:
    """Evaluation pipeline for ocean emulator models.

    Runs long autoregressive rollouts and computes metrics against ground-truth
    ocean states.
    """

    def __init__(self, cfg: EvalConfig) -> None:
        cfg.prepare_output_dirs()

        self.device = init_eval_backend(cfg.backend)

        # Adjust workers and memory pinning based on device
        data_num_workers = cfg.data.loading.num_pytorch_workers()
        if not using_gpu():
            data_num_workers = 0  # Disable multi-processing on CPU
        elif cfg.disk_mode:
            data_num_workers = torch.cuda.device_count() * data_num_workers

        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting prognostic and boundary variables
        self.dataset_spec = cfg.data.dataset.build()
        self.prognostic_var_names: PrognosticVarNames = (
            self.dataset_spec.prognostic_var_names
        )
        self.boundary_var_names: BoundaryVarNames = self.dataset_spec.boundary_var_names
        self.levels = self.dataset_spec.num_prognostic_depth_levels

        str_prognostics = ", ".join([i for i in self.prognostic_var_names])
        str_boundaries = ", ".join([i for i in self.boundary_var_names])

        logger.info(f"Prognostic variables: {str_prognostics}")
        logger.info(f"Boundary variables: {str_boundaries}")
        logger.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_var_names)
        self.N_prog = len(self.prognostic_var_names)

        self.num_prog_in = int((cfg.data.hist + 1) * self.N_prog)
        self.num_boundary_in = int((cfg.data.hist + 1) * self.N_bound)
        self.num_in = self.num_prog_in + self.num_boundary_in
        self.num_out = self.num_prog_in

        self.tensor_map = TensorMap(dataset_spec=self.dataset_spec).to(self.device)

        logger.info(f"Number of inputs (prognostic + boundary): {self.num_in}")
        logger.info(f"Number of outputs (prognostic): {self.num_out}")

        # Dataloaders
        logger.info(f"Loading data")
        self.data_container = cfg.data.build(
            cfg.experiment.resolved_data_root,
        )

        self.src = self.data_container.inference_source
        self.data = self.src.data
        self.static_data = self.data_container.static_data
        self.metadata = self.src.metadata
        self.wet = self.src.masks.prognostic_with_hist(cfg.data.hist)
        self.area_weights: Grid = spherical_area_weights(self.data)
        self.area_weights = self.area_weights.to(self.device)

        self.normalize = Normalize(
            self.src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
        )

        # Model
        self.model = cfg.model.build(
            prog_channels=self.num_prog_in,
            boundary_channels=self.num_boundary_in,
            out_channels=self.num_out,
            hist=cfg.data.hist,
            static_data_for_corrector=self.static_data,
            srcs=self.data_container.sources,
            tensor_map=self.tensor_map,
            normalize=self.normalize,
            dataset_spec=self.dataset_spec,
        ).to(self.device)

        get_model_summary(self.model, None, cfg.debug)

        if cfg.ckpt_path is None:
            raise ValueError(
                "ckpt_path must be set; try --ckpt_path=path/to/checkpoint"
            )
        self.load_checkpoint(cfg.ckpt_path)

        self.network = self.model.__class__.__name__

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode == "online", is_main_process()
        )

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            None, cfg, data_container=self.data_container, finetune=False
        )

        # Eval
        self.hist = cfg.data.hist
        self.output_dir = cfg.experiment.output_dir
        self.debug = cfg.debug
        self.num_workers = data_num_workers
        self.inference_time = cfg.inference_time
        self.num_model_steps_forward = cfg.num_model_steps_forward
        self.save_zarr = cfg.save_zarr
        self.model_path = cfg.ckpt_path
        self.normalize_before_mask = cfg.data.normalize_before_mask
        self.masked_fill_value = cfg.data.masked_fill_value
        self.init_inference_store()

    def load_checkpoint(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        model_state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k.removeprefix("module.")
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    def init_inference_store(self):
        sliced_src = self.src.slice(self.inference_time)
        self.num_time_steps = get_inference_steps(
            sliced_src,
            hist=self.hist,
        )
        self.inference_dataset = InferenceDataset(
            src=sliced_src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            hist=self.hist,
            normalize_before_mask=self.normalize_before_mask,
            masked_fill_value=self.masked_fill_value,
            long_rollout=True,
        )

    def run(self) -> None:
        start_time = time.perf_counter()
        inf_stats = self.standalone_inference()
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
        self.finish()

    @torch.no_grad()
    def standalone_inference(self):
        self.model.eval()
        inf_aggregator = Aggregator.get_standalone_inference_aggregator(
            self.num_time_steps,
            self.metadata,
            self.hist,
            self.area_weights,
            self.src.masks.prognostic.to(self.device),
            self.num_out,
            self.tensor_map,
            self.normalize,
            self.prognostic_var_names,
        )

        run_rollout(
            model=self.model,
            dataset=self.inference_dataset,
            inf_aggregator=inf_aggregator,
            epoch=0,
            output_dir=self.output_dir,
            model_path=self.model_path,
            num_model_steps_forward=self.num_model_steps_forward,
            save_zarr=self.save_zarr,
            tensor_map=self.tensor_map,
            normalize=self.normalize,
        )
        logs = inf_aggregator.get_summary_logs()
        return {f"inference/{k}": v for k, v in logs.items()}

    def finish(self):
        self.wandb_logger.finish()


def main():
    cfg = EvalConfig.from_yaml_and_cli()
    cfg.prepare_output_dirs()  # we do this first so logging can use them

    handle_logging(cfg.debug, cfg.experiment.output_dir)
    handle_warnings()

    Evaluator = Eval(cfg)

    try:
        Evaluator.run()
    except Exception as e:
        # Log the exception with traceback
        logger.exception("Evaluation failed with an exception")
        raise e


if __name__ == "__main__":
    main()
