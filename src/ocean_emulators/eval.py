import argparse
import datetime
import logging
import os
import time
from collections import OrderedDict

import torch
import xarray as xr

from ocean_emulators.aggregator import Aggregator
from ocean_emulators.backend import init_eval_backend
from ocean_emulators.config import EvalConfig
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    PROGNOSTIC_VARS,
    BoundaryVarNames,
    PrognosticVarNames,
    TensorMap,
    construct_metadata,
)
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.models.samudra import Samudra
from ocean_emulators.stepper import Stepper
from ocean_emulators.utils.data import (
    Normalize,
    extract_wet_mask,
    get_inference_steps,
    spherical_area_weights,
    validate_data,
)
from ocean_emulators.utils.device import using_gpu
from ocean_emulators.utils.distributed import is_main_process, set_seed
from ocean_emulators.utils.logging import handle_logging, handle_warnings
from ocean_emulators.utils.model import get_model_summary
from ocean_emulators.utils.wandb import WandBLogger


class Eval:
    def __init__(self, cfg: EvalConfig) -> None:
        cfg.prepare_output_dirs()

        self.device = init_eval_backend(cfg.backend)

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.data.num_workers = 0  # Disable multi-processing on CPU
        elif cfg.disk_mode:
            cfg.data.num_workers = torch.cuda.device_count() * cfg.data.num_workers

        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting prognostic and boundary variables
        self.prognostic_var_names: PrognosticVarNames = PROGNOSTIC_VARS[
            cfg.experiment.prognostic_vars_key
        ]
        self.boundary_var_names: BoundaryVarNames = BOUNDARY_VARS[
            cfg.experiment.boundary_vars_key
        ]

        levels = cfg.experiment.prognostic_vars_key.split("_")[-1]
        if "all" in levels:
            self.levels = 19
        else:
            self.levels = int(levels)

        str_prognostics = ", ".join([i for i in self.prognostic_var_names])
        str_boundaries = ", ".join([i for i in self.boundary_var_names])

        logging.info(f"Prognostic variables: {str_prognostics}")
        logging.info(f"Boundary variables: {str_boundaries}")
        logging.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_var_names)
        self.N_prog = len(self.prognostic_var_names)

        self.num_in = int((cfg.data.hist + 1) * (self.N_prog + self.N_bound))
        self.num_out = int((cfg.data.hist + 1) * self.N_prog)

        self.tensor_map = TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        logging.info(f"Number of inputs (prognostic + boundary): {self.num_in}")
        logging.info(f"Number of outputs (prognostic): {self.num_out}")

        # Dataloaders
        logging.info(f"Loading data")
        self.data_dir = cfg.experiment.data_dir
        self.data_path = cfg.data.data_path
        self.data_means_path = cfg.data.data_means_path
        self.data_stds_path = cfg.data.data_stds_path
        self.scaling_residuals_file = cfg.data.scaling_residuals_file

        if "*" in self.data_path:
            data = xr.open_mfdataset(
                os.path.join(self.data_dir, self.data_path),
                engine="netcdf4",
                chunks={"time": 1, "lat": 180, "lon": 360},
            )
        else:
            data = xr.open_zarr(os.path.join(self.data_dir, self.data_path), chunks={})
        data_mean = xr.open_dataset(
            os.path.join(self.data_dir, self.data_means_path),
            engine="netcdf4" if self.data_means_path.endswith(".nc") else "zarr",
            chunks={},
        )
        data_std = xr.open_dataset(
            os.path.join(self.data_dir, self.data_stds_path),
            engine="netcdf4" if self.data_stds_path.endswith(".nc") else "zarr",
            chunks={},
        )

        self.data, self.data_mean, self.data_std = validate_data(
            data, data_mean, data_std
        )

        self.metadata = construct_metadata(self.data)
        self.wet, self.wet_surface = extract_wet_mask(
            self.data, self.prognostic_var_names, cfg.data.hist
        )
        wet_without_hist, _ = extract_wet_mask(self.data, self.prognostic_var_names, 0)
        self.area_weights = spherical_area_weights(self.data)
        self.area_weights = self.area_weights.to(self.device)

        self.normalize = Normalize.init_instance(
            data_mean=self.data_mean,
            data_std=self.data_std,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            wet_mask=wet_without_hist,
        )

        # Model
        logging.info(f"Getting model {cfg.experiment.network}")
        if "Samudra" == cfg.experiment.network:
            if cfg.samudra.ch_width[0] != self.num_in:
                logging.info(
                    f"NOTE: Changing input channels to match data "
                    f"{cfg.samudra.ch_width[0]}->{self.num_in}"
                )
                cfg.samudra.ch_width[0] = self.num_in
            if cfg.samudra.n_out != self.num_out:
                logging.info(
                    f"NOTE: Changing output channels to match data "
                    f"{cfg.samudra.n_out}->{self.num_out}"
                )
                cfg.samudra.n_out = self.num_out
            model = Samudra(
                cfg.samudra, hist=cfg.data.hist, wet=self.wet.to(self.device)
            ).to(self.device)
        else:
            raise NotImplementedError

        get_model_summary(model, self.num_in)

        self.model = model
        self.load_checkpoint(cfg.ckpt_path)

        self.network = cfg.experiment.network

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode == "online", is_main_process()
        )

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            None, cfg, finetune=False
        )

        # Eval
        self.hist = cfg.data.hist
        self.output_dir = cfg.experiment.output_dir
        self.network = cfg.experiment.network
        self.debug = cfg.debug
        self.num_workers = cfg.data.num_workers
        self.inference_time = cfg.inference
        self.time_delta = cfg.data.time_delta
        self.record_every = cfg.record_every
        self.num_model_steps_forward = cfg.num_model_steps_forward
        self.save_zarr = cfg.save_zarr
        self.model_path = cfg.ckpt_path
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
        self.num_time_steps = get_inference_steps(
            self.inference_time,
            time_delta=self.time_delta,
            hist=self.hist,
        )
        inference_data = self.data.sel(time=self.inference_time.time_slice)
        self.inference_dataset = InferenceDataset(
            data=inference_data,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            wet=self.wet,
            wet_surface=self.wet_surface,
            hist=self.hist,
            long_rollout=True,
        )

    def run(self) -> None:
        start_time = time.time()
        inf_stats = self.standalone_inference()
        time_elapsed = time.time() - start_time

        log_stats = {
            **inf_stats,
            "eval_total_seconds": time_elapsed,
        }

        if is_main_process():
            self.wandb_logger.log(log_stats, step=None)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info(f"Eval time (Including wandb logging) {total_time_str}")
        self.finish()

    @torch.no_grad()
    def standalone_inference(self):
        self.model.eval()
        inf_aggregator = Aggregator.get_standalone_inference_aggregator(
            self.num_time_steps,
            self.metadata,
            self.hist,
            self.area_weights,
            self.num_out,
        )

        Stepper.inference(
            model=self.model,
            dataset=self.inference_dataset,
            inf_aggregator=inf_aggregator,
            epoch=0,
            output_dir=self.output_dir,
            model_path=self.model_path,
            num_model_steps_forward=self.num_model_steps_forward,
            record_every=self.record_every,
            save_zarr=self.save_zarr,
        )
        logs = inf_aggregator.get_summary_logs()
        return {f"inference/{k}": v for k, v in logs.items()}

    def finish(self):
        self.wandb_logger.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--subname", type=str, required=False)
    parser.add_argument("--ckpt_path", type=str, required=False)
    parser.add_argument("--save_zarr", default=False, action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.subname:
        overrides["sub_name"] = args.subname
    if args.ckpt_path:
        print(args.ckpt_path)
        overrides["ckpt_path"] = args.ckpt_path
    if args.save_zarr:
        overrides["save_zarr"] = args.save_zarr

    # Load config from YAML
    cfg = EvalConfig.from_yaml(args.config, overrides)
    cfg.prepare_output_dirs()  # we do this first so logging can use them

    handle_logging(cfg)
    handle_warnings()

    Evaluator = Eval(cfg)

    try:
        Evaluator.run()
    except Exception as e:
        # Log the exception with traceback
        logging.exception("Evaluation failed with an exception")
        raise e


if __name__ == "__main__":
    main()
