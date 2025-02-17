import argparse
import datetime
import logging
import os
import time

import torch
import xarray as xr

from aggregator import Aggregator
from config import EvalConfig
from constants import EXTRA_VARS, INPT_VARS, OUT_VARS, TensorMap, construct_metadata
from datasets import InferenceDataset
from models.unet import UNet
from stepper import Stepper
from utils.data import (
    Normalize,
    extract_wet_mask,
    get_time_slice,
    spherical_area_weights,
)
from utils.device import get_device, using_gpu
from utils.distributed import is_main_process, set_seed
from utils.logging import handle_logging, handle_warnings
from utils.model import get_model_summary
from utils.wandb import WandBLogger


class Eval:
    def __init__(self, cfg) -> None:
        self.device = get_device()

        # Adjust workers and memory pinning based on device
        if not using_gpu():
            cfg.data.num_workers = 0  # Disable multi-processing on CPU
            cfg.pin_mem = False
        elif cfg.disk_mode:
            cfg.data.num_workers = torch.cuda.device_count() * cfg.data.num_workers
            cfg.pin_mem = True

        # Set seeds
        set_seed(cfg.experiment.rand_seed)

        # Getting input, extra input and output
        self.inputs = INPT_VARS[cfg.experiment.exp_num_in]
        self.extra_in = EXTRA_VARS[cfg.experiment.exp_num_extra]
        self.outputs = OUT_VARS[cfg.experiment.exp_num_out]

        assert self.inputs == self.outputs, "Input and output "
        "variables must be the same"

        levels = cfg.experiment.exp_num_in.split("_")[-1]
        if "all" in levels:
            self.levels = 19
        elif "2D" in levels:
            self.levels = 1
        else:
            self.levels = int(levels)

        self.str_in = "".join([i + "_" for i in self.inputs])
        self.str_ext = "".join([i + "_" for i in self.extra_in])
        self.str_out = "".join([i + "_" for i in self.outputs])

        logging.info(f"inputs: {self.str_in}")
        logging.info(f"extra inputs: {self.str_ext}")
        logging.info(f"outputs: {self.str_out}")
        logging.info(f"levels: {self.levels}")

        self.N_atm = len(self.extra_in)
        self.N_in = len(self.inputs)
        self.N_extra = self.N_atm  # Number of atmosphere variables
        self.N_out = len(self.outputs)

        self.num_in = int((cfg.data.hist + 1) * self.N_in + self.N_extra)
        self.num_out = int((cfg.data.hist + 1) * len(self.outputs))

        self.tensor_map = TensorMap.init_instance(cfg.experiment.exp_num_out)

        logging.info(f"Number of inputs: {self.num_in}")
        logging.info(f"Number of outputs: {self.num_out}")

        # Dataloaders
        logging.info(f"Loading data")
        assert cfg.data.depth_mode == "surface" or cfg.data.depth_mode == "all"
        self.data_dir = cfg.experiment.data_dir
        self.data_path = cfg.data.data_path
        self.data_means_path = cfg.data.data_means_path
        self.data_stds_path = cfg.data.data_stds_path
        self.scaling_residuals_file = cfg.data.scaling_residuals_file

        if "*" in self.data_path:
            self.data = xr.open_mfdataset(
                os.path.join(self.data_dir, self.data_path),
                engine="netcdf4",
                chunks={"time": 1, "lat": 180, "lon": 360},
            )
        else:
            self.data = xr.open_zarr(
                os.path.join(self.data_dir, self.data_path), chunks={}
            )
        self.data_mean = xr.open_dataset(
            os.path.join(self.data_dir, self.data_means_path),
            engine="netcdf4",
            chunks={},
        )
        self.data_std = xr.open_dataset(
            os.path.join(self.data_dir, self.data_stds_path),
            engine="netcdf4",
            chunks={},
        )

        self.metadata = construct_metadata(self.data)
        self.wet, self.wet_surface = extract_wet_mask(
            self.data, self.outputs, cfg.data.hist
        )
        wet_without_hist, _ = extract_wet_mask(self.data, self.outputs, 0)
        self.area_weights = spherical_area_weights(self.data)
        self.area_weights = self.area_weights.to(self.device)

        self.normalize = Normalize.init_instance(
            self.data_mean,
            self.data_std,
            self.inputs,
            self.extra_in,
            self.outputs,
            wet_without_hist,
        )

        # Model
        logging.info(f"Getting model {cfg.experiment.network}")
        if "convnextunet" == cfg.experiment.network:
            if cfg.unet.ch_width[0] != self.num_in:
                logging.info(
                    f"NOTE: Changing input channels to match data "
                    f"{cfg.unet.ch_width[0]}->{self.num_in}"
                )
                cfg.unet.ch_width[0] = self.num_in
            if cfg.unet.n_out != self.num_out:
                logging.info(
                    f"NOTE: Changing output channels to match data "
                    f"{cfg.unet.n_out}->{self.num_out}"
                )
                cfg.unet.n_out = self.num_out
            model = UNet(cfg.unet, hist=cfg.data.hist, wet=self.wet.to(self.device)).to(
                self.device
            )
        else:
            raise NotImplementedError

        get_model_summary(model, self.num_in)

        model.load_state_dict(
            torch.load(cfg.ckpt_path, map_location=torch.device("cuda"))["model"]
        )

        self.model = model

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

    def init_inference_store(self):
        time_slice_with_initial_condition, self.num_time_steps = get_time_slice(
            self.inference_time,
            time_delta=self.time_delta,
            hist=self.hist,
        )
        inference_data = self.data.sel(time=time_slice_with_initial_condition)
        self.inference_dataset = InferenceDataset(
            inference_data,
            self.inputs,
            self.extra_in,
            self.outputs,
            self.wet,
            self.wet_surface,
            self.hist,
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

    if not os.path.exists(cfg.experiment.output_dir):
        os.makedirs(cfg.experiment.output_dir, exist_ok=True)

    handle_logging(cfg)
    handle_warnings()

    Evaluator = Eval(cfg)
    Evaluator.run()


if __name__ == "__main__":
    main()
