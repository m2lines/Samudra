import datetime
import logging
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from ocean_emulators.aggregator import Aggregator
from ocean_emulators.backend import init_eval_backend
from ocean_emulators.config import EvalAblationConfig, EvalConfig
from ocean_emulators.constants import (
    BOUNDARY_VARS,
    PROGNOSTIC_VARS,
    BoundaryVarNames,
    Grid,
    PrognosticVarNames,
    TensorMap,
    construct_metadata,
)
from ocean_emulators.datasets import InferenceDataset
from ocean_emulators.models.modules import ConvNeXtBlock, UNetBackbone
from ocean_emulators.stepper import Stepper
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


def parse_skip_indices(raw_indices: str) -> set[int]:
    if raw_indices.strip() == "":
        return set()

    indices: set[int] = set()
    for raw_index in raw_indices.split(","):
        raw_index = raw_index.strip()
        if raw_index == "":
            continue
        try:
            indices.add(int(raw_index))
        except ValueError as error:
            raise ValueError(
                "ablation.unet_skip_indices must be a comma-separated list of "
                f"integers; got `{raw_indices}`."
            ) from error
    return indices


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
            self.levels = 51
        else:
            self.levels = int(levels)

        str_prognostics = ", ".join([i for i in self.prognostic_var_names])
        str_boundaries = ", ".join([i for i in self.boundary_var_names])

        logger.info(f"Prognostic variables: {str_prognostics}")
        logger.info(f"Boundary variables: {str_boundaries}")
        logger.info(f"Levels: {self.levels}")

        self.N_bound = len(self.boundary_var_names)
        self.N_prog = len(self.prognostic_var_names)

        self.num_in = int((cfg.data.hist + 1) * (self.N_prog + self.N_bound))
        self.num_out = int((cfg.data.hist + 1) * self.N_prog)

        self.tensor_map = TensorMap.init_instance(
            cfg.experiment.prognostic_vars_key, cfg.experiment.boundary_vars_key
        )

        logger.info(f"Number of inputs (prognostic + boundary): {self.num_in}")
        logger.info(f"Number of outputs (prognostic): {self.num_out}")

        # Dataloaders
        logger.info(f"Loading data")
        self.data_container = cfg.data.build(
            cfg.experiment.resolved_data_root,
            self.prognostic_var_names,
            self.boundary_var_names,
        )

        self.src = self.data_container.source_using_dask
        self.data = self.src.data
        self.static_data = self.data_container.static_data
        self.metadata = construct_metadata(self.data)
        self.wet = self.src.masks.prognostic_with_hist(cfg.data.hist)
        self.area_weights: Grid = spherical_area_weights(self.data)
        self.area_weights = self.area_weights.to(self.device)

        self.normalize = Normalize.init_instance(
            self.src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
        )

        # Model
        self.model = cfg.model.build(
            in_channels=self.num_in,
            out_channels=self.num_out,
            hist=cfg.data.hist,
            wet=self.wet.to(self.device),
            area_weights=self.area_weights,
            static_data=self.static_data,
            lat=torch.from_numpy(self.data.lat.values),
            lon=torch.from_numpy(self.data.lon.values),
        ).to(self.device)

        get_model_summary(self.model, None, cfg.debug)

        if cfg.ckpt_path is None:
            raise ValueError(
                "ckpt_path must be set; try --ckpt_path=path/to/checkpoint"
            )
        self.load_checkpoint(cfg.ckpt_path)
        self.apply_ablation(cfg.ablation)

        self.network = self.model.__class__.__name__

        # Initialize WandB
        self.wandb_logger = WandBLogger.init_instance()
        self.wandb_logger.configure(
            cfg.experiment.wandb.mode != "disabled", is_main_process()
        )

        # Set up wandb run
        self.wandb_id, self.wandb_name = self.wandb_logger.setup_run(
            None, cfg, data_container=self.data_container, finetune=False
        )

        # Eval
        self.hist = cfg.data.hist
        self.output_dir = cfg.experiment.output_dir
        self.debug = cfg.debug
        self.num_workers = cfg.data.num_workers
        self.inference_time = cfg.inference_time
        self.num_model_steps_forward = cfg.num_model_steps_forward
        self.inference_stride = cfg.inference_stride
        self.save_zarr = cfg.save_zarr
        self.model_path = cfg.ckpt_path
        self.normalize_before_mask = cfg.data.normalize_before_mask
        self.masked_fill_value = cfg.data.masked_fill_value
        self.init_inference_store()
        self.resume_prediction_zarr = (
            Path(cfg.resume_prediction_zarr)
            if cfg.resume_prediction_zarr is not None
            else None
        )
        self.resume_steps = 0
        self.resume_initial_prognostic: torch.Tensor | None = None
        if self.resume_prediction_zarr is not None:
            self.resume_steps, self.resume_initial_prognostic = self.load_resume_state(
                self.resume_prediction_zarr
            )

    def load_resume_state(self, prediction_zarr: Path) -> tuple[int, torch.Tensor]:
        """Load and validate the final state of an existing flat prediction store."""
        if not prediction_zarr.is_dir():
            raise FileNotFoundError(
                f"Resume prediction zarr does not exist: {prediction_zarr}"
            )

        try:
            predictions = xr.open_zarr(prediction_zarr, consolidated=True)
        except (KeyError, ValueError):
            predictions = xr.open_zarr(prediction_zarr, consolidated=False)

        expected_variables = list(self.prognostic_var_names)
        missing = [name for name in expected_variables if name not in predictions]
        if missing:
            hint = " This looks like a repacked predictions_4d.zarr store." if "Eta" in missing else ""
            raise ValueError(
                "Resume requires the flat predictions.zarr store with every "
                f"prognostic channel; missing {missing[:5]}.{hint}"
            )
        if predictions.sizes.get("time", 0) == 0:
            raise ValueError(f"Resume prediction zarr is empty: {prediction_zarr}")

        saved_prediction_count = int(predictions.sizes["time"])
        output_times_per_step = self.hist + 1
        if saved_prediction_count % output_times_per_step != 0:
            raise ValueError(
                "Resume prediction zarr ends with an incomplete model output: "
                f"{saved_prediction_count} saved times is not divisible by "
                f"hist + 1 ({output_times_per_step})."
            )
        resume_steps = saved_prediction_count // output_times_per_step
        if resume_steps >= len(self.inference_dataset):
            raise ValueError(
                "Resume prediction zarr already contains "
                f"{saved_prediction_count} saved predictions ({resume_steps} model "
                "steps), but this inference window has only "
                f"{len(self.inference_dataset)} steps. Increase inference_time.end."
            )

        expected_times = self.inference_dataset.get_target_time(0, resume_steps)
        stored_times = predictions.time.values
        if not np.array_equal(stored_times, expected_times.values):
            raise ValueError(
                "Resume prediction times do not match the requested inference "
                "window. Use the original inference_time.start and stride, and "
                "only extend inference_time.end."
            )

        final_history = np.stack(
            [
                predictions[name]
                .isel(time=slice(-output_times_per_step, None))
                .values
                for name in expected_variables
            ],
            axis=1,
        )
        initial_history = torch.from_numpy(final_history).float()
        initial_history = self.normalize.normalize_tensor_prognostic(
            initial_history,
            fill_value=self.masked_fill_value,
        )
        initial_history = torch.where(
            self.inference_dataset.wet,
            initial_history,
            self.masked_fill_value,
        )
        initial_prognostic = initial_history.unsqueeze(0).flatten(start_dim=1, end_dim=2)
        logger.info(
            "Resuming inference from %s after %s saved predictions (%s model "
            "steps); %s steps remain.",
            prediction_zarr,
            saved_prediction_count,
            resume_steps,
            len(self.inference_dataset) - resume_steps,
        )
        return resume_steps, initial_prognostic

    def load_checkpoint(self, ckpt_path: str):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(self.device))
        model_state_dict = checkpoint["model"]
        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            name = k.removeprefix("module.")
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    def apply_ablation(self, cfg: EvalAblationConfig) -> None:
        skip_indices = parse_skip_indices(cfg.unet_skip_indices)
        unet_count = 0
        convnext_count = 0

        for module in self.model.modules():
            if isinstance(module, UNetBackbone):
                module.configure_ablation(
                    disable_middle_block=cfg.disable_unet_middle_block,
                    skip_mode=cfg.unet_skip_mode,
                    skip_indices=skip_indices,
                )
                unet_count += 1
            if isinstance(module, ConvNeXtBlock):
                module.disable_residual = cfg.disable_convnext_block_residuals
                convnext_count += 1

        logger.info(
            "Eval ablation: disable_unet_middle_block=%s, "
            "unet_skip_mode=%s, unet_skip_indices=%s, "
            "disable_convnext_block_residuals=%s",
            cfg.disable_unet_middle_block,
            cfg.unet_skip_mode,
            sorted(skip_indices),
            cfg.disable_convnext_block_residuals,
        )
        logger.info(
            "Applied ablation settings to %s UNetBackbone module(s) and "
            "%s ConvNeXtBlock module(s).",
            unet_count,
            convnext_count,
        )

    def init_inference_store(self):
        sliced_src = self.src.slice(self.inference_time)
        self.num_time_steps = get_inference_steps(
            sliced_src,
            hist=self.hist,
            inference_stride=self.inference_stride,
        )
        self.inference_dataset = InferenceDataset(
            src=sliced_src,
            prognostic_var_names=self.prognostic_var_names,
            boundary_var_names=self.boundary_var_names,
            hist=self.hist,
            normalize_before_mask=self.normalize_before_mask,
            masked_fill_value=self.masked_fill_value,
            long_rollout=True,
            inference_stride=self.inference_stride,
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
            self.prognostic_var_names,
        )

        Stepper.inference(
            model=self.model,
            dataset=self.inference_dataset,
            inf_aggregator=inf_aggregator,
            epoch=0,
            output_dir=self.output_dir,
            model_path=self.model_path,
            num_model_steps_forward=self.num_model_steps_forward,
            save_zarr=self.save_zarr,
            resume_steps=self.resume_steps,
            resume_initial_prognostic=self.resume_initial_prognostic,
            resume_prediction_zarr=self.resume_prediction_zarr,
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
