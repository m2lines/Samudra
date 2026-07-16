# SPDX-FileCopyrightText: 2026 Samudra Authors
#
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from wandb.data_types import WBValue

from samudra.utils.multiton import Multiton

# Metrics supported by wandb -- probably there are more possible types too
Metrics = Mapping[str, float | torch.Tensor | WBValue]

# Same as above but mutable when you're building something up
MetricsDict = dict[str, float | torch.Tensor | WBValue]

if TYPE_CHECKING:
    from samudra.config import AnyTopLevelConfig
    from samudra.utils.data import DataContainer


class WandBLogger(Multiton):
    def _initialize(self):
        self._enabled = False
        self._initialized = False
        self.run = None

    @property
    def enabled(self):
        return self._enabled

    def _make_config(self, cfg: "AnyTopLevelConfig", data_container: "DataContainer"):
        config = {
            f"data_{i}/attrs": src.data.attrs
            for i, src in enumerate(data_container.train_sources)
        }
        config.update(config=cfg.model_dump())
        return config

    def setup_run(
        self,
        checkpoint_path: str | None,
        cfg: "AnyTopLevelConfig",
        data_container: "DataContainer",
        finetune: bool = False,
    ):
        """Set up a wandb run, either resuming from checkpoint or creating new run.

        Args:
            checkpoint_path: Path to checkpoint file, if resuming
            cfg: Configuration object
            data_container: Data container to log attributes of
            finetune: Whether this is a finetuning run

        Returns:
            tuple: (wandb_id, wandb_name)
        """
        if not checkpoint_path:
            return self._init_new_run(cfg, data_container)

        if finetune:
            return self._init_new_run(cfg, data_container)

        if not self._enabled:
            return None, None

        # Load only on the rank that initializes W&B, and force CPU placement.
        # Checkpoints can contain CUDA tensors saved from rank 0; loading them
        # here without map_location makes every rank reserve memory on GPU 0
        # before the real per-rank checkpoint load happens.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        wandb_id = checkpoint.get("wandb_id")
        wandb_name = checkpoint.get("wandb_name")
        del checkpoint

        try:
            self.init(
                config=self._make_config(cfg, data_container),
                name=wandb_name,
                dir=cfg.experiment.output_dir,
                resume="must",
                id=wandb_id,
                **cfg.experiment.wandb.model_dump(),
            )
        except Exception:
            # If resume fails, start new run
            self.init(
                config=self._make_config(cfg, data_container),
                name=wandb_name,
                dir=cfg.experiment.output_dir,
                **cfg.experiment.wandb.model_dump(),
            )

        return wandb_id, wandb_name

    def _init_new_run(self, cfg: "AnyTopLevelConfig", data_container: "DataContainer"):
        """Initialize a new wandb run.

        Args:
            cfg: Configuration object
            data_container: Data container to log attributes of
        Returns:
            tuple: (None, generated_name) for new run
        """
        wandb_name = cfg.experiment.name
        if self._enabled:
            self.init(
                config=self._make_config(cfg, data_container),
                name=wandb_name,
                dir=cfg.experiment.output_dir,
                **cfg.experiment.wandb.model_dump(),
            )

            wandb_id = self.run.id if self.run else None
        else:
            wandb_id = None

        return wandb_id, wandb_name

    def finish(self):
        """Finish the wandb run."""
        if self._enabled:
            wandb.finish()

    def configure(self, enabled: bool, is_main_process: bool):
        """Configure whether wandb should be enabled."""
        assert self._enabled is False, "WandB is already initialized"
        self._enabled = enabled and is_main_process

    def init(self, **kwargs):
        """Initialize wandb run."""
        if self._enabled and not self._initialized:
            try:
                self.run = wandb.init(**kwargs)
                self._initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize wandb: {e}")
                self._enabled = False

    def watch(self, model, **kwargs):
        """Watch model parameters and gradients."""
        if self._enabled:
            wandb.watch(model, **kwargs)

    def log(self, metrics: Metrics, step: int | None, **kwargs):
        """Log metrics to wandb."""
        if self._enabled:
            # Really, this should take a mapping, not a dict
            # (so it is covariant) but it doens't so we convert
            wandb.log(dict(metrics), step=step, **kwargs)

    def Image(self, data, *args, **kwargs):
        if isinstance(data, np.ndarray):
            data = scale_image(data)
        return wandb.Image(data, *args, **kwargs)

    def Video(self, *args, **kwargs):
        return wandb.Video(*args, **kwargs)

    def Table(self, *args, **kwargs):
        return wandb.Table(*args, **kwargs)

    def Histogram(self, *args, **kwargs):
        return wandb.Histogram(*args, **kwargs)

    def log_inference_metrics(
        self,
        loss_value: float,
        loss_per_channel: torch.Tensor,
        outputs: list,
        depth_indices: dict,
        var_indices: dict,
        depth_set: list,
        var_set: list,
        step: int,
        predictions: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
        targets_unnormalized: torch.Tensor | None = None,
        model_pred_unnormalized: torch.Tensor | None = None,
    ):
        """Log validation metrics including depth and variable specific metrics."""
        if not self._enabled:
            return

        self.log({"eval/total_eval_loss_per_batch": loss_value}, step=step)

        # Loss per channel
        for i, var in enumerate(outputs):
            self.log({"eval/per_channel/" + var: loss_per_channel[i].item()}, step=step)

        # Loss per depth
        if all(x is not None for x in [depth_indices, depth_set]):
            for d in depth_set:
                self.log(
                    {
                        "eval/depth/depth_" + str(d) + "_loss": torch.mean(
                            loss_per_channel[depth_indices[d]]
                        ).item()
                    },
                    step=step,
                )

        # Loss per input variable
        if all(x is not None for x in [var_indices, var_set]):
            for k in var_set:
                self.log(
                    {
                        "eval/per_var/" + k + "_loss": torch.mean(
                            loss_per_channel[var_indices[k]]
                        ).item()
                    },
                    step=step,
                )

        # Plot predictions vs targets
        if (
            predictions is not None
            and targets is not None
            and targets_unnormalized is not None
            and model_pred_unnormalized is not None
        ):
            for i, var in enumerate(outputs):
                fig = plt.figure(figsize=(10, 5))
                plt.plot(
                    range(targets.shape[0]),
                    targets_unnormalized[:, :, :, i].mean(dim=(1, 2)),
                    label="Target",
                )
                min_val, max_val = plt.ylim()
                plt.plot(
                    range(predictions.shape[0]),
                    model_pred_unnormalized[:, :, :, i].mean(dim=(1, 2)),
                    label="Prediction",
                )

                # Adjust y-limits based on variable type
                if "thetao" in var:
                    plt.ylim(min_val - 0.25, max_val + 0.25)
                elif "so" in var:
                    plt.ylim(min_val - 0.2, max_val + 0.2)
                elif "KE" in var:
                    plt.ylim(min_val - 0.5, max_val + 0.5)

                plt.title(var)
                plt.legend()
                self.log({f"eval/plots/{var}": wandb.Image(fig)}, step=step)
                plt.close()


def scale_image(image_data):
    """
    Given an array of scalar data, rescale the data to the range [0, 255].
    """
    data_min = np.nanmin(image_data)
    data_max = np.nanmax(image_data)

    image_data = 255 * (image_data - data_min) / (data_max - data_min)
    image_data = np.minimum(image_data, 255)
    image_data = np.maximum(image_data, 0)
    image_data[np.isnan(image_data)] = 0

    return image_data


def get_record_to_wandb(label: str = ""):
    wandb = WandBLogger.get_instance()
    step = 0

    def record_logs(logs):
        nonlocal step
        for j, log in enumerate(logs):
            if len(log) > 0:
                if label != "":
                    log = {f"{label}/{k}": v for k, v in log.items()}
                wandb.log(log, step=step + j)
        step += len(logs)

    return record_logs
