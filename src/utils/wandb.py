# TODO: wandb is not working with mypy. I have put type ignores wherever needed.
import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import torch

import wandb


class WandBLogger:
    _instance: Optional["WandBLogger"] = None

    def __new__(cls, *args, **kwargs) -> "WandBLogger":
        # Prevent direct instantiation
        raise TypeError(
            "WandBLogger cannot be instantiated directly. Use init_instance() instead."
        )

    @classmethod
    def get_instance(cls) -> "WandBLogger":
        if cls._instance is None:
            raise ValueError("WandBLogger not initialized")
        return cls._instance

    @classmethod
    def init_instance(cls) -> "WandBLogger":
        if cls._instance is not None:
            raise ValueError("WandBLogger already initialized")

        instance = super().__new__(cls)
        instance._initialize()
        cls._instance = instance
        return cls._instance

    def _initialize(self):
        self._enabled = False
        self._initialized = False
        self.run = None

    @property
    def enabled(self):
        return self._enabled

    def setup_run(self, checkpoint_path: str, cfg: Any, finetune: bool = False):
        """Set up a wandb run, either resuming from checkpoint or creating new run.

        Args:
            checkpoint_path: Path to checkpoint file, if resuming
            cfg: Configuration object
            finetune: Whether this is a finetuning run

        Returns:
            tuple: (wandb_id, wandb_name)
        """
        if not checkpoint_path:
            return self._init_new_run(cfg)

        if finetune:
            return self._init_new_run(cfg)

        # Load checkpoint and try to resume
        checkpoint = torch.load(checkpoint_path)
        wandb_id = checkpoint.get("wandb_id")
        wandb_name = checkpoint.get("wandb_name")

        if self._enabled:
            try:
                self.init(
                    config=cfg.__dict__,
                    name=wandb_name,
                    dir=cfg.output_dir,
                    resume="must",
                    id=wandb_id,
                    **cfg.wandb.__dict__,
                )
            except Exception:
                # If resume fails, start new run
                self.init(
                    config=cfg.__dict__,
                    name=wandb_name,
                    dir=cfg.output_dir,
                    **cfg.wandb.__dict__,
                )

        return wandb_id, wandb_name

    def _init_new_run(self, cfg: Any):
        """Initialize a new wandb run.

        Args:
            cfg: Configuration object

        Returns:
            tuple: (None, generated_name) for new run
        """
        wandb_name = (
            cfg.name + "//" + cfg.sub_name
            if hasattr(cfg, "sub_name")
            else ".LOCAL" + "//" + cfg.name
        )

        if self._enabled:
            self.init(
                config=cfg.__dict__,
                name=wandb_name,
                dir=cfg.output_dir,
                **cfg.wandb.__dict__,
            )

            wandb_id = self.run.id if self.run else None
        else:
            wandb_id = None

        return wandb_id, wandb_name

    def finish(self):
        """Finish the wandb run."""
        if self._enabled:
            wandb.finish()  # type: ignore[attr-defined]

    def configure(self, enabled: bool, is_main_process: bool):
        """Configure whether wandb should be enabled."""
        assert self._enabled is False, "WandB is already initialized"
        self._enabled = enabled and is_main_process

    def init(self, **kwargs):
        """Initialize wandb run."""
        if self._enabled and not self._initialized:
            try:
                self.run = wandb.init(**kwargs)  # type: ignore[attr-defined]
                self._initialized = True
            except Exception as e:
                logging.error(f"Failed to initialize wandb: {e}")
                self._enabled = False

    def watch(self, model, **kwargs):
        """Watch model parameters and gradients."""
        if self._enabled:
            wandb.watch(model, **kwargs)  # type: ignore[attr-defined]

    def log(self, metrics: Dict[str, Any], step: int, **kwargs):
        """Log metrics to wandb."""
        if self._enabled:
            wandb.log(metrics, step=step, **kwargs)  # type: ignore[attr-defined]

    def log_inference_metrics(
        self,
        loss_value: float,
        loss_per_channel: torch.Tensor,
        outputs: list,
        depth_indices: Dict,
        var_indices: Dict,
        depth_set: list,
        var_set: list,
        step: int,
        predictions: torch.Tensor = None,
        targets: torch.Tensor = None,
        targets_unnormalized: torch.Tensor = None,
        model_pred_unnormalized: torch.Tensor = None,
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
                if k == "zos":  # Skip zos variable
                    continue
                self.log(
                    {
                        "eval/per_var/" + k + "_loss": torch.mean(
                            loss_per_channel[var_indices[k]]
                        ).item()
                    },
                    step=step,
                )

        # Plot predictions vs targets
        if all(
            x is not None
            for x in [
                predictions,
                targets,
                targets_unnormalized,
                model_pred_unnormalized,
            ]
        ):
            for i, var in enumerate(outputs):
                fig = plt.figure(figsize=(10, 5))
                plt.plot(
                    range(targets.shape[0]),
                    targets_unnormalized[:, :, :, i].mean(axis=(1, 2)),
                    label="Target",
                )
                min_val, max_val = plt.ylim()
                plt.plot(
                    range(predictions.shape[0]),
                    model_pred_unnormalized[:, :, :, i].mean(axis=(1, 2)),
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
                self.log({f"eval/plots/{var}": wandb.Image(fig)}, step=step)  # type: ignore[attr-defined]
                plt.close()
