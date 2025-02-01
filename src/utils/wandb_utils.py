import wandb
import logging
import matplotlib.pyplot as plt
import torch
from typing import Optional, Dict, Any

class WandBLogger:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance 
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(WandBLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self._enabled = False
        self._initialized = False
        self.run = None
    
    @property
    def enabled(self):
        return self._enabled
        
    def finish(self):
        """Finish the wandb run"""
        if self._enabled:
            wandb.finish()

    def configure(self, enabled: bool, is_main_process: bool):
        """Configure whether wandb should be enabled"""
        assert self._enabled is False, "WandB is already initialized"
        self._enabled = enabled and is_main_process

    def init(self, **kwargs):
        """Initialize wandb run"""
        if self._enabled and not self._initialized:
            try:
                self.run = wandb.init(**kwargs)
                self._initialized = True
            except Exception as e:
                logging.error(f"Failed to initialize wandb: {e}")
                self._enabled = False

    def watch(self, model, **kwargs):
        """Watch model parameters and gradients"""
        if self._enabled:
            wandb.watch(model, **kwargs)

    def log(self, metrics: Dict[str, Any], **kwargs):
        """Log metrics to wandb"""
        if self._enabled:
            wandb.log(metrics, **kwargs)

    def log_training_metrics(self, epoch: int, loss_value: float, lr: float, 
                            loss_per_channel: torch.Tensor, outputs: list,
                            depth_indices: Dict = None, var_indices: Dict = None,
                            depth_set: set = None, var_set: set = None):
        """Log training metrics including depth and variable specific metrics
        
        Args:
            epoch: Current training epoch
            loss_value: Overall loss value
            lr: Current learning rate
            loss_per_channel: Loss values per output channel
            outputs: List of output variable names
            depth_indices: Mapping of depth levels to channel indices
            var_indices: Mapping of variables to channel indices
            depth_set: Set of depth levels
            var_set: Set of variables
        """
        if not self._enabled:
            return

        self.log({
            "train/epoch": epoch,
            "train/total_train_loss_per_batch": loss_value,
            "train/lr_per_batch": lr,
        })

        # Loss per channel
        for i, var in enumerate(outputs):
            self.log({"train/per_channel/" + var: loss_per_channel[i]})

        # Loss per depth
        if all(x is not None for x in [depth_indices, depth_set]):
            for d in depth_set:
                self.log({
                    "train/depth/depth_" + str(d) + "_loss": 
                    torch.mean(loss_per_channel[depth_indices[d]]).item()
                })

        # Loss per input variable
        if all(x is not None for x in [var_indices, var_set]):
            for k in var_set:
                self.log({
                    "train/per_var/" + k + "_loss":
                    torch.mean(loss_per_channel[var_indices[k]]).item()
                })

    def log_validation_metrics(self, 
                             loss_value: float, 
                             loss_per_channel: torch.Tensor,
                             outputs: list,
                             predictions=None,
                             targets=None,
                             targets_unnormalized=None,
                             model_pred_unnormalized=None,
                             depth_indices: Dict = None,
                             var_indices: Dict = None,
                             depth_set: set = None,
                             var_set: set = None):
        """Log validation metrics including depth and variable specific metrics"""
        if not self._enabled:
            return

        self.log({"eval/total_eval_loss_per_batch": loss_value})

        # Loss per channel
        for i, var in enumerate(outputs):
            self.log({"eval/per_channel/" + var: loss_per_channel[i].item()})

        # Loss per depth
        if all(x is not None for x in [depth_indices, depth_set]):
            for d in depth_set:
                self.log({
                    "eval/depth/depth_" + str(d) + "_loss": 
                    torch.mean(loss_per_channel[depth_indices[d]]).item()
                })

        # Loss per input variable
        if all(x is not None for x in [var_indices, var_set]):
            for k in var_set:
                if k == "zos":  # Skip zos variable
                    continue
                self.log({
                    "eval/per_var/" + k + "_loss":
                    torch.mean(loss_per_channel[var_indices[k]]).item()
                })

        # Plot predictions vs targets
        if all(x is not None for x in [predictions, targets, targets_unnormalized, model_pred_unnormalized]):
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
                self.log({f"eval/plots/{var}": wandb.Image(fig)})
                plt.close()
    
    def log_inference_metrics(self, 
                             loss_value: float, 
                             loss_per_channel: torch.Tensor,
                             outputs: list,
                             predictions=None,
                             targets=None,
                             targets_unnormalized=None,
                             model_pred_unnormalized=None,
                             depth_indices: Dict = None,
                             var_indices: Dict = None,
                             depth_set: set = None,
                             var_set: set = None):
        pass

    def setup_run(self, checkpoint_path: str, cfg: Any, finetune: bool = False):
        """Set up a wandb run, either resuming from checkpoint or creating new
        
        Args:
            checkpoint_path: Path to checkpoint file, if resuming
            cfg: Configuration object
            finetune: Whether this is a finetuning run
        
        Returns:
            tuple: (wandb_id, wandb_name) from checkpoint if resuming, else (None, generated_name)
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
            except:
                # If resume fails, start new run
                self.init(
                    config=cfg.__dict__,
                    name=wandb_name,
                    dir=cfg.output_dir,
                    **cfg.wandb.__dict__,
                )
        
        return wandb_id, wandb_name

    def _init_new_run(self, cfg: Any):
        """Initialize a new wandb run
        
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
            wandb_id = self.run.id
        else:
            wandb_id = None
        
        return wandb_id, wandb_name