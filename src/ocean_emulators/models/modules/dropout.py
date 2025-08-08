import torch
import torch.nn as nn

from ocean_emulators.config import DropoutSchedule


class EarlyDropPath(nn.Module):
    """Epoch-aware Drop Path for early dropout experiments.

    Implements stochastic depth with epoch-based scheduling as described in:

    Early Dropout:
    - "Early Dropout: Dropping layers to reduce underfitting" (2303.01500)
      https://arxiv.org/abs/2303.01500

    Stochastic Depth:
    - "Deep Networks with Stochastic Depth" (1603.09382)
      https://arxiv.org/abs/1603.09382
    - "A ConvNet for the 2020s" (2201.03545)
      https://arxiv.org/abs/2201.03545
    """

    def __init__(
        self,
        drop_prob: float = 0.0,
        early_epochs: int = 0,
        schedule: DropoutSchedule = "early_only",
        linear_decay: bool = True,
    ):
        super().__init__()
        self.base_drop_prob = drop_prob
        self.early_epochs = early_epochs
        self.schedule = schedule
        self.linear_decay = linear_decay
        self.current_epoch = 0  # Will be set by training loop

    def set_epoch(self, epoch: int):
        """Called by training loop to update current epoch."""
        self.current_epoch = epoch

    def get_current_drop_prob(self) -> float:
        """Calculate dropout probability based on current epoch.

        Implements epoch-based scheduling as described in:
        "Early Dropout: Dropping layers to reduce underfitting" (2303.01500)
        """
        if self.base_drop_prob == 0.0 or self.early_epochs == 0:
            return 0.0

        if self.schedule == "early_only":
            if self.current_epoch >= self.early_epochs:
                return 0.0  # No dropout after early period

            if self.linear_decay:
                # Linear decay from base_drop_prob to 0 over early_dropout_epochs
                decay_factor = 1.0 - (self.current_epoch / self.early_epochs)
                return self.base_drop_prob * decay_factor
            else:
                # Constant rate during early period
                return self.base_drop_prob

        elif self.schedule == "late_only":
            if self.current_epoch < self.early_epochs:
                return 0.0  # No dropout during early period
            return self.base_drop_prob

        else:  # constant
            return self.base_drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current_drop_prob = self.get_current_drop_prob()

        if not self.training or current_drop_prob == 0.0:
            return x

        # Standard stochastic depth implementation (1603.09382)
        # During training: randomly drop layers, during inference: scale by survival probability
        keep_prob = 1 - current_drop_prob
        random_tensor = keep_prob + torch.rand(
            (x.shape[0], 1, 1, 1), dtype=x.dtype, device=x.device
        )
        binary_mask = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_mask
