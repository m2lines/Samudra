from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from ocean_emulators.config import StochasticDepthConfig
    from ocean_emulators.models.modules.dropout import ScheduledDepthDropout


class StochasticDepthManager:
    """Centralized manager for applying stochastic depth to models.

    Handles all dropout rate calculations, layer registration, and
    epoch tracking in one place.
    """

    def __init__(self, config: "StochasticDepthConfig"):
        self.config = config  # StochasticDepthConfig
        self.drop_path_modules: list[ScheduledDepthDropout] = []

    def calculate_drop_rate(self, layer_index: int) -> float:
        """Calculate dropout rate for a specific layer.

        Implements stage-wise dropout rate calculation with optional per-stage
        multipliers as described in the papers.
        """
        if self.config.drop_path_rate == 0.0 or self.config.early_dropout_epochs == 0:
            return 0.0

        # Apply per-stage multiplier if specified
        stage_multiplier = 1.0
        if self.config.per_stage_multipliers:
            # Ensure we don't go out of bounds
            multiplier_index = min(
                layer_index, len(self.config.per_stage_multipliers) - 1
            )
            stage_multiplier = self.config.per_stage_multipliers[multiplier_index]

        return self.config.drop_path_rate * stage_multiplier

    def create_drop_path(self, layer_index: int) -> "ScheduledDepthDropout | None":
        """Create a DropPath module for a specific layer.

        Returns None if dropout is disabled for this configuration.
        """
        drop_rate = self.calculate_drop_rate(layer_index)

        if drop_rate == 0.0:
            return None

        # Import at runtime to avoid circular import
        from ocean_emulators.models.modules.dropout import ScheduledDepthDropout

        drop_path = ScheduledDepthDropout(
            drop_prob=drop_rate,
            early_epochs=self.config.early_dropout_epochs,
            schedule=self.config.dropout_schedule,
            linear_decay=self.config.linear_decay_to_zero,
        )

        # Register for epoch tracking
        self.drop_path_modules.append(drop_path)

        return drop_path

    def register_model(self, model: nn.Module) -> None:
        """Register all dropout modules in the model for epoch tracking.

        This scans the model for any EarlyDropPath modules that might not have
        been registered during creation and adds them to the tracking list.
        """
        # Import at runtime to avoid circular import
        from ocean_emulators.models.modules.dropout import ScheduledDepthDropout

        for module in model.modules():
            if (
                isinstance(module, ScheduledDepthDropout)
                and module not in self.drop_path_modules
            ):
                self.drop_path_modules.append(module)

    def update_epoch(self, epoch: int) -> None:
        """Update epoch for all registered dropout modules.

        This is the centralized method for updating epochs, replacing the
        need to traverse the model manually in the training loop.
        """
        for drop_path_module in self.drop_path_modules:
            drop_path_module.set_epoch(epoch)

    def is_enabled(self) -> bool:
        """Check if dropout is enabled in this configuration."""
        return self.config.drop_path_rate > 0.0 and self.config.early_dropout_epochs > 0
