# SPDX-FileCopyrightText: 2024 Allen Institute for Artificial Intelligence
# SPDX-FileCopyrightText: 2026 Ocean Emulator Authors
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import torch

from samudra.utils.wandb import Metrics


class ValidateSubAggregator(ABC):
    @abstractmethod
    def get_logs(self, label: str) -> Metrics: ...

    @abstractmethod
    def record_batch(
        self,
        *,
        loss: torch.Tensor,
        target_data,
        gen_data,
        input_data,
        target_data_norm,
        gen_data_norm,
        input_data_norm,
    ): ...
