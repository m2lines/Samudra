import logging

import numpy as np
import torch
from torchinfo import summary

from utils.device import get_device


def get_model_summary(model: torch.nn.Module, num_input_channels: int):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info(f"Number of parameters: {params}")
    summary(model)

    # Model summary with proper device tensors
    input_tensor = torch.zeros(1, num_input_channels, 180, 360, device=get_device())
    logging.info(
        summary(
            model,
            input_data=[[input_tensor] * 2],
            col_names=["kernel_size", "output_size", "num_params"],
            depth=10,
        )
    )

    input_tensor = torch.zeros(1, num_input_channels, 180, 360, device=get_device())
    logging.info(
        summary(model, input_data=[[input_tensor] * 8], col_names=[], depth=10)
    )
