import os
import copy
import wandb
import warnings
import time
import datetime
import json
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
from functools import partial
import logging

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset
import numpy as np
from torch.cuda import amp
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt

import sys
sys.path.append('src/')

from constants import INPT_VARS, EXTRA_VARS, OUT_VARS, DEPTH_LEVELS, get_eval_maps
from utils.train_utils import (
    decomposed_mse,
    decomposed_mse_diff_weighted,
    decomposed_mse_cos_weighted,
    decomposed_mse_scaled,
    decomposed_mse_mae,
    SmoothedValue,
    MetricLogger,
    extract_wet,
    extract_surface_wet,
)
from utils.dist_utils import (
    set_seed,
    init_distributed_mode,
    get_world_size,
    get_rank,
    is_main_process,
    all_reduce_mean,
)
from utils.eval_utils import generate_model_rollout, get_corr_rmse
from utils.data_utils import (
    data_CNN_Disk,
    data_CNN_Disk_steps,
)

import xarray as xr
import dask



if __name__ == "__main__":
    print("Hello World")
