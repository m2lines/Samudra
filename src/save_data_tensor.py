import xarray as xr
import numpy as np
import torch
import os
from pathlib import Path

from constants import REGIONS, INPT_VARS, EXTRA_VARS, OUT_VARS
from utils.data_utils import (
    get_wet_mask,
    get_train_test_ranges,
    gen_data_025_lateral,
    gen_data_in,
    gen_data_out,
    data_CNN_Lateral,
    data_CNN_steps_Lateral,
    get_oceanGPT_data,
)
from utils.dist_utils import set_seed


def main(args):
    # Set seeds
    set_seed(args.rand_seed)

    # Check dirs
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir, exist_ok=True)

    # Getting input, extra input and output
    inputs = INPT_VARS[args.exp_num_in]
    extra_in = EXTRA_VARS[args.exp_num_extra]
    outputs = OUT_VARS[args.exp_num_out]

    str_in = "".join([i + "_" for i in inputs])
    str_ext = "".join([i + "_" for i in extra_in])
    str_out = "".join([i + "_" for i in outputs])

    print("inputs: " + str_in)
    print("extra inputs: " + str_ext)
    print("outputs: " + str_out)

    N_atm = len(extra_in)  # Number of atmosphere variables
    N_in = len(inputs)
    N_extra = (
        N_atm + N_in
    )  # Number of atmosphere variables + Lateral boundary variables
    N_out = len(outputs)

    num_in = int((args.hist + 1) * N_in + N_extra)

    print("Number of inputs: ", num_in)  # 3 (ocean speeds + ocean temp)(t) +
    # 3 (atm wind stresses + atm temp)(t) +
    # 3 (boundary ocean speeds + boundary ocean temp)(t) -> 3 (ocean speeds + ocean temp)(t+1)
    print("Number of outputs: ", N_out)  # 3

    str_video = (
        "steps_"
        + str(args.steps)
        + "_"
        + args.region
        + "_Test_in_"
        + str_in
        + "ext_"
        + str_ext
        + "_out"
        + str_out
        + "N_train_"
        + str(args.N_samples)
        + "_Lateral_Data_025_no_smooth"
    )

    # Getting start and end indices of train and test
    s_train, e_train, e_test = get_train_test_ranges(
        args.N_samples, args.N_val, args.lag, args.hist, args.interval
    )

    # Generate inputs, extra inputs and outputs
    inputs, extra_in, _ = gen_data_025_lateral(
        inputs,
        extra_in,
        outputs,
        args.lag,
        REGIONS[args.region]["lat"],
        REGIONS[args.region]["lon"],
        args.Nb,
    )

    # Generate Wet mask
    wet, _ = get_wet_mask(inputs, "cpu")

    train_input_data, train_extra_data = get_oceanGPT_data(s_train, e_train, args.steps, inputs, extra_in, wet)
    val_input_data, val_extra_data = get_oceanGPT_data(e_train, e_test, args.steps, inputs, extra_in, wet)
    train_data = torch.concat([train_input_data.unsqueeze(0), train_extra_data.unsqueeze(0)])
    val_data = torch.concat([val_input_data.unsqueeze(0), val_extra_data.unsqueeze(0)])

    train_data = train_data.type(torch.FloatTensor)
    val_data = train_data.type(torch.FloatTensor)
    print(train_data.shape)
    print(val_data.shape)


    # Saving datasets
    torch.save(train_data, Path(args.data_dir) / 'train_OceanGPT_data_{0}.pt'.format(str_video))
    torch.save(val_data, Path(args.data_dir) / 'val_OceanGPT_data_{0}.pt'.format(str_video))
