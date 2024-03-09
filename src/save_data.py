import xarray as xr
import numpy as np
import torch
import os
from pathlib import Path

from constants import REGIONS, INPT_VARS, EXTRA_VARS, OUT_VARS
from utils.subgrid_utils import coarse_grid
from utils.data_utils import gen_data_025_lateral, gen_data_in, gen_data_out, data_CNN_Lateral, data_CNN_steps_Lateral
from utils.dist_utils import set_seed

def get_train_test_ranges(N_samples, N_val, lag, hist, interval):
    s_train = lag*hist # 1*0=0
    e_train = s_train + N_samples*interval # 0 + 4000*1 = 4000
    e_test = e_train + interval*N_val # 4000 + 1*300 = 4300
    return s_train, e_train, e_test


def get_wet_mask(inputs):
    wet = xr.zeros_like(inputs[0][0])
    # inputs[0][0,12,12] = np.nan
    for data in inputs:
        wet +=np.isnan(data[0])
    wet = np.isnan(xr.where(wet==0,np.nan,0))
    wet = np.nan_to_num(wet.to_numpy())
    wet = torch.from_numpy(wet).type(torch.float32).to(device="cpu") # change to device

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

    N_atm = len(extra_in) # Number of atmosphere variables
    N_in = len(inputs) 
    N_extra = N_atm + N_in # Number of atmosphere variables + Lateral boundary variables
    N_out = len(outputs)

    num_in = int((args.hist+1)*N_in + N_extra)

    print("Number of inputs: ", num_in) # 3 (ocean speeds + ocean temp)(t) +
                                        # 3 (atm wind stresses + atm temp)(t) +
                                        # 3 (boundary ocean speeds + boundary ocean temp)(t) -> 3 (ocean speeds + ocean temp)(t+1)
    print("Number of outputs: ", N_out) # 3

    str_video = 'steps_'+str(args.steps)+'_'+args.region+'_Test_in_' + str_in + 'ext_' + str_ext +'_out'+ str_out + 'N_train_' + str(args.N_samples) + "_Lateral_Data_025_no_smooth"
    
    # Getting start and end indices of train and test
    s_train, e_train, e_test = get_train_test_ranges(args.N_samples, args.N_val, args.lag, args.hist, args.interval)

    # Generate inputs, extra inputs and outputs
    inputs, extra_in, outputs = gen_data_025_lateral(inputs,extra_in,outputs,args.lag,REGIONS[args.region]["lat"], REGIONS[args.region]["lon"],args.Nb)

    # Generate Wet mask
    wet = get_wet_mask(inputs)
    
    # Generating Validation dataset
    data_in_val = gen_data_in(0,e_train,e_test,args.interval,args.lag,args.hist,inputs,extra_in)
    data_out_val = gen_data_out(0,e_train,e_test,args.lag,args.interval,outputs)
    val_data = data_CNN_Lateral(data_in_val,data_out_val,wet,N_atm,args.Nb,args.device)

    # Generating Training dataset
    data_in_train = []
    data_out_train = []
    for i in range(args.steps):
        offset = 0*args.interval
        data_in_train.append(gen_data_in(i,s_train+offset,e_train,
                                        args.interval,args.lag, args.hist,inputs,extra_in))
        data_out_train.append(gen_data_out(i,s_train+offset,e_train,
                                        args.lag,args.interval, outputs))
    
    train_data = data_CNN_steps_Lateral(data_in_train,data_out_train,
                                        args.steps,wet,N_atm,args.Nb,device=args.device)  

    # Saving datasets
    torch.save(train_data, Path(args.data_dir) / 'train_data_{0}.pt'.format(str_video))
    torch.save(val_data, Path(args.data_dir) / 'val_data_{0}.pt'.format(str_video))