#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local save data
# ./.python-greene submitit_hydra.py $comp exp=save_data name="$(date +%F)-save_data_test"

# local save data tensor
./.python-greene submitit_hydra.py $comp exp=save_data_tensor name="$(date +%F)-1GPU_save_data_tensor_16step"

# local train
# ./.python-greene submitit_hydra.py $comp exp=train name="$(date +%F)-train_1step"

# local eval
# ./.python-greene submitit_hydra.py $comp exp=eval name="$(date +%F)-eval_test"
