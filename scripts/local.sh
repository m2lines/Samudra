#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local save data
# ./.python-greene submitit_hydra.py $comp exp=save_data name="$(date +%F)-save_data_1step"

# local train
# ./.python-greene submitit_hydra.py $comp exp=train name="$(date +%F)-train_1step"

# local eval
./.python-greene submitit_hydra.py $comp exp=eval name="$(date +%F)-eval_test"
