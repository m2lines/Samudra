#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local save data
# ./.python-greene submitit_hydra.py $comp exp=save_data name="$(date +%F)-save_data_test"

# local train
./.python-greene submitit_hydra.py $comp exp=train name="$(date +%F)-train_test"
