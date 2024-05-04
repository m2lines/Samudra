#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local save data
# ./.python-greene submitit_hydra.py $comp exp=save_data_global name="$(date +%F)-save_data_global1"

./.python-greene submitit_hydra.py $comp exp=save_data_global name="$(date +%F)-save_data_global2x" region=global_2x
