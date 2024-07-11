#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local save data

# ./.python-perlmutter submitit_hydra.py $comp exp=save_3D_data_global name="$(date +%F)-save_3D_data_surface_test" region=global_3D depth_mode=surface

# ./.python-perlmutter submitit_hydra.py $comp exp=save_3D_data_global name="$(date +%F)-save_3D_data" region=global_3D depth_mode=all

# ./.python-perlmutter submitit_hydra.py $comp exp=save_data_global name="$(date +%F)-save_data_global1_7k" region=global_1 N_samples=7000 N_val=300 N_test=5

# ./.python-perlmutter submitit_hydra.py $comp exp=save_data_global name="$(date +%F)-save_data_combined_global2x" region=combined_global_2x
