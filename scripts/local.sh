#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local save data
# ./.python-greene submitit_hydra.py $comp exp=save_data name="$(date +%F)-save_data_gulfext3"

# local save data tensor
# ./.python-greene submitit_hydra.py $comp exp=save_data_tensor_oceangpt name="$(date +%F)-test_save_data_tensor_16step"

./.python-greene submitit_hydra.py $comp exp=save_data_tensor_recunet name="$(date +%F)-save_data_tensor_recunet_Gulfext_with_wet"
