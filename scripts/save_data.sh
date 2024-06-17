#!/bin/bash


# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=save_data name="$(date +%F)-1GPU_save_data_8steps"

./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_3hrs exp=save_3D_data_global name="$(date +%F)-save_3D_data_surface_test" region=global_3D depth_mode=surface

# ./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_6hrs exp=save_3D_data_global name="$(date +%F)-save_3D_data_all" region=global_3D depth_mode=all
