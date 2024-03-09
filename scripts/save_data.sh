#!/bin/bash


./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=save_data name="$(date +%F)-1GPU_save_data_8steps"
