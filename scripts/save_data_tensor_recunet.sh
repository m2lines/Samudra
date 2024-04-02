#!/bin/bash


./.python-greene submitit_hydra.py compute/greene=1x1 compute/greene/node=rtx8000_2hrs exp=save_data_tensor_recunet name="$(date +%F)-save_recunet"
