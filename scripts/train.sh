#!/bin/bash


./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train name="$(date +%F)-1GPU_train_CosineAnnealWarmLR_T10_1e-3"
