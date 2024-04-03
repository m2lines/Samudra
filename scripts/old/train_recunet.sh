#!/bin/bash


./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_6hrs exp=train_recunet name="$(date +%F)-2GPU_train_recunet_"
