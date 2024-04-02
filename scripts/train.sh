#!/bin/bash


# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_3hrs exp=train name="$(date +%F)-1GPU_train_CosineAnnealWarmLR_1step"

./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_12hrs exp=train_norecunet_on_vit_119 name="$(date +%F)-2GPU_train_norecunet_gulfext"

# ./.python-greene submitit_hydra.py compute/greene=1x2 compute/greene/node=rtx8000_12hrs exp=train_norecunet_on_vit name="$(date +%F)-2GPU_train_norecunet_gulfext3"
