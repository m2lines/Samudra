#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local train
# ./.python-greene submitit_hydra.py $comp exp=train_vit_gulfext3 name="$(date +%F)-train_vit_gulfext3"

# ./.python-greene submitit_hydra.py $comp exp=train_recunet name="$(date +%F)-recunet_119"

# ./.python-greene submitit_hydra.py $comp exp=train_norecunet_on_vit name="$(date +%F)-test_train_norecunet_gulfext3"

# ./.python-greene submitit_hydra.py $comp exp=train_norecunet_on_vit_119 name="$(date +%F)-test_train_norecunet_gulfext"
