#!/bin/bash

# Obvious / static arguments
comp="compute=local"

# EXPERIMENT LAUNCHES
# GO BOTTOM TO TOP

# local eval
# ./.python-greene submitit_hydra.py $comp exp=eval name="$(date +%F)-eval_test"

# ./.python-greene submitit_hydra.py $comp exp=eval_recunet name="$(date +%F)-eval_recunet_119"

./.python-greene submitit_hydra.py $comp exp=eval_norecunet name="$(date +%F)-eval_norecunet_on_vit_ext3"
