"""Generate matched current-preset training configs, with separate pipeline pilots."""
import argparse
import copy
from pathlib import Path
import yaml
from samudra.config import TrainConfig

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--data-root',required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[2]
base=TrainConfig.from_yaml(repo/'src/samudra/configs/samudra_om4/train.yaml').model_dump(mode='json')
base['experiment'].update(data_root=a.data_root,base_output_dir=str(a.root/'outputs'),rand_seed=15)
base['experiment']['wandb'].update(mode='online',entity='ocean_emulators',project='default')
base.update(save_freq=1,preemptible=True,backend='cuda')
for mode in ['pilot','full']:
 for arm,loss in [('local',{'type':'structure','kind':'local','weights_path':str(a.root/'scales/scales.npz')}),('spatial',{'type':'structure','kind':'curvature','coefficient':100}),('gradient',{'type':'structure','kind':'horizontal','coefficient':.1})]:
  cfg=copy.deepcopy(base);cfg['loss']=loss;cfg['experiment']['name']=f'current-1deg-structure-{arm}-{mode}-seed15'
  cfg['experiment']['wandb']['group']='current-1deg-structure-20261020'
  if mode=='pilot':
   cfg['epochs']=1;cfg['data']['sources'][0]['train_time']['end']='1975-07-01'
  TrainConfig.model_validate(cfg)
  dest=a.root/'configs';dest.mkdir(exist_ok=True,parents=True)
  (dest/f'{mode}-{arm}.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
print('Validated six configs; full runs retain current preset 70 epochs, batch4, four model calls, 360-day validation.')
