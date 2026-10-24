"""Rewrite location/worker settings; qualify by completing epoch nine before seventy."""
import argparse,copy,shutil
from pathlib import Path
from samudra.config import TrainConfig
import yaml
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
for arm in ('local','spatial','gradient'):
 cfg=yaml.safe_load((a.root/f'configs/full-{arm}.yaml').read_text())
 cfg['experiment']['base_output_dir']=str(a.root/'outputs')
 cfg['experiment']['data_root']='/orcd/data/abodner/002/jrusak/om4_onedeg_v3'
 cfg['data']['loading']['num_workers']=8
 for source in cfg['data']['sources']:
  source['data_means_location']=str(a.root/'normalization/OM4_means.zarr')
  source['data_stds_location']=str(a.root/'normalization/OM4_stds.zarr')
 cfg['scheduler']['target_epochs']=70
 if arm=='local':cfg['loss']['weights_path']=str(a.root/'scales/scales.npz')
 for phase,epochs in [('qualify',9),('full',70)]:
  c=copy.deepcopy(cfg);c['epochs']=epochs;TrainConfig.model_validate(c)
  (a.root/f'configs/engaging-{phase}-{arm}.yaml').write_text(yaml.safe_dump(c,sort_keys=False))
 ck=a.root/'outputs'/cfg['experiment']['name']/'saved_nets'
 assert not (ck/'ckpt.pt').exists(), 'Do not overwrite an existing resume checkpoint'
 shutil.copyfile(ck/'ckpt_8.pt',ck/'ckpt.pt')
print('Three epoch-eight checkpoints prepared; qualification ends at epoch nine with original seventy-epoch scheduler.')
