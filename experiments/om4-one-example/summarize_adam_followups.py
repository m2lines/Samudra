"""Verify matched starts/objectives and summarize the two October19 Adam follow-ups."""
import hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path(__file__).resolve().parent
DATA=Path('/tmp/om4-overfit-torch-input')
meta=json.loads((DATA/'metadata.json').read_text())
std=np.array(meta['stds'][:77]);mean=np.array(meta['means'][:77])
wet=np.load(DATA/'mask.npy');truth=np.load(DATA/'prognostic.npy',mmap_mode='r')[2].astype(float)
scales=BASE/'results/local-range-scales/scales.npz'
weights=np.load(scales)['weights'].astype(float)
patch=(slice(50,82),slice(200,232))
channels=[13,32,51,70,76];labels=['U · 1850m','V · 1850m','T · 1850m','S · 1850m','SSH']
groups={'U':slice(0,19),'V':slice(19,38),'T':slice(38,57),'S':slice(57,76),'SSH':slice(76,77)}

def loss(e,w=1):return float(((e*e*w*wet).sum((-2,-1))/wet.sum((-2,-1))).mean())

def main():
 for campaign,nsteps,arms,titles in [
  ('local-range',5000,['mse','spatial_100','local_range'],['Adam + MSE','Adam + spatial loss','Adam + local weighting']),
  ('salinity-head',10000,['multiplier_1','multiplier_0.1','multiplier_10'],['Salinity LR 1x','Salinity LR 0.1x','Salinity LR 10x'])]:
  root=BASE/'results'/campaign;out=BASE/'report'/campaign
  verified=json.loads((out/'verification.json').read_text())['arms']
  protocols={a:json.loads((root/a/'protocol.json').read_text()) for a in arms}
  histories={a:[json.loads(l) for l in (root/a/'history.jsonl').read_text().splitlines()] for a in arms}
  initial=[np.load(root/a/'maps-0.npz')['prediction'] for a in arms]
  assert all(np.array_equal(initial[0],p) for p in initial)
  assert len({p['parameters'] for p in protocols.values()})==1
  if campaign=='local-range':
   assert len({p['initial_parameters_sha256'] for p in protocols.values()})==1
   assert {p['scales_sha256'] for p in protocols.values()}=={hashlib.sha256(scales.read_bytes()).hexdigest()}
  else:
   assert len({p['source_checkpoint_sha256'] for p in protocols.values()})==1
   source=np.load(BASE/'results/absolute-relative-loss/identity_weight_100/maps-5000.npz')['prediction']
   assert np.array_equal(source,initial[0])
  predictions={};summary={}
  for a in arms:
   pred=np.load(root/a/f'maps-{nsteps}.npz')['prediction'][0].astype(float);predictions[a]=pred
   r=verified[a];end=histories[a][-1]
   state=loss(pred-truth);weighted=loss(pred-truth,weights)
   np.testing.assert_allclose(state,end['eval_mse'],rtol=3e-6)
   if campaign=='local-range':
    np.testing.assert_allclose(weighted,end['weighted_mse_eval'],rtol=3e-6)
    objective=weighted if a=='local_range' else state+(100*r['highpass_mse'] if a=='spatial_100' else 0)
    np.testing.assert_allclose(objective,end['training_objective_eval'],rtol=3e-6)
   else:np.testing.assert_allclose(state+100*r['highpass_mse'],end['training_objective'],rtol=3e-6)
   summary[a]={'state_mse':state,'highpass_mse':r['highpass_mse'],'local_weighted_mse':weighted,
               'seconds':end['seconds'],'selected_fields':r['selected_fields'],
               'group_state_mse':{name:float(np.mean(r['per_channel_mse'][sl])) for name,sl in groups.items()},
               'group_highpass_mse':{name:float(np.mean(r['per_channel_highpass_mse'][sl])) for name,sl in groups.items()}}
  baseline=summary[arms[0]]
  for a in arms:
   summary[a]['relative_to_control']={k:summary[a][k]/baseline[k] for k in ['state_mse','highpass_mse','local_weighted_mse']}
   summary[a]['salinity_state_ratio_to_control']=summary[a]['group_state_mse']['S']/baseline['group_state_mse']['S']
  (out/'summary.json').write_text(json.dumps({'verification':'Exact identical initial maps and source hashes; independent CPU float64 objectives reproduce logs; all77 endpoints verified separately','arms':summary},indent=2)+'\n')
  fig,axs=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
  keys=['eval_mse','highpass_mse','weighted_mse_eval' if campaign=='local-range' else 'training_objective']
  for a,title in zip(arms,titles):
   for ax,key in zip(axs,keys):ax.semilogy([r['step'] for r in histories[a]],[r[key] for r in histories[a]],label=title)
  for ax,key in zip(axs,keys):ax.set(xlabel='Adam updates',ylabel=key);ax.grid(alpha=.2)
  axs[0].legend(fontsize=8);fig.savefig(out/'comparison-curves.png',dpi=150);plt.close(fig)
  fig,axs=plt.subplots(5,4,figsize=(13,13),layout='constrained')
  for row,(c,label) in enumerate(zip(channels,labels)):
   values=[x[c][patch] for x in [truth,*predictions.values()]]
   lo=min(x.min() for x in values);hi=max(x.max() for x in values)
   for col,v in enumerate(values):
    im=axs[row,col].imshow(v,origin='lower',interpolation='nearest',vmin=lo,vmax=hi,cmap='viridis')
    fig.colorbar(im,ax=axs[row,col],shrink=.65);axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
   axs[row,0].set_ylabel(label)
  for ax,title in zip(axs[0],['Target',*titles]):ax.set_title(title,fontsize=10)
  fig.suptitle(f'{campaign}: same quiet South Pacific patch; normalized units\nCommon color limits within each row; raw cells, no smoothing')
  fig.savefig(out/'matched-patch.png',dpi=150,bbox_inches='tight');plt.close(fig)
  print(campaign,{a:{k:v for k,v in s.items() if k in ['state_mse','highpass_mse','relative_to_control','salinity_state_ratio_to_control']} for a,s in summary.items()})

if __name__=='__main__':main()
