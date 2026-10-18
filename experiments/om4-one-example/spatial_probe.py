"""Describe original residual texture without assuming a causal mechanism."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--maps',required=True);ap.add_argument('--metadata',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True);z=np.load(a.maps);meta=json.loads(Path(a.metadata).read_text())
 p=z['prediction'][0];y=z['truth'][0];wet=z['mask'];e=p-y;inside=wet.copy()
 for axis,shift in [(-2,-1),(-2,1),(-1,-1),(-1,1)]:inside &= np.roll(wet,shift,axis)
 inside[:,0]=False;inside[:,-1]=False
 h=e-sum(np.roll(e,s,a) for a,s in [(-2,-1),(-2,1),(-1,-1),(-1,1)])/4
 selected=[(13,'U at 1850 m'),(32,'V at 1850 m'),(51,'T at 1850 m'),(70,'S at 1850 m'),(76,'SSH')]
 parity={};grid=np.indices(e.shape[-2:])
 for c,name in selected:
  parity[name]={}
  for period in (2,4,8):
   phase=(grid[0]%period)*period+grid[1]%period
   means=np.array([h[c][(phase==k)&inside[c]].mean() for k in range(period**2)])
   template=means[phase]
   parity[name][period]={'template_fraction_highpass_energy':float((template[inside[c]]**2).mean()/(h[c][inside[c]]**2).mean()),'max_phase_bias_normalized':float(abs(means).max())}
 (out/'global-parity.json').write_text(json.dumps(parity,indent=2))
 # Common open-ocean box, no land/bottom cells for any selected channel.
 sy,sx=slice(50,82),slice(200,232);std=np.asarray(meta['stds'])
 assert all(wet[c,sy,sx].all() for c,_ in selected)
 fig,axs=plt.subplots(5,3,figsize=(11,14),layout='constrained')
 for row,(c,name) in enumerate(selected):
  t=y[c,sy,sx]*std[c];pred=p[c,sy,sx]*std[c];err=e[c,sy,sx]*std[c]
  vmin,vmax=min(t.min(),pred.min()),max(t.max(),pred.max());limit=np.max(abs(err))
  for col,arr in enumerate((t,pred,err)):
   im=axs[row,col].imshow(arr,origin='lower',interpolation='nearest',cmap='viridis' if col<2 else 'RdBu_r',vmin=vmin if col<2 else -limit,vmax=vmax if col<2 else limit)
   fig.colorbar(im,ax=axs[row,col],shrink=.7);axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
  axs[row,0].set_ylabel(name)
 for ax,title in zip(axs[0],('Target minus normalization mean','Prediction minus normalization mean','Prediction − target')):ax.set_title(title)
 fig.suptitle('Original 5,000-update fit: fully wet South Pacific 32×32-cell patch\nPhysical units; independent scales by row; target/prediction share scales')
 fig.savefig(out/'original-patch.png',dpi=170);plt.close(fig)
 # Cosines of masked high-pass maps. This captures shared spatial error, not
 # causation or exact correlations on pairwise common wet domains.
 flat=(h*inside).reshape(77,-1).astype(np.float64);norm=np.linalg.norm(flat,axis=1)
 cos=(flat@flat.T)/np.maximum(norm[:,None]*norm[None,:],1e-30)
 np.save(out/'highpass-map-cosines.npy',cos)
 fig,ax=plt.subplots(figsize=(8,7),layout='constrained');im=ax.imshow(cos,vmin=-1,vmax=1,cmap='RdBu_r');fig.colorbar(im,ax=ax)
 ticks=[9,28,47,66,76];labels=['U','V','T','S','SSH'];ax.set_xticks(ticks,labels);ax.set_yticks(ticks,labels)
 for edge in [18.5,37.5,56.5,75.5]:ax.axhline(edge,c='k',lw=.4);ax.axvline(edge,c='k',lw=.4)
 ax.set_title('Cosine similarity of residual high-pass maps\nDry/edge cells zeroed; includes shared-domain coverage effects')
 fig.savefig(out/'highpass-map-cosines.png',dpi=170);plt.close(fig)
 ratios={}
 target_hp=y-sum(np.roll(y,s,a) for a,s in [(-2,-1),(-2,1),(-1,-1),(-1,1)])/4
 for c,name in selected:
  ratios[name]={'global_error_to_truth_hp_rms':float(np.sqrt((h[c][inside[c]]**2).mean()/(target_hp[c][inside[c]]**2).mean())),
                'pacific_patch_error_to_truth_hp_rms':float(np.sqrt((h[c,sy,sx]**2).mean()/(target_hp[c,sy,sx]**2).mean()))}
 (out/'texture-relative-to-target.json').write_text(json.dumps(ratios,indent=2))
 (out/'spatial-protocol.json').write_text(json.dumps(vars(a)|{'patch_rows':[50,82],'patch_columns':[200,232],
     'latitude_bounds':np.asarray(meta['latitude'])[[50,81]].tolist(),'longitude_bounds':np.asarray(meta['longitude'])[[200,231]].tolist(),
     'highpass':'center minus four nearest neighbors, periodic longitude; metrics exclude dry cells and mask edges'},indent=2))
 print('Wrote spatial probes to',out)
if __name__=='__main__':main()
