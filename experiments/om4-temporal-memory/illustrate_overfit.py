"""CPU-only illustration of exact saved inputs, supervised target and output."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent/'results/overfit-one'
data=Path('/tmp/om4-overfit-torch-input')
m=json.loads((data/'metadata.json').read_text())
p=np.load(data/'prognostic.npy',mmap_mode='r');b=np.load(data/'boundary.npy',mmap_mode='r')
a=np.load(root/'maps-5000.npz'); pred=a['prediction'][0];mask=a['mask'].astype(bool)
assert np.array_equal(p[2],a['truth'][0]), 'Input subset does not match saved target'
lat=np.array(m['latitude']);lon=(np.array(m['longitude'])+180)%360-180;order=np.argsort(lon);lon=lon[order]
region=(lat[:,None]>=-10)&(lat[:,None]<=60)&((np.array(m['longitude'])[None,:]+180)%360-180>=-80)&((np.array(m['longitude'])[None,:]+180)%360-180<=-10)
means=np.array(m['means']);stds=np.array(m['stds'])
def paint(ax,z,wet,lo,hi,cmap):
 im=ax.pcolormesh(lon,lat,np.where(wet,z,np.nan)[:,order],vmin=lo,vmax=hi,cmap=cmap,shading='nearest',rasterized=True)
 ax.set_xlim(-80,-10);ax.set_ylim(-10,60);ax.set_facecolor('#bbb');ax.tick_params(labelsize=7)
 return im
for filename, rows in [('velocity-input-output',[(0,'U, 2.5 m','m/s'),(19,'V, 2.5 m','m/s'),(13,'U, 1850 m','m/s'),(32,'V, 1850 m','m/s')]),('scalar-input-output',[(38,'Temperature, 2.5 m','°C'),(47,'Temperature, 550 m','°C'),(54,'Temperature, 4000 m','°C'),(57,'Salinity, 2.5 m','psu'),(73,'Salinity, 4000 m','psu')])]:
 fig,axs=plt.subplots(len(rows),5,figsize=(17,2.5*len(rows)),layout='constrained')
 for row,(k,label,unit) in enumerate(rows):
  values=[p[0,k]*stds[k]+means[k],p[1,k]*stds[k]+means[k],p[2,k]*stds[k]+means[k],pred[k]*stds[k]+means[k],(pred[k]-p[2,k])*stds[k]]
  valid=mask[k]&region
  lo,hi=np.percentile(np.concatenate([z[valid] for z in values[:4]]),[1,99])
  if k<38:hi=max(abs(lo),abs(hi));lo=-hi
  lim=np.percentile(abs(values[-1][valid]),99)
  for col,z in enumerate(values):
   im=paint(axs[row,col],z,mask[k],lo if col<4 else -lim,hi if col<4 else lim,'RdBu_r' if k<38 or col==4 else 'viridis')
   if col in (3,4):fig.colorbar(im,ax=axs[row,col],label=unit,shrink=.8)
   if row==0:axs[row,col].set_title(['Input Oct 10','Input Oct 15','Target Oct 20','Fitted Oct 20','Residual (expanded scale)'][col],fontsize=10)
  axs[row,0].set_ylabel(label)
 fig.suptitle('Single-example fit: two inputs → one supervised target · Atlantic detail\nFirst four columns share each row’s scale; gray = dry or below-bottom cells',fontsize=12)
 fig.savefig(root/(filename+'.png'),dpi=150);fig.savefig(root/(filename+'.pdf'))
fig,axs=plt.subplots(4,2,figsize=(9,10),layout='constrained')
for i,(name,unit) in enumerate([('Zonal wind stress','N/m²'),('Meridional wind stress','N/m²'),('Surface heat flux','W/m²'),('Heat-flux anomaly','W/m²')]):
 vals=b[:2,i]*stds[77+i]+means[77+i];wet=mask[0];lo,hi=np.percentile(vals[:,wet&region],[1,99]);hi=max(abs(lo),abs(hi))
 for j in range(2):
  im=paint(axs[i,j],vals[j],wet,-hi,hi,'RdBu_r');fig.colorbar(im,ax=axs[i,j],label=unit,shrink=.8);axs[i,j].set_title(f'{name} · Oct {10+5*j}',fontsize=10)
fig.suptitle('Additional model inputs: four forcing channels at both input timestamps')
fig.savefig(root/'forcing-inputs.png',dpi=150)
print('Exact target match verified; generated velocity, scalar, and forcing illustrations.')
