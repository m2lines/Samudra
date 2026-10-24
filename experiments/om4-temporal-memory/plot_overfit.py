import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).parent/'results/overfit-one'
m=json.loads((r/'protocol.json').read_text()); source=m['source']
h=[json.loads(x) for x in (r/'history.jsonl').read_text().splitlines()]
a=np.load(r/'maps-5000.npz');p=a['prediction'][0];t=a['truth'][0];wet=a['mask'].astype(bool)
std=np.asarray(source['stds'][:77]);mean=np.asarray(source['means'][:77]);lon=np.asarray(source['longitude']);lat=np.asarray(source['latitude']);lon=(lon+180)%360-180;order=np.argsort(lon);lon=lon[order]
fig,axs=plt.subplots(3,3,figsize=(14,10),layout='constrained'); rows=[]
for row,(k,name,unit) in enumerate([(51,'Temperature, 1850 m','°C'),(70,'Salinity, 1850 m','psu'),(76,'Sea surface height','m')]):
    tr=t[k]*std[k]+mean[k];pr=p[k]*std[k]+mean[k];er=pr-tr
    rmse=np.sqrt(np.mean(er[wet[k]]**2)); rows.append({'variable':name,'rmse':float(rmse),'unit':unit})
    region=wet[k] & (np.asarray(source['longitude'])[None,:]>280) & (np.asarray(source['longitude'])[None,:]<350) & (lat[:,None]>-10)&(lat[:,None]<60)
    low,high=np.percentile(tr[region],[1,99]);limit=np.percentile(abs(er[region]),99)
    for col,(z,title) in enumerate([(tr,'Truth'),(pr,'After 5,000 updates'),(er,'Residual error')]):
        ax=axs[row,col];z=np.where(wet[k],z,np.nan)[:,order]
        im=ax.pcolormesh(lon,lat,z,shading='nearest',cmap='viridis' if col<2 else 'RdBu_r',vmin=low if col<2 else -limit,vmax=high if col<2 else limit,rasterized=True)
        ax.set_xlim(-80,-10);ax.set_ylim(-10,60);ax.set_facecolor('#bbb');ax.set_title(f'{name}: {title}',fontsize=10)
        fig.colorbar(im,ax=ax,label=unit,shrink=.8,extend='both')
fig.suptitle('Single-example fit · first target 2014-10-20 · Atlantic detail\nTruth and prediction share scales; residuals have their own expanded scales',fontsize=13)
fig.savefig(r/'final-maps.png',dpi=160);fig.savefig(r/'final-maps.pdf')
fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
for key,label in [('train_mse','Train-mode MSE'),('eval_mse','Eval-mode MSE'),('highpass_error_mse','High-pass error MSE')]: ax.semilogy([x['step'] for x in h],[x[key] for x in h],label=label)
ax.set_xlabel('Optimizer updates');ax.set_ylabel('Normalized MSE');ax.legend();ax.grid(alpha=.25);ax.set_title('One fixed target · random initialization · no temporal feedback')
fig.savefig(r/'learning-curve.png',dpi=180)
(r/'physical-summary.json').write_text(json.dumps(rows,indent=2))
print(rows)
