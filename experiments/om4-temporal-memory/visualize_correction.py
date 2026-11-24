"""Visualize saved diagnostic intervention; no model execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent / 'results/diagnostics'
meta = json.loads((root / 'provenance.json').read_text())
source = meta['source']
a = np.load(root / 'origin-0-maps.npz')
mask = np.load(root / 'mask.npy')
p, t, d = [a[k].reshape(2, 77, 180, 360) for k in ('prediction', 'truth', 'error_texture')]
rows = []
for group in ('temperature_deep', 'salinity_deep', 'ssh'):
    ids = meta['groups'][group]
    for call in (1, 4):
        values = []
        for path in sorted(root.glob('origin-*.json')):
            records = json.loads(path.read_text())['interventions']
            before = next(r for r in records if r['kind'] == 'baseline' and r['call'] == call)
            after = next(r for r in records if r['kind'] == 'remove_error_texture' and r['group'] == group and r['alpha'] == .5 and r['call'] == call)
            b = np.mean(np.square(np.array(before['rmse_normalized'])[ids]))
            c = np.mean(np.square(np.array(after['rmse_normalized'])[ids]))
            values.append(100*(c/b-1))
        rows.append({'group':group, 'call':call, 'mean_group_mse_percent_change':float(np.mean(values)), 'per_origin_percent_change':values})
(root/'group-correction-summary.json').write_text(json.dumps(rows, indent=2)+'\n')
print(json.dumps(rows, indent=2))

# A fixed, representative depth and the first evaluated timestamp, not an extreme cell.
k = 38 + 13  # thetao, 1850 m
scale, mean = source['stds'][k], source['means'][k]
truth = t[0,k]*scale+mean
before = p[0,k]*scale+mean
after = (p[0,k]-.5*d[0,k])*scale+mean
wet = mask[k].astype(bool)
rmse = lambda z: np.sqrt(np.mean((z[wet]-truth[wet])**2))
b, c = rmse(before), rmse(after)
print(f'1850m first timestamp physical RMSE: {b:.9g} -> {c:.9g} degC; MSE delta {100*(c*c/(b*b)-1):.6g}%')
lon = np.asarray(source['longitude']); lat=np.asarray(source['latitude'])
lon=(lon+180)%360-180; order=np.argsort(lon);lon=lon[order]
lo,hi=np.percentile(truth[wet],[1,99]);lim=np.percentile(abs((before-truth)[wet]),99)
fig, axes=plt.subplots(2,3,figsize=(16,7.6),layout='constrained')
fields=[truth,before,after,before-truth,after-truth,before-after]
titles=['OM4 truth','Original forecast', 'Corrected: remove half the error texture','Original error (forecast − truth)','Corrected error','Removed component (original − corrected)']
for i,(ax,z,title) in enumerate(zip(axes.flat,fields,titles)):
    z=np.where(wet,z,np.nan)[:,order]
    im=ax.pcolormesh(lon,lat,z,cmap='viridis' if i<3 else 'RdBu_r',vmin=lo if i<3 else -lim,vmax=hi if i<3 else lim,shading='nearest',rasterized=True)
    ax.set_facecolor('#bbb');ax.set_title(title,fontsize=11);ax.set_xlim(-180,180);ax.set_ylim(-80,80);ax.set_xlabel('Longitude');ax.set_ylabel('Latitude')
    fig.colorbar(im,ax=ax,label='°C',shrink=.8,extend='both')
fig.suptitle('Official Samudra 1° · temperature at 1850 m · 2014-10-20 (first forecast timestamp)\n'
             f'Wet-cell RMSE: {b:.4f} → {c:.4f} °C; MSE change: {100*(c*c/(b*b)-1):.2f}%\n'
             'Identical scales within each row · truth-assisted diagnostic, not a deployable correction',fontsize=13)
fig.savefig(root/'temperature-before-after.png',dpi=180)
fig.savefig(root/'temperature-before-after.pdf')
