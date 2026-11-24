"""Report completed paired-seed arms using saved metrics only."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).parent/'results/comparison'
arms=['full','detach','latent_full','latent_only']
labels=['Full temporal gradients','Detach physical feedback','Latents + full gradients','Latents-only temporal gradients']
data={a:json.loads((root/a/'heldout.json').read_text()) for a in arms}
calls=[1,3,9,36,72]
def vals(a,c,mode='persistent'):
 return np.array([r['mse'] for r in sorted(data[a]['records'],key=lambda r:r['origin']) if r['call']==c and r['mode']==mode])
for a in arms:
 assert len(list((root/a).glob('validation-*.json')))==70
 assert (root/a/'COMPLETE').exists() and (root/a/'EVAL_COMPLETE').exists()
 assert all(len(vals(a,c))==6 and np.isfinite(vals(a,c)).all() for c in calls)
rows=[{'arm':a,'selected_epoch':data[a]['selected_epoch'],'last_lead_day':c*10,'mean_normalized_mse':float(vals(a,c).mean()),'percent_vs_full':float(100*(vals(a,c).mean()/vals('full',c).mean()-1)),'paired_origins_better_than_full':int((vals(a,c)<vals('full',c)).sum())} for a in arms for c in calls]
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for a,label in zip(arms,labels):
 axs[0].plot(np.array(calls)*10,[vals(a,c).mean() for c in calls],'-o',label=label)
 curve=[]
 for p in sorted((root/a).glob('validation-*.json')):
  rec=json.loads(p.read_text());curve.append(np.mean([r['mse'] for r in rec if r['call']==36]))
 axs[1].plot(range(1,71),curve,label=label)
axs[0].set(xlabel='Last day in evaluated two-timestamp block',ylabel='Mean normalized wet-cell MSE',title='Six held-out origins');axs[0].set_xscale('log');axs[0].legend(fontsize=8)
axs[1].set(xlabel='Training epoch',ylabel='Validation normalized MSE',title='360-day validation: checkpoint selection')
axs[1].set_yscale('log')
for ax in axs:ax.grid(alpha=.2)
fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf')
lines=['# Four-arm temporal-memory experiment','', 'All four arms completed 70 epochs and final evaluations on Engaging. One paired random seed (15), fixed normalized wet-cell MSE, 1° OM4, four autoregressive calls with two timestamps per call. EMA checkpoint selection used the lowest 360-day validation MSE; held-out evaluation used six origins and 72 calls.','', '## Held-out results','', 'Each number averages all 77 channels and both timestamps in the indicated block, then six origins. Day 720 denotes the days 715/720 block, not a cumulative loss.','', '| Arm | Selected epoch | Days 5/10 | Days 355/360 | Days 715/720 | Change vs full at 720 |','|---|---:|---:|---:|---:|---:|']
for a,label in zip(arms,labels):
 lines.append(f'| {label} | {data[a]["selected_epoch"]} | {vals(a,1).mean():.6f} | {vals(a,36).mean():.6f} | {vals(a,72).mean():.6f} | {100*(vals(a,72).mean()/vals("full",72).mean()-1):+.2f}% |')
lines += ['', '![Comparison and validation curves](comparison.png)','', '## First-block high-frequency error','', 'Percentage change in normalized high-pass error MSE versus full gradients, averaged across group channels and six origins. The stencil is center minus its four-neighbor mean on all-wet interiors; it is not a sharp spectral band.','', '| Arm | Deep temperature | Deep salinity | SSH |','|---|---:|---:|---:|']
groups=[range(49,57),range(68,76),[76]]
def hf(a,ids):return np.mean([np.mean(np.array(r['roughness_error'])[list(ids)]**2) for r in data[a]['records'] if r['mode']=='persistent' and r['call']==1])
for a,label in zip(arms,labels):lines.append('| '+label+' | '+' | '.join(f'{100*(hf(a,g)/hf("full",g)-1):+.1f}%' for g in groups)+' |')
lines+=['','## Latent reset control','', 'Resetting latent state to zero before every forecast call leaves the first call unchanged, but increases day-720 MSE:','']
for a in ['latent_full','latent_only']:
 lines.append(f'- {a}: {vals(a,72).mean():.6f} → {vals(a,72,"reset_every_call").mean():.6f} ({100*(vals(a,72,"reset_every_call").mean()/vals(a,72).mean()-1):+.2f}%).')
lines += ['', '## Interpretation and limits','', f'Detaching physical feedback beats full gradients at day720 in {(vals("detach",72)<vals("full",72)).sum()}/6 paired origins. Latent-only temporal gradients improve mean day720 MSE by {100*(1-vals("latent_only",72).mean()/vals("detach",72).mean()):.2f}% relative to detachment alone. That small difference needs replication; six correlated ocean origins do not substitute for independent training seeds.', '', 'The results support a harmful contribution from temporal gradients through physical outputs in this setup. They do not prove that the network deliberately stores historical information in texture. The single-example experiment produced residual texture without temporal feedback, so BPTT is not necessary for all such artifacts. The reset control shows functional dependence on carried latent state, but resetting it also creates an out-of-training-distribution intervention and does not establish what information it encodes.','', 'Checkpoints selected at epochs29–31, although every arm completed70epochs; this is the predeclared validation-selected comparison, not final-epoch weights. Latent arms have extra parameters and no matched nonpersistent-parameter control. Batch size/normalization and fixed-MSE training differ from the official dynamic-loss training. No claim of statistical significance or generality to quarter-degree models is made.','', '## Provenance','', 'Original array26067267 and scratch continuation26280150 all completed. Runtime Python3.12/PyTorch2.9.1+cu128, one L40S per arm, FP32. Code snapshot code-v4; scratch root ~/orcd/scratch/om4-temporal-memory-20261016. Per-arm configs,70validation records, heldout metrics, and selected-checkpoint SHA256 values accompany this report. No GPU execution was used to generate this report.','']
for a in arms:lines.append(f'- [{a} W&B](https://wandb.ai/ocean_emulators/default/runs/om4mem-{a}-15-20261016); selected checkpoint SHA256 `{data[a]["checkpoint_sha256"]}`.')
(root/'REPORT.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:17])); print('\n'.join(lines[20:38]))
