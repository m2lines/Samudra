"""Reproducible common-scale plots and tables from single-example sweep artifacts."""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import DEPTHS, PROG


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--data',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((Path(a.data)/'metadata.json').read_text());std=np.asarray(meta['stds'][:77]);mean=np.asarray(meta['means'][:77])
    histories={}
    for path in sorted(root.glob('*/history.jsonl')):
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        # Resumes can duplicate measurement steps. Keep latest measurement.
        histories[path.parent.name]=list({r['step']:r for r in rows}.values())
    if not histories:raise RuntimeError('No histories')
    summary={name:{'complete':(root/name/'COMPLETE').exists(),'step':h[-1]['step'],'eval_mse':h[-1]['eval_mse'],
                   'highpass_mse':h[-1]['highpass_mse'],'best_recorded_mse':min(r['eval_mse'] for r in h),
                   'rmse_physical':(np.sqrt(h[-1]['per_channel_mse'])*std).tolist()} for name,h in histories.items()}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axs=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for name,h in histories.items():
        for ax,key in zip(axs,('eval_mse','highpass_mse')):
            ax.semilogy([r['step'] for r in h],[r[key] for r in h],label=name)
    axs[0].set_ylabel('Normalized wet-cell MSE');axs[1].set_ylabel('Interior high-pass error MSE')
    for ax in axs:ax.set_xlabel('Additional optimizer updates (LBFGS: outer iterations)');ax.grid(alpha=.2)
    axs[1].legend(fontsize=8);fig.savefig(out/'curves.png',dpi=170);plt.close(fig)
    completed=[n for n in histories if summary[n]['complete']]
    if not completed:return
    chosen=min(completed,key=lambda n:summary[n]['eval_mse'])
    files=list((root/chosen).glob('maps-*.npz'));final=max(files,key=lambda p:int(p.stem.split('-')[-1]))
    z=np.load(final);pred=z['prediction'][0];truth=z['truth'][0];wet=z['mask'];initial=np.load(root/chosen/'maps-0.npz')['prediction'][0]
    # For scratch architecture screens compare against the trained baseline.
    reference='checkpoint before continuation'
    if (root/'baseline'/'COMPLETE').exists():
        baseline=max((root/'baseline').glob('maps-*.npz'),key=lambda p:int(p.stem.split('-')[-1]))
        initial=np.load(baseline)['prediction'][0];reference='trained tiny baseline'
    channels=[38,51,57,70,76,13]
    fig,axs=plt.subplots(len(channels),3,figsize=(12,15),layout='constrained')
    for row,c in enumerate(channels):
        mask=wet[c];t=np.where(mask,truth[c]*std[c]+mean[c],np.nan)
        e0=np.where(mask,(initial[c]-truth[c])*std[c],np.nan);e1=np.where(mask,(pred[c]-truth[c])*std[c],np.nan)
        limit=max(np.nanpercentile(np.abs(e0),99),np.nanpercentile(np.abs(e1),99),1e-12)
        # Display source grid without interpolation; common residual color limits.
        for col,values in enumerate((t,e0,e1)):
            im=axs[row,col].imshow(values,origin='lower',interpolation='nearest',aspect='auto',
                cmap='viridis' if col==0 else 'RdBu_r',**({} if col==0 else {'vmin':-limit,'vmax':limit}))
            fig.colorbar(im,ax=axs[row,col],shrink=.75)
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
        axs[row,0].set_ylabel(PROG[c])
    axs[0,0].set_title('Target (physical units)');axs[0,1].set_title(reference+' error');axs[0,2].set_title(chosen+' error')
    fig.suptitle('Residuals: prediction − target; same scale within each row\nBest completed arm selected by overall MSE; not an independent validation')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    # Zoomed, fully wet patch makes small texture changes legible.
    fig,axs=plt.subplots(5,5,figsize=(17,14),layout='constrained')
    sy,sx=slice(50,82),slice(200,232)
    for row,c in enumerate([13,32,51,70,76]):
        assert wet[c,sy,sx].all()
        t=truth[c,sy,sx]*std[c]+mean[c]
        before=initial[c,sy,sx]*std[c]+mean[c]
        after=pred[c,sy,sx]*std[c]+mean[c]
        e0=before-t;e1=after-t
        limit=max(np.max(abs(e0)),np.max(abs(e1)),1e-12)
        vmin=min(t.min(),before.min(),after.min());vmax=max(t.max(),before.max(),after.max())
        for col,values in enumerate((t,before,after,e0,e1)):
            im=axs[row,col].imshow(values,origin='lower',interpolation='nearest',cmap='viridis' if col<3 else 'RdBu_r',
                                  vmin=vmin if col<3 else -limit,vmax=vmax if col<3 else limit)
            fig.colorbar(im,ax=axs[row,col],shrink=.65)
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
        axs[row,0].set_ylabel(PROG[c])
    for ax,title in zip(axs[0],('Target','Reference prediction',chosen+' prediction','Reference error',chosen+' error')):ax.set_title(title)
    fig.suptitle('Fully wet South Pacific patch: common prediction and error scales within each row\nReference: '+reference+'; selected arm is best completed overall MSE')
    fig.savefig(out/'patch-before-after.png',dpi=170);plt.close(fig)
    fig,axs=plt.subplots(1,4,figsize=(13,5),layout='constrained')
    for name in completed:
        for k,ax in enumerate(axs):ax.semilogx(summary[name]['rmse_physical'][k*19:(k+1)*19],DEPTHS,label=name)
    for ax,title in zip(axs,('U (m/s)','V (m/s)','Temperature (°C)','Salinity (psu)')):
        ax.invert_yaxis();ax.set_xlabel('RMSE, '+title);ax.grid(alpha=.2)
    axs[0].set_ylabel('Depth (m)');axs[-1].legend(fontsize=7);fig.savefig(out/'depth-rmse.png',dpi=170);plt.close(fig)
    print(json.dumps({'selected':chosen,'summaries':{n:{k:v for k,v in s.items() if k!='rmse_physical'} for n,s in summary.items()}},indent=2))
if __name__=='__main__':main()
