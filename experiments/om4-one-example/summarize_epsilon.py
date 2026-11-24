"""Summarize completed Adam epsilon control from verified metrics and histories."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.root/'eps_1e-8/maps-0.npz') as a, np.load(args.root/'eps_1e-12/maps-0.npz') as b:
        assert set(a.files)==set(b.files)
        assert all(np.array_equal(a[k],b[k]) for k in a.files)
    v=json.loads((args.report/'verification.json').read_text())['arms']
    a,b=v['eps_1e-8'],v['eps_1e-12']
    summary={'state_mse_reduction_fraction':1-b['state_mse']/a['state_mse'],
             'highpass_mse_reduction_fraction':1-b['highpass_mse']/a['highpass_mse']}
    for metric in ('per_channel_mse','per_channel_highpass_mse'):
        ratio=np.asarray(b[metric])/np.asarray(a[metric])
        summary[metric+'_ratio_min_max']=[float(ratio.min()),float(ratio.max())]
    summary['endpoints']={}
    for arm in ('eps_1e-8','eps_1e-12'):
        assert (args.root/arm/'COMPLETE').exists()
        history=[json.loads(line) for line in (args.root/arm/'history.jsonl').read_text().splitlines()]
        assert history[-1]['step']==10000
        summary['endpoints'][arm]={k:v for k,v in history[-1].items() if k!='per_channel_mse'}
    (args.report/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({k:v for k,v in summary.items() if k!='endpoints'})


if __name__=='__main__':
    main()
