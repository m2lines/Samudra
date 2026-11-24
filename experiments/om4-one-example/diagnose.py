"""Frozen-statistics gradients and a frozen-backbone linear-head fitting control."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel, Prepared, mse, highpass, interior_mask


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--head-steps',type=int,default=100)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists() or time.time()>=1792414800:return
    torch.set_num_threads(4);torch.manual_seed(15)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Prepared(a.data);x,bs,ys=data.window(0,1,'cuda');b=bs[0];y=ys[0][:,:77]
    wet=torch.tensor(data.mask[None],device='cuda');inside=interior_mask(wet)
    model=OceanModel(wet[0]).cuda().eval()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model']);del checkpoint
    params=list(model.named_parameters());gradients={};losses={};norms={}
    groups={'u':slice(0,19),'v':slice(19,38),'temperature':slice(38,57),'salinity':slice(57,76),'ssh':slice(76,77)}
    objectives=['total','fine','coarse',*groups]
    for name in objectives:
        model.zero_grad(set_to_none=True)
        p=model(x,b)[0][:,:77];e=p-y
        if name=='total':loss=mse(p,y,wet)
        elif name=='fine':loss=mse(highpass(e,inside),torch.zeros_like(e),inside)
        elif name=='coarse':loss=mse(e-highpass(e,inside),torch.zeros_like(e),inside)
        else:
            s=groups[name];loss=mse(p[:,s],y[:,s],wet[:,s])
        loss.backward();losses[name]=float(loss.detach())
        norms[name]={k:float(v.grad.norm()) for k,v in params}
        gradients[name]=torch.cat([v.grad.detach().flatten().cpu() for _,v in params])
        print(name,losses[name],flush=True)
        del p,e,loss
    def cosine(u,v):return float(torch.dot(u,v)/(u.norm()*v.norm()).clamp_min(1e-30))
    cosines={n:{m:cosine(g,h) for m,h in gradients.items()} for n,g in gradients.items()}
    # Finite actual-loss tests of gradient directions, with no BN state change.
    original=[p.detach().clone() for _,p in params]
    weight_norm=sum(p.square().sum() for p in original).sqrt()
    trials=[]
    for direction in ('total','fine','coarse'):
        g=gradients[direction];gnorm=float(g.norm())
        for relative in (0.,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4):
            offset=0
            with torch.no_grad():
                for (_,p),base in zip(params,original):
                    chunk=g[offset:offset+p.numel()].reshape(p.shape).to('cuda');offset+=p.numel()
                    p.copy_(base-relative*weight_norm/max(gnorm,1e-30)*chunk)
                prediction=model(x,b)[0][:,:77];err=prediction-y
                trials.append({'direction':direction,'relative_parameter_step':relative,'mse':float(mse(prediction,y,wet)),
                               'highpass_mse':float(mse(highpass(err,inside),torch.zeros_like(err),inside))})
        with torch.no_grad():
            for (_,p),base in zip(params,original):p.copy_(base)
    del original,gradients
    (out/'gradients.json').write_text(json.dumps({'protocol':vars(a),'losses':losses,'cosines':cosines,'per_parameter_gradient_norm':norms,'finite_step_trials':trials,
        'notes':'Fine uses center-minus-four-neighbors curvature. Coarse=e-fine, both restricted to wet interiors; these are not an orthogonal partition of MSE. Variable losses average their own channel group. All diagnostics use frozen BatchNorm.'},indent=2))
    model.zero_grad(set_to_none=True)
    # A convex quadratic problem in the output head with fixed learned features.
    with torch.no_grad():
        features=model._features(torch.cat((x,b),dim=1),None)[0].detach()
        padded=F.pad(F.pad(features,(1,1,0,0),mode='circular'),(0,0,1,1))
        original_prediction=model(x,b)[0][:,:77].detach()
    weight=nn.Parameter(model.decoder.weight[:77].detach().clone());bias=nn.Parameter(model.decoder.bias[:77].detach().clone())
    del model,features,x,b,bs,ys
    torch.cuda.empty_cache()
    opt=torch.optim.LBFGS([weight,bias],lr=1,max_iter=10,history_size=20,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-15)
    calls=0;began=time.monotonic();start=0
    if (out/'head.pt').exists():
        saved=torch.load(out/'head.pt',map_location='cuda',weights_only=False)
        with torch.no_grad():weight.copy_(saved['weight']);bias.copy_(saved['bias'])
        opt.load_state_dict(saved['optimizer']);start=saved['step'];calls=saved['calls']
        del saved
    def prediction():return F.conv2d(padded,weight,bias).masked_fill(~wet,0)
    def closure():
        nonlocal calls
        calls+=1;opt.zero_grad(set_to_none=True);loss=mse(prediction(),y,wet);loss.backward();return loss
    for step in range(start,a.head_steps+1):
        if step>start:opt.step(closure)
        if step%10==0:
            with torch.no_grad():
                p=prediction();error=p-y
                row={'step':step,'closure_calls':calls,'seconds':time.monotonic()-began,'mse':float(mse(p,y,wet)),
                     'highpass_mse':float(mse(highpass(error,inside),torch.zeros_like(error),inside))}
            with (out/'head-history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(row,flush=True)
            torch.save({'weight':weight.detach().cpu(),'bias':bias.detach().cpu(),'optimizer':opt.state_dict(),'step':step,'calls':calls},out/'head.tmp.pt')
            (out/'head.tmp.pt').replace(out/'head.pt')
        if time.time()>=1792414800:break
    with torch.no_grad():
        np.savez_compressed(out/'head-maps.npz',prediction=prediction().cpu().numpy(),before=original_prediction.cpu().numpy(),truth=y.cpu().numpy(),mask=data.mask)
    (out/'COMPLETE').write_text('Gradient and frozen-head diagnostics complete; no exact optimum claim.\n')
if __name__=='__main__':main()
