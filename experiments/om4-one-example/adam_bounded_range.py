"""Matched random-init one-example Adam fits: MSE, spatial loss, local-range MSE."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel, Prepared, mse, highpass, interior_mask
from llc_gradient_loss import gradient_components


def weighted_mse(pred, target, wet, weights):
    return (((pred-target).square()*wet*weights).sum((-2,-1))/wet.sum((-2,-1))).mean()


def objective(pred,target,wet,inside,weights,arm):
    if arm=='local_range':return weighted_mse(pred,target,wet,weights)
    state=mse(pred,target,wet)
    if arm=='mse':return state
    if arm in ('gradient_h', 'gradient_hz'):
        h,z=gradient_components(pred,target,wet)
        return state + .1*h + (.1*z if arm=='gradient_hz' else 0)
    assert arm=='spatial_100'
    h=highpass(pred-target,inside)
    return state+100*mse(h,torch.zeros_like(h),inside)


def self_test():
    torch.manual_seed(0)
    p=torch.randn(1,3,7,8,dtype=torch.float64,requires_grad=True)
    y=torch.randn_like(p);wet=torch.ones_like(p,dtype=torch.bool);wet[...,0,:]=False
    one=torch.ones_like(p)
    torch.testing.assert_close(weighted_mse(p,y,wet,one),mse(p,y,wet),rtol=0,atol=0)
    weights=torch.rand_like(p)+.1
    loss=weighted_mse(p,y,wet,weights);grad=torch.autograd.grad(loss,p)[0]
    assert (grad[~wet]==0).all()
    index=(0,1,3,4);eps=1e-6
    q=p.detach().clone();q[index]+=eps
    r=p.detach().clone();r[index]-=eps
    fd=(weighted_mse(q,y,wet,weights)-weighted_mse(r,y,wet,weights))/(2*eps)
    torch.testing.assert_close(fd,grad[index],rtol=1e-7,atol=1e-9)
    assert weighted_mse(y,y,wet,weights)==0
    print('CPU loss checks passed: unit-weight MSE parity, exact target zero, land gradients zero, finite-difference gradient.')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True);ap.add_argument('--scales',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--arm',choices=['mse','spatial_100','local_range','gradient_h','gradient_hz'],required=True)
    ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--seed',type=int,default=15)
    a=ap.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists():return
    torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Prepared(a.data);x,bs,ys=data.window(0,1,'cuda');b,y=bs[0],ys[0][:,:77]
    scales=np.load(a.scales)
    assert np.array_equal(scales['mask'],data.mask)
    weights=torch.tensor(scales['weights'][None],device='cuda')
    assert torch.isfinite(weights).all() and (weights>=0).all()
    wet=torch.tensor(data.mask[None],device='cuda');inside=interior_mask(wet)
    model=OceanModel(wet[0].cpu()).train()
    digest=hashlib.sha256()
    for p in model.parameters():digest.update(p.detach().numpy().tobytes())
    model.cuda();optimizer=torch.optim.Adam(model.parameters(),lr=6e-4,foreach=False)
    protocol=vars(a)|{'initial_parameters_sha256':digest.hexdigest(),
        'parameters':sum(p.numel() for p in model.parameters()),
        'scales_sha256':hashlib.sha256(Path(a.scales).read_bytes()).hexdigest(),
        'normalization':'unchanged global training channel mean/std; weights only affect loss',
        'optimizer':'Adam, fresh random initialization; default betas/epsilon; no weight decay; foreach=False',
        'schedule':f'cosine6e-4 to1e-7 over{a.steps}updates',
        'supervision':'first target all77 channels; two input timestamps; train-mode BatchNorm; no temporal feedback',
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
        'objective':{'mse':'original normalized wet equal-channel MSE','spatial_100':'MSE+100*interior curvature-error MSE','local_range':'fixed spatial weighting loaded from recorded scale archive; no derivative term','gradient_h':'MSE+0.1*horizontal forward-difference L1','gradient_hz':'MSE+0.1*horizontal L1+0.1*spacing-normalized vertical L1'}[a.arm]}
    start=0;elapsed=0.
    if (out/'latest.pt').exists():
        old=json.loads((out/'protocol.json').read_text())
        for key in ['arm','steps','seed','scales_sha256','initial_parameters_sha256']:assert old[key]==protocol[key],key
        saved=torch.load(out/'latest.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['step'];elapsed=saved['elapsed']
        torch.set_rng_state(saved['rng_cpu'].cpu());torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
        del saved
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    began=time.monotonic();stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGUSR1,stopping);signal.signal(signal.SIGTERM,stopping)
    def save(step):
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'step':step,
                    'elapsed':elapsed+time.monotonic()-began,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()},out/'latest.tmp.pt')
        (out/'latest.tmp.pt').replace(out/'latest.pt')
    def measure(step):
        with torch.no_grad():
            # Match the earlier training-vs-evaluation diagnostic without updating BN statistics.
            buffers={n:v.clone() for n,v in model.named_buffers()}
            model.train();train=model(x,b)[0][:,:77]
            for n,v in model.named_buffers():v.copy_(buffers[n])
            model.eval();pred=model(x,b)[0][:,:77];e=pred-y;h=highpass(e,inside)
            gh,gz=gradient_components(pred,y,wet)
            row={'step':step,'seconds':elapsed+time.monotonic()-began,
                 'train_mse':float(mse(train,y,wet)),'eval_mse':float(mse(pred,y,wet)),
                 'training_objective_eval':float(objective(pred,y,wet,inside,weights,a.arm)),
                 'weighted_mse_eval':float(weighted_mse(pred,y,wet,weights)),
                 'highpass_mse':float(mse(h,torch.zeros_like(h),inside)),
                 'horizontal_l1':float(gh),'vertical_l1':float(gz),
                 'per_channel_mse':((e.square()*wet).sum((-2,-1))/wet.sum((-2,-1))).mean(0).tolist(),
                 'per_channel_highpass_mse':((h.square()*inside).sum((-2,-1))/inside.sum((-2,-1))).mean(0).tolist()}
            if step in (0, a.steps//2, a.steps):
                np.savez_compressed(out/f'maps-{step}.npz',prediction=pred.cpu().numpy(),truth=y.cpu().numpy(),mask=data.mask)
        model.train()
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print({k:v for k,v in row.items() if not k.startswith('per_channel')},flush=True)
    measure(start)
    for step in range(start,a.steps):
        lr=1e-7+(6e-4-1e-7)*.5*(1+math.cos(math.pi*step/(a.steps-1)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        loss=objective(model(x,b)[0][:,:77],y,wet,inside,weights,a.arm)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();optimizer.step()
        if (step+1)%100==0 or step+1==a.steps:measure(step+1);save(step+1)
        if stop:save(step+1);sys.exit(75)
    (out/'COMPLETE').write_text('Completed matched Adam local-range comparison\n')


if __name__=='__main__':
    if sys.argv[1:]==['--self-test']:self_test()
    else:main()
