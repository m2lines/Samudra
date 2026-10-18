"""Matched Adam with FP32 moments and either FP32 or FP64 master weights."""
import argparse,copy,json,math,signal,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel,Prepared,mse,highpass,interior_mask

ARMS=[(lr,precision) for lr in (1e-7,1e-6,1e-5) for precision in ('fp32','fp64')]

class MasterAdam:
    def __init__(self, params, saved, precision):
        self.params=list(params);assert len(saved['param_groups'])==1
        group=saved['param_groups'][0];assert not group.get('amsgrad',False) and not group.get('weight_decay',0)
        self.betas=group['betas'];self.eps=group['eps'];self.precision=precision
        self.master=[p.detach().to(dtype=torch.float64 if precision=='fp64' else torch.float32).clone() for p in self.params]
        self.states=[]
        for p,i in zip(self.params,group['params']):
            state=saved['state'][i]
            self.states.append({'m':state['exp_avg'].to(p.device,dtype=torch.float32).clone(),
                'v':state['exp_avg_sq'].to(p.device,dtype=torch.float32).clone(),'step':int(state['step'])})
        assert len(self.states)==len(self.params)

    @torch.no_grad()
    def step(self,lr):
        b1,b2=self.betas
        for p,master,state in zip(self.params,self.master,self.states):
            if p.grad is None:continue
            state['step']+=1;state['m'].lerp_(p.grad,1-b1);state['v'].mul_(b2).addcmul_(p.grad,p.grad,value=1-b2)
            denom=state['v'].sqrt().div_(math.sqrt(1-b2**state['step'])).add_(self.eps)
            master.addcdiv_(state['m'],denom,value=-lr/(1-b1**state['step']))
            p.copy_(master)

    def state_dict(self):
        return {'master':self.master,'states':self.states,'precision':self.precision}

    def load_state_dict(self,state):
        assert state['precision']==self.precision
        for p,target,source,saved in zip(self.params,self.master,state['master'],state['states']):
            target.copy_(source.to(p.device))
        self.states=[{'m':s['m'].to(p.device),'v':s['v'].to(p.device),'step':s['step']} for p,s in zip(self.params,state['states'])]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--arm',type=int,required=True);ap.add_argument('--steps',type=int,default=5000);a=ap.parse_args()
    lr,precision=ARMS[a.arm];out=Path(a.output)/f'{precision}-{lr:g}';out.mkdir(parents=True,exist_ok=True)
    if (out/'COMPLETE').exists() or time.time()>=1792414800:return
    torch.set_num_threads(4);torch.manual_seed(15);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=Prepared(a.data);x,bs,ys=data.window(0,1,'cuda');b=bs[0];y=ys[0][:,:77];wet=torch.tensor(data.mask[None],device='cuda');inside=interior_mask(wet)
    model=OceanModel(wet[0]).cuda().train();source=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(source['model'])
    # Fixed-gradient first-step parity against public torch.optim.Adam.
    loss=mse(model(x,b)[0][:,:77],y,wet);loss.backward()
    reference=torch.optim.Adam(model.parameters(),lr=lr,foreach=False);reference.load_state_dict(copy.deepcopy(source['optimizer']))
    for group in reference.param_groups:group['lr']=lr;group['foreach']=False
    reference.step();expected=[p.detach().clone() for p in model.parameters()]
    model.load_state_dict(source['model']);parity=MasterAdam(model.parameters(),source['optimizer'],'fp32');parity.step(lr)
    maximum=max(float((p-e).abs().max()) for p,e in zip(model.parameters(),expected))
    assert all(torch.equal(p,e) for p,e in zip(model.parameters(),expected)),f'Adam parity failed: {maximum}'
    del reference,expected,parity
    model.load_state_dict(source['model']);opt=MasterAdam(model.parameters(),source['optimizer'],precision);del source
    model.zero_grad(set_to_none=True);start=0;elapsed=0.
    if (out/'latest.pt').exists():
        saved=torch.load(out/'latest.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['step'];elapsed=saved['elapsed'];del saved
    stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGUSR1,stopping);signal.signal(signal.SIGTERM,stopping)
    began=time.monotonic()
    (out/'protocol.json').write_text(json.dumps(vars(a)|{'lr':lr,'master_precision':precision,'moment_precision':'fp32','forward_precision':'fp32','first_step_torch_adam_bitwise_parity':True,'parameters':sum(p.numel() for p in model.parameters())},indent=2))
    def save(step):
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'elapsed':elapsed+time.monotonic()-began},out/'latest.tmp.pt');(out/'latest.tmp.pt').replace(out/'latest.pt')
    def measure(step):
        model.eval()
        with torch.no_grad():
            p=model(x,b)[0][:,:77];err=p-y
            row={'step':step,'seconds':elapsed+time.monotonic()-began,'eval_mse':float(mse(p,y,wet)),'highpass_mse':float(mse(highpass(err,inside),torch.zeros_like(err),inside)),
                 'per_channel_mse':((err.square()*wet).sum((-2,-1))/wet.sum((-2,-1))).mean(0).tolist()}
            if step in (0,a.steps):np.savez_compressed(out/f'maps-{step}.npz',prediction=p.cpu().numpy(),truth=y.cpu().numpy(),mask=data.mask)
        model.train()
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print({k:v for k,v in row.items() if k!='per_channel_mse'},flush=True)
    measure(start)
    for step in range(start,a.steps):
        model.zero_grad(set_to_none=True);loss=mse(model(x,b)[0][:,:77],y,wet)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();opt.step(lr)
        if (step+1)%100==0 or step+1==a.steps:measure(step+1);save(step+1)
        if stop:save(step+1);sys.exit(75)
        if (step+1)%10==0 and time.time()>=1792414800:save(step+1);(out/'DEADLINE').write_text('Deadline\n');return
    (out/'COMPLETE').write_text('Master-precision continuation complete.\n')
if __name__=='__main__':main()
