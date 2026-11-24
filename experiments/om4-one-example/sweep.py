"""Matched single-example continuation experiments; atomic preemption recovery."""
import argparse,json,math,os,signal,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel,Prepared,mse,highpass,interior_mask
from legacy.activations import CappedGELU

ARMS=[('tiny_lr',1e-7,False,False),('lr1e5',1e-5,False,False),('lr1e4',1e-4,False,False),('lr3e4',3e-4,False,False),('reset1e5',1e-5,True,False),('reset1e4',1e-4,True,False),('reset3e4',3e-4,True,False),('adam_eval',1e-4,False,True),('lbfgs',1.,True,True)]

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);ap.add_argument('--arm',type=int,required=True);ap.add_argument('--steps',type=int,default=5000);args=ap.parse_args()
 name,lr,reset,eval_mode=ARMS[args.arm];out=Path(args.output)/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists():return
 torch.set_num_threads(4);torch.manual_seed(15);np.random.seed(15);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=Prepared(args.data);x,bs,ys=data.window(0,1,'cuda');b=bs[0];y=ys[0][:,:77];wet=torch.tensor(data.mask[None],device='cuda');inside=interior_mask(wet)
 model=OceanModel(wet[0]).cuda();model.train(not eval_mode)
 opt=torch.optim.LBFGS(model.parameters(),lr=lr,max_iter=10,history_size=10,line_search_fn='strong_wolfe',tolerance_grad=1e-12,tolerance_change=1e-15) if name=='lbfgs' else torch.optim.Adam(model.parameters(),lr=lr)
 saved=torch.load(out/'latest.pt' if (out/'latest.pt').exists() else args.checkpoint,map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
 resume=(out/'latest.pt').exists()
 if resume or not reset:opt.load_state_dict(saved['optimizer'])
 for group in opt.param_groups:group['lr']=lr
 start=saved['step'] if resume else 0;elapsed=saved.get('elapsed',0.) if resume else 0.
 if resume:torch.set_rng_state(saved['rng_cpu'].cpu());torch.cuda.set_rng_state(saved['rng_cuda'].cpu())
 del saved
 stop=False
 def stopping(*_):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGUSR1,stopping);signal.signal(signal.SIGTERM,stopping)
 began=time.monotonic();count=100 if name=='lbfgs' else args.steps
 (out/'protocol.json').write_text(json.dumps(vars(args)|{'name':name,'lr':lr,'reset_moments':reset,'eval_mode':eval_mode,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'parameters':sum(p.numel() for p in model.parameters()),'preemptible':True},indent=2))
 def checkpoint(step):
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'elapsed':elapsed+time.monotonic()-began,'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state()},out/'latest.tmp.pt');(out/'latest.tmp.pt').replace(out/'latest.pt')
 def measure(step):
  was=model.training;model.eval()
  with torch.no_grad():
   p=model(x,b)[0][:,:77];e=p-y
   per=((e.square()*wet).sum((-2,-1))/wet.sum((-2,-1))).mean(0)
   hp=highpass(e,inside)
   row={'step':step,'seconds':elapsed+time.monotonic()-began,'eval_mse':float(mse(p,y,wet)),'highpass_mse':float(mse(hp,torch.zeros_like(hp),inside)),'per_channel_mse':per.tolist(),'cuda_peak_bytes':torch.cuda.max_memory_allocated()}
   if step in (0,count):np.savez_compressed(out/f'maps-{step}.npz',prediction=p.cpu().numpy(),truth=y.cpu().numpy(),mask=data.mask)
  model.train(was)
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print(json.dumps({k:v for k,v in row.items() if k!='per_channel_mse'}),flush=True)
 # Gradient and saturation probe with frozen statistics, unchanged model/optimizer.
 if not resume:
  model.eval();caps={};hooks=[]
  for key,module in model.named_modules():
   if isinstance(module,CappedGELU):
    def hook(mod,inputs,output,key=key):caps[key]={'fraction_capped':float((output>=mod.cap).float().mean()),'rms':float(output.square().mean().sqrt())}
    hooks.append(module.register_forward_hook(hook))
  loss=mse(model(x,b)[0][:,:77],y,wet);loss.backward()
  grads={k:{'grad_norm':float(p.grad.norm()),'weight_norm':float(p.norm())} for k,p in model.named_parameters() if p.grad is not None}
  for hook in hooks:hook.remove()
  (out/'gradient-probe.json').write_text(json.dumps({'eval_mse':float(loss),'layers':grads,'activations':caps},indent=2));opt.zero_grad(set_to_none=True);model.train(not eval_mode)
 measure(start)
 for step in range(start,count):
  def closure():
   opt.zero_grad(set_to_none=True);loss=mse(model(x,b)[0][:,:77],y,wet)
   if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
   loss.backward();return loss
  loss=opt.step(closure) if name=='lbfgs' else closure()
  if name!='lbfgs':opt.step()
  if (step+1)%100==0 or name=='lbfgs' and (step+1)%10==0 or step+1==count:
   measure(step+1);checkpoint(step+1)
  if stop:
   checkpoint(step+1);sys.exit(75)
  if (step+1)%10==0 and time.time()>=1792414800: # 2026-10-19 09:00 America/New_York
   checkpoint(step+1);(out/'DEADLINE').write_text('Budget deadline reached\n');return
 (out/'COMPLETE').write_text('Completed planned continuation, not a zero-error claim\n')
if __name__=='__main__':main()
