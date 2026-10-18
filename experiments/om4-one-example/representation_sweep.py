"""Matched full-width, single-example output-parameterization controls."""
import argparse,hashlib,json,math,signal,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel,Prepared,mse,highpass,interior_mask
ARMS=[('absolute_random',None,False),('absolute_zero',None,True),('same_slot_random',0,False),('same_slot_zero',0,True),('latest_zero',77,True)]


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--output',required=True);ap.add_argument('--arm',type=int,required=True);ap.add_argument('--steps',type=int,default=5000);ap.add_argument('--seed',type=int,default=15);a=ap.parse_args()
 name,base_start,zero=ARMS[a.arm];out=Path(a.output)/name;out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists() or time.time()>=1792414800:return
 torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=Prepared(a.data);x,bs,ys=data.window(0,1,'cuda');b=bs[0];y=ys[0][:,:77];wet=torch.tensor(data.mask[None],device='cuda');inside=interior_mask(wet)
 model=OceanModel(wet[0].cpu()).train();digest=hashlib.sha256()
 for p in model.unet.parameters():digest.update(p.detach().numpy().tobytes())
 if zero:
  with torch.no_grad():model.decoder.weight.zero_();model.decoder.bias.zero_()
 model.cuda();opt=torch.optim.Adam(model.parameters(),lr=6e-4)
 start=0;elapsed=0.
 if (out/'latest.pt').exists():
  saved=torch.load(out/'latest.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);start=saved['step'];elapsed=saved['elapsed'];del saved
 began=time.monotonic();stop=False
 def stopping(*_):
  nonlocal stop
  stop=True
 signal.signal(signal.SIGUSR1,stopping);signal.signal(signal.SIGTERM,stopping)
 (out/'protocol.json').write_text(json.dumps(vars(a)|{'name':name,'base_first_channel':base_start,'zero_head':zero,'backbone_initial_sha256':digest.hexdigest(),
  'parameters':sum(p.numel() for p in model.parameters()),'schedule':'cosine 6e-4 to 1e-7, denominator steps-1 as original','supervision':'first target, all 77 channels','temporal_feedback':False,
  'same_slot':'first output anchored to first input, as existing legacy 2-in/2-out residual option','latest':'first output anchored to latest input, 5-day persistence'},indent=2))
 def predict():
  raw=model(x,b)[0][:,:77]
  return raw if base_start is None else raw+x[:,base_start:base_start+77]
 def save(step):
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'elapsed':elapsed+time.monotonic()-began},out/'latest.tmp.pt');(out/'latest.tmp.pt').replace(out/'latest.pt')
 def measure(step):
  with torch.no_grad():
   buffers={k:v.clone() for k,v in model.named_buffers()};model.train();pt=predict()
   for k,v in model.named_buffers():v.copy_(buffers[k])
   model.eval();p=predict();e=p-y
   row={'step':step,'seconds':elapsed+time.monotonic()-began,'train_mse':float(mse(pt,y,wet)),'eval_mse':float(mse(p,y,wet)),
    'highpass_mse':float(mse(highpass(e,inside),torch.zeros_like(e),inside)),
    'per_channel_mse':((e.square()*wet).sum((-2,-1))/wet.sum((-2,-1))).mean(0).tolist()}
   if step in (0,a.steps):np.savez_compressed(out/f'maps-{step}.npz',prediction=p.cpu().numpy(),truth=y.cpu().numpy(),mask=data.mask)
  model.train()
  with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  print({k:v for k,v in row.items() if k!='per_channel_mse'},flush=True)
 measure(start)
 for step in range(start,a.steps):
  lr=1e-7+(6e-4-1e-7)*.5*(1+math.cos(math.pi*step/(a.steps-1)))
  for group in opt.param_groups:group['lr']=lr
  opt.zero_grad(set_to_none=True);loss=mse(predict(),y,wet)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  loss.backward();opt.step()
  if (step+1)%100==0 or step+1==a.steps:measure(step+1);save(step+1)
  if stop:save(step+1);sys.exit(75)
  if (step+1)%10==0 and time.time()>=1792414800:save(step+1);(out/'DEADLINE').write_text('Deadline\n');return
 (out/'COMPLETE').write_text('Matched single-example representation control complete.\n')
if __name__=='__main__':main()
