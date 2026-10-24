"""Separate learned prediction error from BatchNorm train/evaluation-statistics gaps."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel,Prepared,mse,highpass,interior_mask


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--root',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if time.time()>=1792414800:return
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=Prepared(a.data);x,bs,ys=data.window(0,1,'cuda');y=ys[0][:,:77];wet=torch.tensor(data.mask[None],device='cuda');inside=interior_mask(wet)
 model=OceanModel(wet[0]).cuda();report={}
 cases=[('original',Path(a.checkpoint))]+[(p.parent.name,p) for p in sorted(Path(a.root).glob('*/latest.pt'))]
 def measure(p):
  e=p-y
  return {'mse':float(mse(p,y,wet)),'highpass_mse':float(mse(highpass(e,inside),torch.zeros_like(e),inside)),
          'per_channel_mse':((e.square()*wet).sum((-2,-1))/wet.sum((-2,-1))).mean(0).tolist()}
 with torch.no_grad():
  for name,path in cases:
   saved=torch.load(path,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);del saved
   model.eval();p0=model(x,bs[0])[0][:,:77];row={'checkpoint':str(path),'stored_eval':measure(p0)}
   # Capture per-batch biased moments used in training, without changing weights.
   moments={};hooks=[]
   for n,m in model.named_modules():
    if isinstance(m,nn.BatchNorm2d):
     def hook(mod,inputs,n=n):
      t=inputs[0];moments[n]=(t.mean((0,2,3)),t.var((0,2,3),unbiased=False))
     hooks.append(m.register_forward_pre_hook(hook))
   model.train();pt=model(x,bs[0])[0][:,:77];row['train_batch_statistics']=measure(pt)
   for hook in hooks:hook.remove()
   for n,m in model.named_modules():
    if n in moments:m.running_mean.copy_(moments[n][0]);m.running_var.copy_(moments[n][1])
   model.eval();pc=model(x,bs[0])[0][:,:77];row['eval_exact_batch_calibration']=measure(pc)
   row['train_vs_calibrated_max_difference']=float((pt-pc).abs().max())
   report[name]=row;(out/'report.json').write_text(json.dumps(report,indent=2))
   np.savez_compressed(out/f'{name}-selected.npz',before=p0[:,[13,32,51,70,76]].cpu().numpy(),calibrated=pc[:,[13,32,51,70,76]].cpu().numpy(),truth=y[:,[13,32,51,70,76]].cpu().numpy())
   print(name,{k:v['mse'] for k,v in row.items() if isinstance(v,dict)},flush=True)
 (out/'COMPLETE').write_text('All available endpoint normalization audits complete. Weights unchanged.\n')
if __name__=='__main__':main()
