"""Measure actual FP32 Adam parameter changes from a fixed checkpoint/gradient."""
import argparse,json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel,Prepared,mse


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists() or time.time()>=1792414800:return
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=Prepared(a.data);x,bs,ys=data.window(0,1,'cuda');wet=torch.tensor(data.mask[None],device='cuda')
 model=OceanModel(wet[0]).cuda();saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model'])
 model.train();loss=mse(model(x,bs[0])[0][:,:77],ys[0][:,:77],wet);loss.backward()
 gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None}
 report={'protocol':vars(a),'gradient_mode':'train; same fixed gradient and optimizer state for every LR','loss':float(loss),'learning_rates':{}}
 for lr in (1e-7,1e-6,1e-5,1e-4):
  model.load_state_dict(saved['model']);opt=torch.optim.Adam(model.parameters(),lr=lr);opt.load_state_dict(saved['optimizer'])
  for g in opt.param_groups:g['lr']=lr
  for n,p in model.named_parameters():p.grad=gradients[n].to('cuda') if n in gradients else None
  opt.step();rows={};total_active=0;total_unchanged=0;total_subulp=0
  for n,p in model.named_parameters():
   if p.grad is None:continue
   before=saved['model'][n].to('cuda');state=opt.state[p]
   active=(p.grad!=0)|(state['exp_avg']!=0)
   changed=(p.detach()!=before);count=int(active.sum());unchanged=int((active&~changed).sum())
   # Diagnostic magnitude of the next-step formula with post-step moments.
   group=opt.param_groups[0];b1,b2=group['betas'];step=float(state['step'])
   expected=lr*state['exp_avg'].double()/(1-b1**step)/(state['exp_avg_sq'].double().sqrt()/(1-b2**step)**.5+group['eps'])
   spacing=(torch.nextafter(before,torch.full_like(before,float('inf')))-before).double()
   sub=int((active&(expected.abs()<.5*spacing)).sum())
   rows[n]={'active_parameters':count,'unchanged_parameters':unchanged,'fraction_unchanged':unchanged/max(count,1),'expected_below_half_positive_ulp':sub,'actual_update_rms':float((p.detach().double()-before.double()).square().mean().sqrt())}
   total_active+=count;total_unchanged+=unchanged;total_subulp+=sub
  report['learning_rates'][str(lr)]={'active_parameters':total_active,'fraction_unchanged':total_unchanged/max(total_active,1),'fraction_expected_below_half_ulp':total_subulp/max(total_active,1),'layers':rows}
  (out/'report.json').write_text(json.dumps(report,indent=2));print(lr,total_unchanged/max(total_active,1),flush=True)
  del opt
 (out/'COMPLETE').write_text('Actual FP32 one-update diagnostic complete. No persistent checkpoint changed.\n')
if __name__=='__main__':main()
