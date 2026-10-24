"""Single fixed target overfit diagnostic on the pinned official v2 architecture."""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from common import OceanModel, Prepared, mse, highpass, interior_mask

ap = argparse.ArgumentParser()
ap.add_argument('--data', required=True)
ap.add_argument('--output', required=True)
ap.add_argument('--steps', type=int, default=5000)
args = ap.parse_args()
if args.steps < 2:
    raise ValueError('At least two updates required')
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
torch.manual_seed(15)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
data = Prepared(args.data)
meta = data.meta
start_index = meta['diagnostic_origins'][0]
x, forcings, targets = data.window(start_index, 1, 'cuda')
f = forcings[0]
y = targets[0][:, :77]
truth = y.cpu().numpy()
wet = torch.tensor(data.mask[None], device='cuda')
inside=interior_mask(wet)
m=OceanModel(wet[0]).cuda().train()
opt=torch.optim.Adam(m.parameters(),lr=6e-4)
steps=args.steps
first_step=0
if (out/"latest.pt").exists():
    ckpt=torch.load(out/"latest.pt", map_location="cuda", weights_only=False)
    m.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    first_step=ckpt["step"]
    del ckpt
(out/'protocol.json').write_text(json.dumps({'seed':15,'steps':steps,'initial_lr':6e-4,'final_lr':1e-7,'schedule':'cosine decay across all 5000 updates','init':'random','input_times':['2014-10-10','2014-10-15'],'target_time':'2014-10-20','loss':'first target only; equal-channel normalized wet-cell MSE','outputs':'second output head retained but unsupervised','temporal_feedback':False,'dtype':'float32; TF32 disabled','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'source':meta},indent=2))
start=time.monotonic()
for step in range(first_step,steps+1):
    if step % 100 == 0 or step in (0,1,10):
        # Record both batch-stat and running-stat predictions to expose BN effects.
        with torch.no_grad():
            # Snapshot train-mode buffers so measurement does not alter training.
            saved={k:v.clone() for k,v in m.named_buffers()}
            train_pred=m(x,f)[0][:,:77]
            for k,v in m.named_buffers():v.copy_(saved[k])
            m.eval(); pred=m(x,f)[0][:,:77]; m.train()
            err=pred-y
            row={'step':step,'seconds':time.monotonic()-start,'train_mse':float(mse(train_pred,y,wet)), 'eval_mse':float(mse(pred,y,wet)), 'highpass_error_mse':float(mse(highpass(err,inside),torch.zeros_like(err),inside)), 'channel_rmse_normalized':((err.square()*wet).sum((-2,-1))/wet.sum((-2,-1))).sqrt()[0].tolist()}
        with (out/'history.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='channel_rmse_normalized'}),flush=True)
        if step in (0,100,1000,steps):
            np.savez_compressed(out/f'maps-{step}.npz',prediction=pred.cpu().numpy(),train_prediction=train_pred.cpu().numpy(),truth=truth,mask=data.mask)
        if step==steps:break
    lr=1e-7+(6e-4-1e-7)*.5*(1+math.cos(math.pi*step/(steps-1)))
    for group in opt.param_groups:group['lr']=lr
    opt.zero_grad(set_to_none=True)
    pred=m(x,f)[0][:,:77]
    loss=mse(pred,y,wet)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    loss.backward();opt.step()
    if (step+1)%100==0 or step+1==steps:
        torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'step':step+1},out/'latest.tmp.pt')
        (out/'latest.tmp.pt').replace(out/'latest.pt')
(out/'COMPLETE').write_text(f'{steps} updates completed; inspect residuals before claiming zero error\n')
