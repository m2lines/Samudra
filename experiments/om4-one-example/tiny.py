"""Parameter-matched tiny-model architecture controls with hard local memory guards."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'om4-temporal-memory'))
from common import OceanModel, Prepared, mse, highpass, interior_mask
from legacy.blocks import CoreBlock, ZonallyPeriodicBilinearUpsample
from legacy.activations import CappedGELU

DEADLINE = 1792414800
FULL_PARAMETERS = 84066474
VARIANTS = ('baseline', 'final_dilation1', 'decoder_dilation1', 'all_dilation1', 'uncapped', 'exact_upsample')

class TinyModel(OceanModel):
    def __init__(self, wet, variant):
        super().__init__(wet, widths=(16, 24, 32, 40))
        self.variant = variant
        for i, block in enumerate(self.unet.layers):
            change = (variant == 'all_dilation1' or
                      variant == 'decoder_dilation1' and i > self.middle_index or
                      variant == 'final_dilation1' and i == len(self.unet.layers)-1)
            if change and isinstance(block, CoreBlock):
                block.N_pad = 1
                for layer in block.modules():
                    if isinstance(layer, nn.Conv2d) and layer.kernel_size == (3, 3):
                        layer.dilation = (1, 1)
        if variant == 'uncapped':
            for layer in self.modules():
                if isinstance(layer, CappedGELU):
                    layer.cap.fill_(float('inf'))

    def _features(self, x, h):
        if self.variant != 'exact_upsample':
            return super()._features(x, h)
        skips = []
        up = 0
        for i, layer in enumerate(self.unet.layers):
            if isinstance(layer, ZonallyPeriodicBilinearUpsample):
                skip = skips[-1-up]
                # Resize both axes directly to the skip shape. Three periodic
                # copies make the middle copy interpolate across the seam.
                width = skip.shape[-1]
                padded = torch.cat((x,x,x), dim=-1)
                x = F.interpolate(padded, size=(skip.shape[-2], 3*width),
                                  mode='bilinear', align_corners=False)[...,width:2*width]
                x = x + skip
                up += 1
            else:
                x = self._block(layer, x)
            if i < self.middle_index and isinstance(layer, CoreBlock):
                skips.append(x)
        return x, h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--variant', choices=VARIANTS, default='baseline')
    ap.add_argument('--steps', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=15)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--target', choices=('real','constant','smooth'), default='real')
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if time.time() >= DEADLINE or (out/'COMPLETE').exists():
        return
    torch.set_num_threads(2)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.device == 'cuda':
        _, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction(min(.08, 6*1024**3/total))
    data = Prepared(args.data)
    x, boundaries, targets = data.window(0, 1, args.device)
    b, y = boundaries[0], targets[0][:,:77]
    wet = torch.tensor(data.mask[None], device=args.device)
    inside = interior_mask(wet)
    if args.target != 'real':
        yy = torch.linspace(-1,1,y.shape[-2],device=args.device)[None,None,:,None]
        xx = torch.arange(y.shape[-1],device=args.device)[None,None,None,:]*2*math.pi/y.shape[-1]
        y = ((torch.zeros_like(y)+0.5) if args.target=='constant' else
             (torch.zeros_like(y)+.5*torch.cos(xx)*torch.cos(yy*math.pi/2))) * wet
    model = TinyModel(wet[0], args.variant)
    count = sum(p.numel() for p in model.parameters())
    assert count < .1*FULL_PARAMETERS, count
    digest = hashlib.sha256(b''.join(p.detach().numpy().tobytes() for p in model.parameters())).hexdigest()
    model.to(args.device).train()
    opt = torch.optim.Adam(model.parameters(), lr=6e-4)
    start = 0; elapsed = 0.
    if (out/'latest.pt').exists():
        saved = torch.load(out/'latest.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        start=saved['step']; elapsed=saved['elapsed']; del saved
    (out/'protocol.json').write_text(json.dumps(vars(args)|{'parameters':count,'fraction_full':count/FULL_PARAMETERS,'initial_parameter_sha256':digest,'gpu_allocator_limit_gib':6,'minimum_available_ram_gib':32,'deadline':DEADLINE},indent=2))
    began = time.monotonic(); stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM, stopping);signal.signal(signal.SIGUSR1, stopping)
    def save(step):
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'elapsed':elapsed+time.monotonic()-began},out/'latest.tmp.pt')
        (out/'latest.tmp.pt').replace(out/'latest.pt')
    def measure(step):
        model.eval()
        with torch.no_grad():
            p=model(x,b)[0][:,:77]; e=p-y
            row={'step':step,'seconds':elapsed+time.monotonic()-began,'eval_mse':float(mse(p,y,wet)),
                 'highpass_mse':float(mse(highpass(e,inside),torch.zeros_like(e),inside)),
                 'per_channel_mse':((e.square()*wet).sum((-2,-1))/wet.sum((-2,-1))).mean(0).tolist(),
                 'cuda_peak_bytes':torch.cuda.max_memory_allocated() if args.device=='cuda' else 0}
            if step in (0,args.steps):
                np.savez_compressed(out/f'maps-{step}.npz',prediction=p.cpu().numpy(),truth=y.cpu().numpy(),mask=data.mask)
        model.train()
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='per_channel_mse'}),flush=True)
    measure(start)
    for step in range(start,args.steps):
        available = next(int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
        if available < 32*1024**3 or stop or time.time()>=DEADLINE:
            save(step);(out/'STOPPED').write_text(f'RAM={available}, signal={stop}, time={time.time()}\n');return
        # Same schedule across arms, with an actual cooldown, as in original fit.
        lr=1e-7+(6e-4-1e-7)*.5*(1+math.cos(math.pi*step/args.steps))
        for g in opt.param_groups:g['lr']=lr
        opt.zero_grad(set_to_none=True)
        loss=mse(model(x,b)[0][:,:77],y,wet)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();opt.step()
        if (step+1)%100==0 or step+1==args.steps:
            measure(step+1);save(step+1)
    (out/'COMPLETE').write_text('Planned updates finished.\n')

if __name__=='__main__':main()
