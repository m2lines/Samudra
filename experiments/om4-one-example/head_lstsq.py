"""Float64 QR/SVD least-squares floor for selected outputs of a frozen backbone."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'om4-temporal-memory'))
from common import OceanModel,Prepared,PROG


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--checkpoint',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE').exists() or time.time()>=1792414800:return
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 data=Prepared(a.data);x,bs,ys=data.window(0,1,'cuda');y=ys[0][:,:77];wet=torch.tensor(data.mask,device='cuda')
 model=OceanModel(wet).cuda().eval();saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);del saved
 with torch.no_grad():
  before=model(x,bs[0])[0][:,:77]
  features=model._features(torch.cat((x,bs[0]),dim=1),None)[0]
  padded=F.pad(F.pad(features,(1,1,0,0),mode='circular'),(0,0,1,1))
  design=F.unfold(padded,kernel_size=3)[0].T.contiguous()
  design=torch.cat((design,torch.ones_like(design[:,:1])),dim=1)
  # Verify patch ordering and inclusion of bias against the actual decoder.
  decoder=torch.cat((model.decoder.weight[:77].flatten(1),model.decoder.bias[:77,None]),dim=1)
  direct=(design@decoder.T).T.reshape_as(before).masked_fill(~wet[None],0)
  maximum=float((direct-before).abs().max())
  torch.testing.assert_close(direct,before,rtol=2e-5,atol=2e-5)
 del model,padded,features,x,bs,ys,direct,decoder
 selected=[13,32,51,70,76];groups={}
 for c in selected:groups.setdefault(data.mask[c].tobytes(),[]).append(c)
 report={'protocol':vars(a),'method':'Frozen FP32 features; float64 thin QR followed by SVD of R; relative rank threshold max(m,n)*float64 epsilon','decoder_reconstruction_max_error':maximum,'groups':[]}
 fitted={};began=time.monotonic()
 with torch.no_grad():
  for group in groups.values():
   mask=wet[group[0]].flatten();X=design[mask].double();Y=y[0,group].flatten(1).T[mask].double()
   Q,R=torch.linalg.qr(X,mode='reduced')
   U,S,Vh=torch.linalg.svd(R,full_matrices=False)
   cutoff=max(X.shape)*torch.finfo(torch.float64).eps*S[0];keep=S>cutoff
   rhs=U.T@(Q.T@Y);scaled=torch.where(keep[:,None],rhs/S.clamp_min(torch.finfo(torch.float64).tiny)[:,None],0.)
   coef=Vh.T@scaled;fit=X@coef;residual=fit-Y
   gradient=X.T@residual/X.shape[0]
   row={'channels':[PROG[c] for c in group],'rows':X.shape[0],'columns':X.shape[1],'rank':int(keep.sum()),'largest_singular':float(S[0]),'smallest_singular':float(S[-1]),'rank_cutoff':float(cutoff),'elapsed':time.monotonic()-began,'outputs':{}}
   for j,c in enumerate(group):
    pred=(design.double()@coef[:,j]).reshape(wet[c].shape).masked_fill(~wet[c],0)
    # Measure the actual float64 solution and its deployable float32 coefficient version.
    pred32=(design@coef[:,j].float()).reshape(wet[c].shape).masked_fill(~wet[c],0)
    err0=(before[0,c]-y[0,c])[wet[c]];err64=(pred-y[0,c])[wet[c]];err32=(pred32-y[0,c])[wet[c]]
    row['outputs'][PROG[c]]={'before_mse':float(err0.square().mean()),'least_squares_mse_float64':float(err64.square().mean()),'least_squares_mse_float32':float(err32.square().mean()),'normal_equation_gradient_rms':float(gradient[:,j].square().mean().sqrt()),'coefficient_norm':float(coef[:,j].norm()),'rmse_physical':float(err64.square().mean().sqrt()*data.std[c])}
    fitted[PROG[c]]=pred.cpu().numpy()
   report['groups'].append(row);(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
   np.savez_compressed(out/f'coefficients-{group[0]}.npz',coefficients=coef.cpu().numpy(),channels=np.array(group))
   del X,Y,Q,R,U,S,Vh,coef,fit,residual,gradient,pred,pred32
 np.savez_compressed(out/'fitted-maps.npz',**fitted)
 (out/'COMPLETE').write_text('Selected-channel frozen-feature least-squares diagnostic complete.\n')
if __name__=='__main__':main()
