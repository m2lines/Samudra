"""CPU reference and gradient checks for all production structure objectives."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import torch
from samudra.utils.structure_loss import StructureLoss


def main():
    torch.manual_seed(9)
    mask = torch.rand(3, 7, 8) > .2
    wet = mask.repeat(2, 1, 1)
    ctx = SimpleNamespace(label_mask=wet)
    p = torch.randn(2, 6, 7, 8, dtype=torch.float64, requires_grad=True)
    y = torch.randn_like(p)
    w = torch.rand(3, 7, 8) * .15 + .95
    e = (p-y)*wet
    state = e.square().mean((0, 2, 3))
    hp = torch.zeros_like(e)
    h = torch.zeros(6, dtype=p.dtype)
    for c in range(6):
        for j in range(7):
            for i in range(8):
                if not wet[c,j,i]:
                    continue
                h[c] += (e[:,c,j,(i+1)%8]-e[:,c,j,i]).abs().sum()
                if j < 6:
                    h[c] += (e[:,c,j+1,i]-e[:,c,j,i]).abs().sum()
                if 0 < j < 6 and all(wet[c,jj,ii] for jj,ii in [(j-1,i),(j+1,i),(j,(i-1)%8),(j,(i+1)%8)]):
                    hp[:,c,j,i] = e[:,c,j,i] - (e[:,c,j-1,i]+e[:,c,j+1,i]+e[:,c,j,(i-1)%8]+e[:,c,j,(i+1)%8])/4
    h /= 4*wet.sum((-2,-1))
    with TemporaryDirectory() as tmp:
        archive = Path(tmp)/'weights.npz'
        np.savez(archive, weights=w.numpy(), mask=mask.numpy())
        refs = {'local':(e.square()*w.repeat(2,1,1)).mean((0,2,3)), 'curvature':state+100*hp.square().mean((0,2,3)), 'horizontal':state+.1*h}
        for kind, ref in refs.items():
            loss = StructureLoss(kind, 100 if kind == 'curvature' else .1, str(archive))
            actual = loss(p,y,ctx)
            torch.testing.assert_close(actual,ref)
            assert torch.equal(loss(y,y,ctx),torch.zeros_like(actual))
            grad = torch.autograd.grad(actual.mean(),p,retain_graph=True)[0]
            assert (grad[:,~wet] == 0).all()
            q=p.detach().clone();r=q.clone();idx=(0,0,3,4);eps=1e-6
            q[idx]+=eps;r[idx]-=eps
            torch.testing.assert_close((loss(q,y,ctx).mean()-loss(r,y,ctx).mean())/(2*eps),grad[idx],rtol=1e-5,atol=1e-8)
    print('PASS: three independent references, repeated time channels, exact targets, land gradients, finite differences')

if __name__ == '__main__':
    main()
