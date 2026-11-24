"""OM4 adaptation of Cody's LLC first-difference L1 penalties (2651acfff).

Uses normalized fields, wet-cell reductions, zero-filled land for horizontal
forward differences, and mean-one reciprocal depth spacing for vertical pairs.
No geographic area weighting, sample/tile weighting, or special SSH multiplier.
"""
import torch
import torch.nn.functional as F

DEPTHS = [2.5, 10, 22.5, 40, 65, 105, 165, 250, 375, 550, 775, 1050,
          1400, 1850, 2400, 3100, 4000, 5000, 6000]


def gradient_components(pred, target, wet):
    if pred.shape[1] % 77:
        raise ValueError('Expected time-major groups of 77 OM4 channels')
    wet = wet.expand_as(pred)
    error = (pred - target) * wet
    dx = error.roll(-1, -1) - error
    dy = F.pad(error[..., 1:, :] - error[..., :-1, :], (0, 0, 0, 1))
    denominator = wet.sum((0, 2, 3)).clamp_min(1)
    horizontal = (((dx.abs() + dy.abs()) * wet).sum((0, 2, 3)) / denominator / 2).mean()
    spacing = torch.diff(pred.new_tensor(DEPTHS))
    pair_weight = spacing.reciprocal()
    pair_weight = pair_weight / pair_weight.mean()
    channel_terms = []
    for t in range(pred.shape[1] // 77):
        for v in range(4):
            start = t * 77 + v * 19
            e = (pred - target)[:, start:start+19]
            valid = wet[:, start:start+18] & wet[:, start+1:start+19]
            counts = valid.sum((0, 2, 3))
            pair = ((e[:, 1:] - e[:, :-1]).abs() * valid).sum((0, 2, 3))
            pair = pair / counts.clamp_min(1) * pair_weight
            exists = (counts > 0).to(pred.dtype)
            sums = F.pad(pair, (0, 1)) + F.pad(pair, (1, 0))
            n = F.pad(exists, (0, 1)) + F.pad(exists, (1, 0))
            channel_terms.append(sums / n.clamp_min(1))
        channel_terms.append(pred.new_zeros(1))  # SSH has no vertical column.
    return horizontal, torch.cat(channel_terms).mean()


def self_test():
    torch.manual_seed(1)
    pred = torch.randn(2, 154, 5, 7, dtype=torch.float64, requires_grad=True)
    target = torch.randn_like(pred)
    wet = torch.ones(1, 154, 5, 7, dtype=torch.bool)
    wet[..., 0, :] = False
    wet[:, 10::19, 2, 2] = False
    h, z = gradient_components(pred, target, wet)
    assert all(x == 0 for x in gradient_components(target, target, wet))
    # Batch replication and time-major grouping must not change reduction.
    h1, z1 = gradient_components(pred[:1], target[:1], wet)
    hr, zr = gradient_components(pred[:1].repeat(2,1,1,1), target[:1].repeat(2,1,1,1), wet)
    torch.testing.assert_close(h1, hr); torch.testing.assert_close(z1, zr)
    halves = [gradient_components(pred[:, s:s+77], target[:, s:s+77], wet[:, s:s+77]) for s in (0,77)]
    torch.testing.assert_close(h, (halves[0][0]+halves[1][0])/2)
    torch.testing.assert_close(z, (halves[0][1]+halves[1][1])/2)
    grad = torch.autograd.grad(h+z, pred)[0]
    assert (grad[~wet.expand_as(pred)] == 0).all()
    i = (0, 58, 3, 3); eps = 1e-6
    plus = pred.detach().clone(); minus = pred.detach().clone()
    plus[i] += eps; minus[i] -= eps
    fd = (sum(gradient_components(plus,target,wet))-sum(gradient_components(minus,target,wet)))/(2*eps)
    torch.testing.assert_close(fd, grad[i], rtol=1e-5, atol=1e-9)
    # Independent explicit loop reproduces vertical channel attribution.
    reference = pred.new_zeros(154); number = pred.new_zeros(154)
    inv = 1/torch.diff(pred.new_tensor(DEPTHS)); inv /= inv.mean()
    for t in range(2):
        for v in range(4):
            for d in range(18):
                c=t*77+v*19+d
                valid=(wet[:,c]&wet[:,c+1]).expand(2,-1,-1)
                if valid.any():
                    diff=(pred[:,c+1]-target[:,c+1])-(pred[:,c]-target[:,c])
                    val=diff[valid].abs().mean()*inv[d]
                    reference[c:c+2]+=val; number[c:c+2]+=1
    torch.testing.assert_close(z,(reference/number.clamp_min(1)).mean())
    print('Gradient checks passed: exact target, land gradients, batch/time parity, finite differences, independent vertical reference.')

if __name__ == '__main__':
    self_test()
