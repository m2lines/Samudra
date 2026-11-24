"""Separate first-target salinity decoder parameters without changing convolution."""
import torch
from torch import nn
from torch.nn import functional as F


class SalinityHead(nn.Module):
    """Preserve all 154 output rows; separately own first-target rows 57:76."""
    def __init__(self, conv):
        super().__init__()
        assert isinstance(conv, nn.Conv2d) and conv.out_channels == 154
        assert conv.bias is not None and conv.groups == 1 and conv.padding_mode == 'zeros'
        self.stride, self.padding, self.dilation = conv.stride, conv.padding, conv.dilation
        for name, sl in [('before', slice(0,57)), ('salinity', slice(57,76)), ('after', slice(76,None))]:
            self.register_parameter(name+'_weight', nn.Parameter(conv.weight[sl].detach().clone()))
            self.register_parameter(name+'_bias', nn.Parameter(conv.bias[sl].detach().clone()))

    def forward(self, x):
        weight = torch.cat([self.before_weight, self.salinity_weight, self.after_weight])
        bias = torch.cat([self.before_bias, self.salinity_bias, self.after_bias])
        return F.conv2d(x, weight, bias, self.stride, self.padding, self.dilation)

    def salinity_parameters(self):
        return [self.salinity_weight, self.salinity_bias]


def parameter_groups(model, multiplier):
    salinity = model.decoder.salinity_parameters()
    ids = {id(p) for p in salinity}
    rest = [p for p in model.parameters() if id(p) not in ids]
    return [{'params': rest, 'lr_multiplier': 1.0},
            {'params': salinity, 'lr_multiplier': multiplier}]


def self_test():
    """Check forward/backward/Adam parity and selective LR effect on CPU."""
    torch.manual_seed(15)
    original = nn.Conv2d(4,154,3)
    split = SalinityHead(original)
    faster = SalinityHead(original)
    x = torch.randn(1,4,8,8); target = torch.randn(1,154,6,6)
    opts = [torch.optim.Adam(original.parameters(),lr=1e-3,foreach=False),
            torch.optim.Adam(split.parameters(),lr=1e-3,foreach=False),
            torch.optim.Adam([{'params':[p for n,p in faster.named_parameters() if not n.startswith('salinity')], 'lr':1e-3},
                              {'params':faster.salinity_parameters(),'lr':1e-2}],foreach=False)]
    for step in range(5):
        assert torch.equal(original(x), split(x))
        models = [original,split,faster] if step==0 else [original,split]
        for model,opt in zip(models,opts):
            opt.zero_grad(); (model(x)-target).square().mean().backward(); opt.step()
        assert torch.equal(original.weight,torch.cat([split.before_weight,split.salinity_weight,split.after_weight]))
        assert torch.equal(original.bias,torch.cat([split.before_bias,split.salinity_bias,split.after_bias]))
        if step==0:
            assert torch.equal(split.before_weight,faster.before_weight)
            assert torch.equal(split.after_weight,faster.after_weight)
            assert not torch.equal(split.salinity_weight,faster.salinity_weight)
    print('CPU parity passed: identical forward and five Adam steps at 1x; only salinity updates change at 10x.')


if __name__ == '__main__':
    self_test()
