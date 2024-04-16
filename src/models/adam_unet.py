
import torch
import torch.nn as nn
import numpy as np
from itertools import tee

def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

class Conv_block(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,kernel_size = 3, num_layers=2, pad = "constant"):
        super().__init__()
        self.N_in = num_in
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad
        
        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_out,kernel_size))
        layers.append(torch.nn.BatchNorm2d(num_out))        
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers-1):
            layers.append(torch.nn.Conv2d(num_out,num_out,kernel_size))
            layers.append(torch.nn.BatchNorm2d(num_out))
            layers.append(torch.nn.ReLU())              

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        for l in self.layers:
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
        return fts

class AdamUNet(torch.nn.Module):
    def __init__(self,ch_width,n_out,wet,kernel_size = 3,pad = "constant"):
        super().__init__()
        self.N_in = ch_width[0]
        self.N_out = ch_width[-1]
        self.wet = wet
        self.N_pad = int((kernel_size-1)/2)
        self.pad = pad

        # going down
        layers = []
        for a,b in pairwise(ch_width):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.MaxPool2d(2))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        ch_width.reverse()
        for a,b in pairwise(ch_width[:-1]):
            layers.append(Conv_block(a,b,pad=pad))
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        layers.append(Conv_block(b,b,pad=pad))    
        layers.append(torch.nn.Conv2d(b,n_out,kernel_size))

        
        self.layers = nn.ModuleList(layers)
        self.num_steps = int(len(ch_width)-1)
        
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        temp = []
        for i in range(self.num_steps):
            temp.append(None)
        count = 0
        for l in self.layers:
            crop = fts.shape[2:]
            if isinstance(l,nn.Conv2d):
                fts = torch.nn.functional.pad(fts,(self.N_pad,self.N_pad,0,0),mode=self.pad)
                fts = torch.nn.functional.pad(fts,(0,0,self.N_pad,self.N_pad),mode="constant")
            fts= l(fts)
            if count < self.num_steps:
                if isinstance(l,Conv_block):
                    temp[count] = fts
                    count += 1
            elif count >= self.num_steps:
                if isinstance(l,nn.Upsample):
                    crop = np.array(fts.shape[2:])
                    shape = np.array(temp[int(2*self.num_steps-count-1)].shape[2:])
                    pads = (shape - crop)
                    pads = [pads[1]//2, pads[1]-pads[1]//2,
                            pads[0]//2, pads[0]-pads[0]//2]
                    fts = nn.functional.pad(fts,pads)
                    fts += temp[int(2*self.num_steps-count-1)]
                    count += 1
        return torch.mul(fts,self.wet)

if __name__ == "__main__":

    from torchinfo import summary
    model = AdamUNet([9,64,128,256,512],3,0) 
    summary(model)
