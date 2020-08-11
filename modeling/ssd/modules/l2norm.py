"""L2Norm
Original author: YutaroOgawa
https://github.com/YutaroOgawa/pytorch_advanced/blob/master/2_objectdetection/utils/ssd_model.py
"""

import torch
import torch.nn as nn

class L2Norm(nn.Module):
    def __init__(self, in_channels=512, scale=20):
        super(L2Norm, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_channels))
        self.scale = scale
        self.reset_parameters()
        self.eps = 1e-10

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.scale)

    def forward(self, x):
        # For a 38×38 feature, calculate the root of the sum of squares over 512 channels.

        # norm: torch.Size([batch_num, 1, 38, 38])
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)

        # weights: torch.Size([batch_num, 512, 38, 38]) from torch.Size([512])
        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        out = weights * x
        
        return out

if __name__ == "__main__":
    weight = nn.Parameter(torch.Tensor(512))
    print(weight.size())

    x = torch.rand(10, 512, 38, 38)
    weights = weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
    print(weights.size())