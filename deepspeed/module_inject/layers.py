import torch
from deepspeed import comm as dist
from torch import nn
from torch.nn import functional as F

from torch.nn.parameter import Parameter


class LinearAllreduce(nn.Module):
    def __init__(self, weight, bias=None, mp_group=None, async_collective=False):
        super(LinearAllreduce, self).__init__()
        self.weight = weight
        self.bias = bias
        self.mp_group = mp_group
        self.async_collective = async_collective

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.mp_group is not None:
            dist.all_reduce(output, group=self.mp_group, async_op=self.async_collective)
        if self.bias is not None:
            output += self.bias
        return output


class LinearLayer(nn.Module):
    def __init__(self, weight_shape=None, dtype=None, weight=None, bias=None, device=None):
        super(LinearLayer, self).__init__()
        if weight is not None:
            self.weight = weight
            self.bias = bias
        else:
            self.weight = Parameter(
                torch.empty(weight_shape,
                            dtype=dtype,
                            device=device))
            self.bias = Parameter(
                torch.empty(weight_shape[0],
                            dtype=dtype,
                            device=device))

    def forward(self, input):
        output = torch.matmul(input, self.weight.transpose(-1, -2))
        if self.bias is not None:
            output += self.bias
        return output


class Normalize(nn.Module):
    def __init__(self, dim, dtype=torch.float, eps=1e-5, device=None):
        super(Normalize, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=eps).to(dtype).to(device)
        self.weight = self.norm.weight
        self.bias = self.norm.bias

    def forward(self, input):
        return self.norm(input)


class EmbeddingLayer(nn.Module):
    def __init__(self, weight_shape, dtype=torch.float,device=None):
        super(EmbeddingLayer, self).__init__()
        self.weight = Parameter(
            torch.empty(weight_shape[0],
                        weight_shape[1],
                        dtype=dtype,
                        device=device))

    def forward(self, input):
        return F.embedding(input, self.weight)
