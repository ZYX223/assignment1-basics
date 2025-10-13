import torch
from torch import nn
from torch.nn import init
from einops import rearrange, einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int ,device: None, dtype: None) :
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Weights of shape (out_features, in_features)
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        # Initialize weights using truncated normal
        std = (2 / (in_features + out_features)) ** 0.5
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x) :
        return einsum(x , self.weight , "... d_in, d_out d_in -> ... d_out")
    
class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, device: None, dtype: None) :
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        factory_kwargs = {'device': device, 'dtype': dtype}

        # Weights of shape (vocab_size, d_model)
        self.weight = nn.Parameter(torch.empty((vocab_size, d_model), **factory_kwargs))
        # Initialize weights using truncated normal
        std = 1
        init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)


    def forward(self, token_ids) :
        return self.weight[token_ids]