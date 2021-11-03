import math
import torch
import torch.nn as nn
from torch.autograd import Variable    
    
        
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0,max_seq_len).unsqueeze(1)
        base = torch.ones(d_model//2).fill_(10000)
        pow_term = torch.arange(0, d_model, 2) / torch.tensor(d_model,dtype=torch.float32)
        div_term = torch.pow(base,pow_term)

        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)

        # register_buffer: set as non-trainable layer but can check in state_dict
        self.register_buffer('positional_encoding', pe)

    def forward(self, x):
        return x + Variable(self.positional_encoding[:, :x.size(1)], requires_grad=False)

    
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
    
class LayerNorm(nn.Module):
    "Construct a layernorm module"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, sublayer):
        # Pre-LN is utilized as suggested in https://arxiv.org/pdf/2002.04745.pdf
        sub_output = sublayer(self.norm(x))
        if isinstance(sub_output, tuple):
            sub_output, rest = sub_output[0], sub_output[1:]
            output = x + self.dropout(sub_output)
            return (output, *rest)
        else:
            return x + self.dropout(sub_output)
