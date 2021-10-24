from ..blocks.layers import GELU
import torch
import torch.nn as nn


class LinearPredictionHead(nn.Module):
    def __init__(self, d_model, d_out):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            GELU(),
            nn.Linear(d_model, d_out)
        )

    def forward(self, x, candidates=None):
        x = self.head(x)  # batch_size x d_out
        if candidates is not None:
            x = x.gather(1, candidates)  # batch_size x num_candidates
        return x


class DotProductPredictionHead(nn.Module):
    def __init__(self, embedding, d_model, d_out):
        super().__init__()
        self.embedding = embedding
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            GELU(),
        )
        self.bias = nn.Parameter(torch.zeros(1, d_out))

    def forward(self, x, candidates=None):
        x = self.head(x)  # B x H
        if candidates is not None: 
            emb = self.embedding(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  
            emb = self.embedding.weight[:]  
            logits = torch.matmul(x, emb.transpose(0, 1)) 
            logits += self.bias
        return logits
    