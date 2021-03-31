import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.train_utils import GELU, PositionalEncoding, LayerNorm


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(embed_dim, embed_dim*4)
        self.w2 = nn.Linear(embed_dim*4, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        """
        :param x: type:(torch.Tensor) shape:(N:batch_size, S:seq_len, E:embed_dim)
        :return x: type:(torch.Tensor) shape:(N, S, E:input_dim)
        """
        N, S, E = x.shape
        x = x.reshape(N*S, E)
        x = self.w1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.w2(x)
        x = x.reshape(N, S, E)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = embed_dim // num_heads

        self.w_qs = nn.Linear(embed_dim, embed_dim)
        self.w_ks = nn.Linear(embed_dim, embed_dim)
        self.w_vs = nn.Linear(embed_dim, embed_dim)
        self.w_o = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None):
        """
        :param q: (N, S, E)
        :param k: (N, S, E)
        :param v: (N, S, E)
        :param attn_mask: (N, S, S) (torch.bool)
        :return:
        """
        N, S, E = q.shape
        # Pass through pre-attention projection

        q = self.w_qs(q.reshape(-1, E)).reshape(N, S, self.num_heads, self.d_k)
        k = self.w_ks(k.reshape(-1, E)).reshape(N, S, self.num_heads, self.d_k)
        v = self.w_vs(v.reshape(-1, E)).reshape(N, S, self.num_heads, self.d_k)

        # Transpose to not attend different heads
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        # Attention
        attn_logits = torch.matmul(q, k.permute(0, 1, 3, 2)) / np.sqrt(self.d_k)  # (N, H, S, S)
        if attn_mask is not None:
            attn_logits = attn_logits.masked_fill(attn_mask.float().unsqueeze(1) == 1, -1e9)
        attn_weights = torch.exp(F.log_softmax(attn_logits, dim=-1))
        attn_weights = self.dropout(attn_weights)

        o = torch.matmul(attn_weights, v)
        o = o.permute(0, 2, 1, 3)
        o = o.reshape(N, S, E)
        o = self.w_o(o)

        return o, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(embed_dim, eps=1e-6)
        self.attention = MultiHeadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = LayerNorm(embed_dim, eps=1e-6)
        self.feed_forward = FeedForward(embed_dim=embed_dim,
                                        dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        """
        :param x: type:(torch.Tensor) shape:(S:seq_len, N:batch_size, I:input_dim)
        :return x: type:(torch.Tensor) shape:(S, N, I)
        """
        y = self.norm1(x)
        y, attn_weights = self.attention(y, y, y, attn_mask=attn_mask)
        x = x + self.dropout1(y)

        y = self.norm2(x)
        y = self.feed_forward(y)
        x = x + self.dropout2(y)
        return x


class Transformer(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(Transformer, self).__init__()
        self.args = args
        self.device = device
        self.num_teams = len(categorical_ids['team'])
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])
        self.num_lanes = len(categorical_ids['lane'])
        self.seq_len = 11

        self.embedding_dim = args.embedding_dim
        self.team_embedding = nn.Embedding(self.num_teams, self.embedding_dim)
        self.ban_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.lane_embedding = nn.Embedding(self.num_lanes, self.embedding_dim)
        self.position_embedding = PositionalEncoding(self.seq_len, self.embedding_dim)
        self.dropout = nn.Dropout(self.args.dropout)

        encoder = []
        for _ in range(self.args.num_hidden_layers):
            encoder.append(EncoderLayer(self.embedding_dim,
                                        self.args.num_heads,
                                        self.args.dropout))
        self.encoder = nn.ModuleList(encoder)
        self.norm = LayerNorm(self.embedding_dim, eps=1e-6)
        self.policy_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                         GELU())
        self.value_head = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        """
        Outputs
            pi: torch.tensor: (N, S, C)
            v: torch.tensor: (N, S, 1)
        """
        team_ids, ban_ids, user_ids, item_ids, lane_ids = x
        N, S = team_ids.shape
        E = self.embedding_dim

        team = self.team_embedding(team_ids)
        ban = self.ban_embedding(ban_ids)
        user = self.user_embedding(user_ids)
        item = self.item_embedding(item_ids)
        lane = self.lane_embedding(lane_ids)

        x = user + item + lane
        x = self.position_embedding(x)
        x = self.dropout(x)

        attn_mask = None
        """
        attn_mask = (torch.arange(S+1).to(self.device)[None, :] <= (torch.arange(S)+1).to(self.device)[:, None]).float()
        attn_mask = attn_mask.unsqueeze(2).matmul(attn_mask.unsqueeze(1)).bool()
        attn_mask = ~attn_mask
        attn_mask = attn_mask.repeat(N, 1, 1)
        """
        for layer in self.encoder:
            x = layer(x, attn_mask)
        x = self.norm(x)
        x = x.reshape(N*S, E)

        pi_logit = self.policy_head(x)
        item_embed = self.item_embedding(torch.arange(self.num_items, device=self.device))
        pi_logit = pi_logit.matmul(item_embed.T).reshape(N, S, -1)
        # banned champion should not be picked
        pi_mask = torch.zeros_like(pi_logit, device=self.device)
        pi_mask.scatter_(-1, ban_ids.repeat(1,1,S).reshape(N,S,S), -1e9)
        pi_logit = pi_logit + pi_mask
        pi = F.log_softmax(pi_logit, dim=-1)  # log-probability is passed to NLL-Loss
        v = self.value_head(x).reshape(N, S, -1)

        return pi, v
