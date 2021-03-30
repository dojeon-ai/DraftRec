import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from common.train_utils import GELU


class PositionalEncoding(nn.Module):
    def __init__(self, n_position, d_hid):
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout):
        super(FeedForward, self).__init__()
        self.w1 = nn.Linear(embed_dim, embed_dim)
        self.w2 = nn.Linear(embed_dim, embed_dim)
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
        x = self.w2(x)
        x = self.dropout(x)
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

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

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
            eps = 1e9
            attn_logits = attn_logits.masked_fill(attn_mask.float().unsqueeze(1) == 1, -eps)
        attn_weights = torch.exp(F.log_softmax(attn_logits, dim=-1))
        attn_weights = self.dropout1(attn_weights)

        o = torch.matmul(attn_weights, v)
        o = o.permute(0, 2, 1, 3)
        o = o.reshape(-1, E)
        o = self.w_o(o).reshape(N, S, E)
        o = self.dropout2(o)

        return o, attn_weights


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_dim,
                                            num_heads=num_heads,
                                            dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.feed_forward = FeedForward(embed_dim=embed_dim,
                                        dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x, attn_mask):
        """
        :param x: type:(torch.Tensor) shape:(S:seq_len, N:batch_size, I:input_dim)
        :return x: type:(torch.Tensor) shape:(S, N, I)
        """
        y, attn_weights = self.attention(x, x, x, attn_mask=attn_mask)
        x = x+y
        x = self.norm1(x)
        x = self.feed_forward(x)
        x = x+y
        x = self.norm2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(Transformer, self).__init__()
        self.args = args
        self.device = device
        self.num_teams = 3  # [0:CLS, 1:BLUE, 2:RED]
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])
        self.num_lanes = len(categorical_ids['lane']) + 1 # TODO: remove +1 after re-processing
        self.num_version = len(categorical_ids['version'])
        self.num_position = 11  # [[CLS] + 10]

        self.embedding_dim = args.embedding_dim
        self.team_embedding = nn.Embedding(self.num_teams, self.embedding_dim)
        self.ban_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.user_embedding = nn.Embedding(self.num_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.lane_embedding = nn.Embedding(self.num_lanes, self.embedding_dim)
        self.version_embedding = nn.Embedding(self.num_version, self.embedding_dim)
        self.position_embedding = PositionalEncoding(self.num_position, self.embedding_dim)
        self.norm = nn.LayerNorm(self.embedding_dim, eps=1e-6)

        encoder = []
        for _ in range(self.args.num_hidden_layers):
            encoder.append(EncoderLayer(self.embedding_dim,
                                        self.args.num_heads,
                                        self.args.dropout))
        self.encoder = nn.ModuleList(encoder)
        self.policy_head = nn.Linear(self.embedding_dim, self.num_items)
        self.value_head = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        team_batch, ban_batch, user_batch, item_batch, lane_batch, version_batch, order_batch = x
        N, S = team_batch.shape

        cls = self.team_embedding(torch.zeros(N, 1).long().to(self.device))
        # version = self.version_embedding(version_batch.unsqueeze(1).long().to(self.device))
        cls = cls #+ version

        team = self.team_embedding(team_batch.long())
        #ban = self.ban_embedding(ban_batch.long())
        user = self.user_embedding(user_batch.long())
        item = self.item_embedding(item_batch.long())
        # lane = self.lane_embedding(lane_batch.long())
        #board = team + ban + user + item + lane
        board = item + user

        x = torch.cat([cls, board], 1)
        x = self.position_embedding(x)
        # x = self.norm(x)

        attn_mask = (torch.arange(S+1).to(self.device)[None, :] <= (order_batch+1)[:, None]).float()
        attn_mask = attn_mask.unsqueeze(2).matmul(attn_mask.unsqueeze(1)).bool()
        attn_mask = ~attn_mask

        for layer in self.encoder:
            x = layer(x, attn_mask)

        pi_logit = self.policy_head(x[torch.arange(N), order_batch.long()+1, :])
        pi_mask = torch.zeros_like(pi_logit).to(self.device)
        pi_mask.scatter_(1, ban_batch.long(), -np.inf)
        pi_logit = pi_logit + pi_mask
        pi = F.log_softmax(pi_logit, dim=1)  # log-probability is passed to NLL-Loss
        v = self.value_head(x[torch.arange(N), 0, :])   # logit is passed to BCE-Loss

        return pi, v
