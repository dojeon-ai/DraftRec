import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import EncoderLayer
from models.modules import GELU, PositionalEncoding, LayerNorm


class UserRec(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(UserRec, self).__init__()
        self.args = args
        self.device = device

        # 0 : PAD, 1 : MASK, 2 : CLS, 3 : UNK
        self.num_special_tokens = 4
        self.num_items = len(categorical_ids['champion'])
        self.num_lanes = len(categorical_ids['lane'])
        # 0: LOSE, 1 : WIN
        self.num_wins = 2 + self.num_special_tokens
        self.seq_len = self.args.max_seq_len

        self.embedding_dim = args.embedding_dim
        self.item_embedding = nn.Embedding(self.num_items, self.embedding_dim)
        self.lane_embedding = nn.Embedding(self.num_lanes, self.embedding_dim)
        self.win_embedding = nn.Embedding(self.num_wins, self.embedding_dim)
        self.stat_embedding = nn.Linear(self.args.num_stats, self.embedding_dim)
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

    def forward(self, x, embedding=False):
        """
        Outputs
            pi: torch.tensor: (N, S, C)
            v: torch.tensor: (N, S, 1)
        """
        ban_ids, item_ids, lane_ids, stat_ids, win_ids = x
        N, S = item_ids.shape
        E = self.embedding_dim

        item = self.item_embedding(item_ids)
        lane = self.lane_embedding(lane_ids)
        win = self.win_embedding(win_ids)
        stat = self.stat_embedding(stat_ids)

        if self.args.use_game_specific_info:
            x = item + lane + win
        else:
            x = item

        if self.args.use_stats:
            x = x + stat

        x = self.position_embedding(x)
        x = self.dropout(x)

        PAD = 0
        attn_mask = (item_ids == PAD).float()
        attn_mask = attn_mask.unsqueeze(-1).matmul(attn_mask.unsqueeze(1))

        for layer in self.encoder:
            x, _ = layer(x, attn_mask)
        x = self.norm(x)
        if embedding:
            return x[torch.arange(N, device=self.device), -1, :].squeeze(1)

        x = x.reshape(N * S, E)
        pi_logit = self.policy_head(x)
        item_embed = self.item_embedding(torch.arange(self.num_items, device=self.device))
        pi_logit = pi_logit.matmul(item_embed.T).reshape(N, S, -1)
        # banned champion should not be picked
        pi_mask = torch.zeros_like(pi_logit, device=self.device)
        pi_mask.scatter_(-1, ban_ids, -1e9)
        pi_logit = pi_logit + pi_mask
        pi = F.log_softmax(pi_logit, dim=-1)  # log-probability is passed to NLL-Loss
        v = self.value_head(x).reshape(N, S, -1)

        return pi, v


class SPOP(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(SPOP, self).__init__()
        self.args = args
        self.PAD = 0
        self.MASK = 1
        self.num_items = len(categorical_ids['champion'])
        self.num_lanes = len(categorical_ids['lane'])
        self.device = device
        self.null = nn.Linear(1, 1)

    def forward(self, x):
        ban_ids, item_ids, _, _, _ = x
        N, S = item_ids.shape
        C = self.num_items
        pi_logit = torch.zeros((N, S, C), device=self.device)
        for idx in range(N):
            item_id = item_ids[idx]
            pi_logit[idx] = torch.bincount(item_id[item_id != self.PAD][:-1], minlength=self.num_items)

        # banned champion should not be picked
        pi_mask = torch.zeros_like(pi_logit, device=self.device)
        pi_mask.scatter_(-1, ban_ids, -1e9)
        pi_logit = pi_logit + pi_mask
        pi = F.log_softmax(pi_logit, dim=-1)  # log-probability is passed to NLL-Loss
        v = torch.zeros((N, S, 1), device=self.device)
        return pi, v
