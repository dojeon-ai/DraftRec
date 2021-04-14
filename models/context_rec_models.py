import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import EncoderLayer
from models.modules import GELU, PositionalEncoding, LayerNorm


class ContextRec(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(ContextRec, self).__init__()
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

        # team = self.team_embedding(team_ids)
        user = self.user_embedding(user_ids)
        item = self.item_embedding(item_ids)
        lane = self.lane_embedding(lane_ids)

        x = user + item + lane
        x = self.position_embedding(x)
        x = self.dropout(x)

        attn_mask = None
        # attn_mask = (torch.arange(S+1).to(self.device)[None, :] <= (torch.arange(S)+1).to(self.device)[:, None]).float()
        # attn_mask = attn_mask.unsqueeze(2).matmul(attn_mask.unsqueeze(1)).bool()
        # attn_mask = ~attn_mask
        # attn_mask = attn_mask.repeat(N, 1, 1)

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
