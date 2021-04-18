import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import EncoderLayer
from models.modules import GELU, PositionalEncoding, LayerNorm
from models.user_rec_models import UserRec


class DraftRec(nn.Module):
    def __init__(self, args, categorical_ids, device):
        super(DraftRec, self).__init__()
        self.args = args
        self.device = device
        self.num_teams = len(categorical_ids['team'])
        self.num_users = len(categorical_ids['user'])
        self.num_items = len(categorical_ids['champion'])
        self.num_lanes = len(categorical_ids['lane'])
        self.board_len = 11

        self.embedding_dim = args.embedding_dim
        self.cls_embedding = nn.Embedding(1, self.embedding_dim)
        self.team_embedding = nn.Embedding(2, self.embedding_dim)
        self.position_embedding = PositionalEncoding(self.board_len, self.embedding_dim)
        self.dropout = nn.Dropout(self.args.dropout)

        self.embedder = UserRec(args, categorical_ids, device)
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
        Inputs
            user_ban_ids: (N, B, S, # of bans)
            user_item_ids: (N, B, S)
            user_lane_ids: (N, B, S)
            user_win_ids: (N, B, S)
        Outputs
            pi: torch.tensor: (N, B, C)
            v: torch.tensor: (N, B, 1)
        """
        (user_ban_ids, user_item_ids, user_lane_ids, user_win_ids) = x
        N, B, S = user_item_ids.shape
        E = self.embedding_dim
        x = [feature.reshape(N*B, *feature.shape[2:]) for feature in x]
        embedding = self.embedder(x, embedding=True)  # [N*B, E]
        embedding = embedding.reshape(N, B, -1)
        if self.args.use_team_info:
            team_ids = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0, 0, 1], device=self.device).long().repeat(N).reshape(N, -1)
            embedding += self.team_embedding(team_ids)

        cls = torch.zeros((N, 1), device=self.device).long()
        cls = self.cls_embedding(cls)
        x = torch.cat((cls, embedding), 1)
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
            x, _ = layer(x, attn_mask)
        x = self.norm(x)
        x = x.reshape(N*(B+1), E)

        pi_logit = self.policy_head(x)
        item_embed = self.embedder.item_embedding(torch.arange(self.num_items, device=self.device))
        pi_logit = pi_logit.matmul(item_embed.T).reshape(N, (B+1), -1)
        # banned champion should not be picked
        pi_mask = torch.zeros_like(pi_logit, device=self.device)
        match_ban_ids = user_ban_ids[torch.arange(N, device=self.device), 0, -1, :]
        match_ban_ids = torch.cat((torch.zeros(N, 1, device=self.device).long(), match_ban_ids), 1)
        pi_mask.scatter_(-1, match_ban_ids.repeat(1,1,(B+1)).reshape(N,(B+1),(B+1)), -1e9)
        pi_logit = pi_logit + pi_mask
        pi = F.log_softmax(pi_logit, dim=-1)  # log-probability is passed to NLL-Loss
        v = self.value_head(x).reshape(N, (B+1), -1)

        return pi, v
