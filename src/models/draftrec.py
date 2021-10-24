from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
import torch
import torch.nn as nn
import torch.nn.functional as F


class DraftRec(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        num_blocks = args.num_blocks
        hidden = args.hidden_units
        
        num_champions = args.num_champions
        num_roles = args.num_roles
        num_teams = args.num_teams
        num_outcomes = args.num_outcomes
        num_stats = args.num_stats
        max_seq_len = args.max_seq_len
        
        self.champion_embedding = nn.Embedding(num_champions, hidden, padding_idx=0)
        self.role_embedding = nn.Embedding(num_roles, hidden, padding_idx=0)
        self.team_embedding = nn.Embedding(num_teams, hidden, padding_idx=0)
        self.outcome_embedding = nn.Embedding(num_outcomes, hidden, padding_idx=0)
        self.stat_embedding = nn.Linear(num_stats, hidden)
        
        self.positional_embedding = PositionalEncoding(max_seq_len, hidden)
        self.dropout = nn.Dropout(p=args.dropout)
        
    @classmethod
    def code(cls):
        return 'draftrec'
        
    def forward(self, batch):
        """
        B: batch_size
        T: num_turns (i.e., 10 in LOL)
        S: max_seq_len
        ST: num_stats
        
        [Params]
            champions: (B, T)
            roles: (B, T)
            teams: (B, T)
            bans: (B, T)
            champion_masks: (B, T)
            player_masks: (B, T)
            turn: (,)
            outcome: (,)
            target_champion: (,)
            
            user_champions: (B, T, S)
            user_roles: (B, T, S)
            user_outcomes (B, T, S)
            user_stats: (B, T, S, ST)
        """
        B, T, S, ST = batch['user_stats'].shape
        
        
        import pdb
        pdb.set_trace()
        
        #user_champions = batch['user_champions']
        
        
        #champion_embedding = 
        #role_embedding
        #team_embedding
        
        
        
        
        
        
        
        """
        (user_ban_ids, user_item_ids, user_lane_ids, user_stat_ids, user_win_ids) = x
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
        #attn_mask = (torch.arange(S+1).to(self.device)[None, :] <= (torch.arange(S)+1).to(self.device)[:, None]).float()
        #attn_mask = attn_mask.unsqueeze(2).matmul(attn_mask.unsqueeze(1)).bool()
        #attn_mask = ~attn_mask
        #attn_mask = attn_mask.repeat(N, 1, 1)
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
        """