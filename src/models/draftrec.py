from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
from .heads import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class DraftRec(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        num_blocks = args.num_blocks
        hidden = args.hidden_units
        
        num_turns = args.num_turns
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
        
        self.user_positional_embedding = PositionalEncoding(max_seq_len, hidden)
        self.match_positional_embedding = PositionalEncoding(num_turns, hidden)

        self.dropout = nn.Dropout(p=args.dropout)
        self.user_blocks = nn.ModuleList(
            [TransformerBlock(args) for _ in range(num_blocks)])
        self.match_blocks = nn.ModuleList(
            [TransformerBlock(args) for _ in range(num_blocks)])
        
        self.user_output_norm = LayerNorm(hidden)
        self.match_output_norm = LayerNorm(hidden)
        
        # policy: (B, H) -> (B, C), value: (B, H) -> (B, 1)
        self.policy_head = DotProductPredictionHead(self.champion_embedding, hidden, num_champions)
        self.value_head = LinearPredictionHead(hidden, 1) 
        
        self.apply(NormInitializer(hidden))
        
    @classmethod
    def code(cls):
        return 'draftrec'
        
    def forward(self, batch):
        """
        B: batch_size
        T: num_turns (i.e., 10 in LOL)
        S: max_seq_len
        ST: num_stats
        C: number of champions
        
        [Params]
            champions: (B, T)
            roles: (B, T)
            teams: (B, T)
            bans: (B, T)
            champion_masks: (B, T)
            user_masks: (B, T)
            turn: (B)
            outcome: (B)
            target_champion: (B)
            
            user_champions: (B, T, S)
            user_roles: (B, T, S)
            user_outcomes (B, T, S)
            user_stats: (B, T, S, ST)
        
        [Output]
            un-normalized scores (i.e., logits)
            pi: (B, C)
            v: (B, 1)
        """
        B, T, S, ST = batch['user_stats'].shape
        H = self.args.hidden_units
        
        # user_embedding: (B, T, S, H)
        user_embedding = self.champion_embedding(batch['user_champions'])
        user_embedding += self.role_embedding(batch['user_roles'])
        user_embedding += self.outcome_embedding(batch['user_outcomes'])
        user_embedding += self.stat_embedding(batch['user_stats'])
        user_embedding = user_embedding.reshape(B*T, S, H)
        user_embedding = self.user_positional_embedding(user_embedding)
        user_embedding = self.dropout(user_embedding)
        
        # user_body
        for block in self.user_blocks:
            user_embedding = block(user_embedding, attn_mask=None)
        user_embedding = user_embedding.reshape(B, T, S, H)
        user_embedding = user_embedding[:, :, -1, :]
        user_embedding = self.user_output_norm(user_embedding)
        
        # match_embedding: (B, T, H)
        champion_masks = batch['champion_masks'].unsqueeze(-1)
        user_masks = batch['user_masks'].unsqueeze(-1)
        
        match_embedding = self.champion_embedding(batch['champions']) * (1-champion_masks)
        match_embedding += self.role_embedding(batch['roles'])
        match_embedding += self.team_embedding(batch['teams'])
        match_embedding += user_embedding * (1-user_masks)
        match_embedding = self.match_positional_embedding(match_embedding)
        match_embedding = self.dropout(match_embedding)
        
        # match_body: (B, T, H)
        for block in self.match_blocks:
            match_embedding = block(match_embedding, attn_mask=None)
        match_embedding = self.match_output_norm(match_embedding) 

        # policy_head: (B, T, H) -> (B, H) -> (B, C)
        turn_idx = copy.deepcopy(batch['turn'])
        # for the full-match data, fill the current_user_embedding with any value 
        # (will be neither trained nor evaluated)
        turn_idx[turn_idx > T] = 1
        turn_idx = turn_idx - 1

        # repeat index to gather 
        # (https://medium.com/analytics-vidhya/understanding-indexing-with-pytorch-gather-33717a84ebc4)
        turn_idx = turn_idx.repeat(1, H).unsqueeze(1)
        current_user_embedding = torch.gather(match_embedding, 1, turn_idx)
        current_user_embedding = current_user_embedding.squeeze(1)
        pi = self.policy_head(current_user_embedding)
        # mask the banned champions
        pi[torch.arange(B)[:, None], batch['bans']] = -1e9
        
        # value_head: (B, H) -> (B)
        # average pooling
        team_mask = (batch['teams'] - self.args.num_special_tokens).unsqueeze(-1)
        blue_team_embedding = torch.mean((1 - team_mask) * match_embedding, axis=1)
        purple_team_embedding = torch.mean(team_mask * match_embedding, axis=1)
        match_embedding = blue_team_embedding - purple_team_embedding
        v = self.value_head(match_embedding)
        
        return pi, v