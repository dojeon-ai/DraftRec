from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
from .heads import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class HOI(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        num_turns = args.num_turns
        num_champions = args.num_champions
        hidden = args.hidden_units
        
        self.W = nn.Embedding(num_champions, 1, padding_idx=0)
        self.V = nn.Embedding(num_champions, hidden, padding_idx=0)
        self.apply(NormInitializer(hidden))
                
    @classmethod
    def code(cls):
        return 'hoi'
        
    def forward(self, batch):
        """
        B: batch_size
        C: number of champions
        
        [Params]
            champions: (B, T)
            teams: (B, T)
        [Output]
            un-normalized scores (i.e., logits)
            pi: (B, C)
            v: (B, 1)
        """
        B, T = batch['champions'].shape
        C = self.args.num_champions
        pi = torch.zeros((B, C), device=self.args.device).float()
        
        champions = batch['champions']
        champion_masks = batch['champion_masks']
        champions = ((1 - champion_masks) * champions).long()
        team_mask = (batch['teams'] - self.args.num_special_tokens).bool()
        blue_team = torch.masked_select(champions, ~team_mask).reshape(B, -1)
        purple_team = torch.masked_select(champions, team_mask).reshape(B, -1)
        
        blue_team_indiv_score = torch.sum(self.W(blue_team), 1)
        purple_team_indiv_score = torch.sum(self.W(purple_team), 1)
        indiv_score = blue_team_indiv_score - purple_team_indiv_score
        
        blue_team_vec = self.V(blue_team)
        blue_team_comb_score = blue_team_vec.matmul(blue_team_vec.permute(0,2,1)).reshape(B, -1)
        purple_team_vec = self.V(purple_team)
        purple_team_comb_score = purple_team_vec.matmul(purple_team_vec.permute(0,2,1)).reshape(B, -1)
        comb_score = torch.sum(blue_team_comb_score, 1) - torch.sum(purple_team_comb_score, 1)
        
        v = indiv_score + comb_score.unsqueeze(-1)
        
        return pi, v