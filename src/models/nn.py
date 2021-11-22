from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
from .heads import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class NN(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        num_turns = args.num_turns
        num_champions = args.num_champions
        hidden = args.hidden_units
        self.value_head = nn.Sequential(
            nn.Linear(num_champions, hidden),
            GELU(),
            nn.Linear(hidden, 1)
        )
                
    @classmethod
    def code(cls):
        return 'nn'
        
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
        
        x = torch.zeros((B, C), device=self.args.device).float()
        x[torch.arange(B)[:, None], blue_team] = 1
        x[torch.arange(B)[:, None], purple_team] = -1
        
        v = self.value_head(x)
        
        return pi, v