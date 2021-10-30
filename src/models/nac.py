from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
from .heads import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class NeuralAC(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        num_turns = args.num_turns
        num_champions = args.num_champions
        hidden = args.hidden_units
        
        self.W = nn.Embedding(num_champions, 1, padding_idx=0)
        self.V = nn.Embedding(num_champions, hidden, padding_idx=0)
        self.P = nn.Embedding(num_champions, hidden, padding_idx=0)
        self.C = nn.Embedding(num_champions, hidden, padding_idx=0)
        
        self.f = nn.ReLU()
        
        self.W_coop = nn.Linear(hidden, hidden)
        self.W_comp = nn.Linear(hidden, hidden)
        
        self.apply(NormInitializer(hidden))
                
    @classmethod
    def code(cls):
        return 'nac'
    
    def get_interaction_score(self, V1, V2, mode='coop'):
        """
        [params] V1: (B, T_A, H)
        [params] V2: (B, T_B, H)
        """
        B, T, H = V1.shape
        
        if mode == 'coop':
            mask = torch.eye(T, device=self.args.device).unsqueeze(0).repeat(B, 1, 1)
        elif mode == 'comp':
            mask = torch.zeros((B, T, T), device=self.args.device)
        score = V1.matmul(V2.permute(0,2,1))
        score = self.f((1-mask) * score).reshape(B, -1)
        
        if mode == 'coop':
            attn = self.W_coop(V1).matmul(V2.permute(0,2,1))
        elif mode == 'comp':
            attn = self.W_comp(V1).matmul(V2.permute(0,2,1))
        attn = attn.masked_fill(mask == 1, -1e9).reshape(B, -1)
        attn = F.softmax(attn, dim=-1)

        return torch.sum(score * attn,1)
        
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
        
        # individual score
        blue_indiv_score = torch.sum(self.W(blue_team), 1).reshape(B)
        purple_indiv_score = torch.sum(self.W(purple_team), 1).reshape(B)
        
        # cooperation effect
        blue_v = self.V(blue_team)   
        purple_v = self.V(purple_team)
        blue_coop_score = self.get_interaction_score(blue_v, blue_v, mode='coop')
        purple_coop_score = self.get_interaction_score(purple_v, purple_v, mode='coop')
        
        # competition effect
        blue_p = self.P(blue_team)   
        blue_c = self.C(blue_team)   
        purple_p= self.P(purple_team)
        purple_c= self.C(purple_team)
        blue_comp_score = self.get_interaction_score(blue_p, purple_c, mode='comp')
        purple_comp_score = self.get_interaction_score(purple_p, blue_c, mode='coop')
        
        blue_score = blue_indiv_score + blue_coop_score + blue_comp_score
        purple_score = purple_indiv_score + purple_coop_score + purple_comp_score
        
        v = (blue_score - purple_score).unsqueeze(-1)
        
        return pi, v