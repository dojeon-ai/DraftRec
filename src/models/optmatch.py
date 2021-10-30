from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import MultiHeadedAttention
from .heads import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class OptMatch(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        num_blocks = args.num_blocks
        hidden = args.hidden_units
        
        num_turns = args.num_turns
        num_champions = args.num_champions
        num_teams = args.num_teams
        num_outcomes = args.num_outcomes
        num_stats = args.num_stats
        
        self.syn_champion_embedding = nn.Embedding(num_champions, hidden, padding_idx=0)
        self.sup_champion_embedding = nn.Embedding(num_champions, hidden, padding_idx=0)
        self.stat_embedding = nn.Linear(num_stats, hidden)

        self.dropout = nn.Dropout(p=args.dropout)
        self.syn_team2vec = MultiHeadedAttention(h=1, d_model=2*hidden, dropout=args.dropout)
        self.sup_team2vec = MultiHeadedAttention(h=1, d_model=2*hidden, dropout=args.dropout)
        self.feat_team2vec = MultiHeadedAttention(h=1, d_model=2*hidden, dropout=args.dropout)
        
        self.syn_output = nn.Linear(2*hidden, 1)
        self.sup_output = nn.Linear(2*hidden, 1)
        self.feat_output = nn.Linear(2*hidden, 1)
        
        self.value_head = nn.Linear(3, 1)

        self.apply(NormInitializer(hidden))
        
    @classmethod
    def code(cls):
        return 'optmatch'
        
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
        C = self.args.num_champions
        pi = torch.zeros((B, C), device=self.args.device).float()
        
        # generate user embedding
        win_mask = ((batch['user_outcomes'] - self.args.num_special_tokens) > 0).float()
        syn_pick_user_embedding = torch.mean(self.syn_champion_embedding(batch['user_champions']), 2)
        syn_win_user_embedding = torch.mean(self.syn_champion_embedding((batch['user_champions'] * win_mask).long()), 2)
        syn_user_embedding = torch.cat((syn_pick_user_embedding, syn_win_user_embedding), -1)
        
        sup_pick_user_embedding = torch.mean(self.sup_champion_embedding(batch['user_champions']), 2)
        sup_win_user_embedding = torch.mean(self.sup_champion_embedding((batch['user_champions'] * win_mask).long()), 2)
        sup_user_embedding = torch.cat((sup_pick_user_embedding, sup_win_user_embedding), -1)
        
        feat_pick_user_embedding = torch.mean(self.stat_embedding(batch['user_stats']), 2)
        feat_win_user_embedding = torch.mean(self.stat_embedding((batch['user_stats'] * win_mask.unsqueeze(-1))), 2)
        feat_user_embedding = torch.cat((feat_pick_user_embedding, feat_win_user_embedding), -1)                                 
        
        # append current match information
        champion_masks = batch['champion_masks'].unsqueeze(-1)
        user_masks = batch['user_masks'].unsqueeze(-1)
        
        syn_user_embedding = syn_user_embedding * (1-user_masks)
        syn_user_embedding[:,:,:H] += self.syn_champion_embedding(batch['champions']) * (1-champion_masks)
        
        sup_user_embedding = sup_user_embedding * (1-user_masks)
        sup_user_embedding[:,:,:H] += self.sup_champion_embedding(batch['champions']) * (1-champion_masks)
            
        feat_user_embedding = feat_user_embedding * (1-user_masks)
        
        # separate to blue and purple team
        team_mask = (batch['teams'] - self.args.num_special_tokens).bool()
        team_mask = team_mask.unsqueeze(-1).repeat(1, 1, 2 * H)
        blue_syn_user_embedding = torch.masked_select(syn_user_embedding, ~team_mask).reshape(B, T//2, -1)
        blue_sup_user_embedding = torch.masked_select(sup_user_embedding, ~team_mask).reshape(B, T//2, -1)
        blue_feat_user_embedding = torch.masked_select(feat_user_embedding, ~team_mask).reshape(B, T//2, -1)
        
        purple_syn_user_embedding = torch.masked_select(syn_user_embedding, team_mask).reshape(B, T//2, -1)
        purple_sup_user_embedding = torch.masked_select(sup_user_embedding, team_mask).reshape(B, T//2, -1)
        purple_feat_user_embedding = torch.masked_select(feat_user_embedding, team_mask).reshape(B, T//2, -1)
        
        # forward team2vec layer
        blue_syn_embedding = torch.mean(self.syn_team2vec(blue_syn_user_embedding, 
                                                          blue_syn_user_embedding, 
                                                          blue_syn_user_embedding), 1)
        blue_sup_embedding = torch.mean(self.sup_team2vec(blue_sup_user_embedding, 
                                                          blue_sup_user_embedding, 
                                                          blue_sup_user_embedding), 1)
        blue_feat_embedding = torch.mean(self.feat_team2vec(blue_feat_user_embedding, 
                                                            blue_feat_user_embedding, 
                                                            blue_feat_user_embedding), 1)
        purple_syn_embedding = torch.mean(self.syn_team2vec(purple_syn_user_embedding, 
                                                            purple_syn_user_embedding, 
                                                            purple_syn_user_embedding), 1)
        purple_sup_embedding = torch.mean(self.sup_team2vec(purple_sup_user_embedding, 
                                                            purple_sup_user_embedding, 
                                                            purple_sup_user_embedding), 1)
        purple_feat_embedding = torch.mean(self.feat_team2vec(purple_feat_user_embedding, 
                                                              purple_feat_user_embedding, 
                                                              purple_feat_user_embedding), 1)
        
        syn_score = self.syn_output(blue_syn_embedding - purple_syn_embedding)
        sup_score = self.sup_output(blue_sup_embedding - purple_sup_embedding)
        feat_score = self.feat_output(blue_feat_embedding - purple_feat_embedding)
        
        v = self.value_head(torch.cat((syn_score, sup_score, feat_score), 1))
        
        return pi, v