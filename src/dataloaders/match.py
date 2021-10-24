from .base import BaseDataloader
import torch
import tqdm
import numpy as np
import pickle
import os
from collections import defaultdict

class MatchDataloader(BaseDataloader):
    def __init__(self, args, mode, match_df, user_history_dict):
        super().__init__(args, mode, match_df, user_history_dict)

    @classmethod
    def code(cls):
        return 'match'

    def _get_dataset(self):
        dataset = MatchDataset(self.args,
                               self.match_df,
                               self.user_history_dict,
                               self.rng)
        return dataset

    
class MatchDataset(torch.utils.data.Dataset):
    def __init__(self, args, match_df, user_history_dict, rng):
        self.args = args
        self.match_df = match_df
        self.user_history_dict = user_history_dict
        self.rng = rng

        self.max_seq_len = args.max_seq_len
        self.num_turns = args.num_turns
        self.num_stats = args.num_stats
        self.index2match_and_turn = self.populate_indices()
        
    def populate_indices(self):
        index2match_and_turn = {}
        idx = 0
        # self.num_turns + 2 to train the model with full-match information
        for match_idx in range(len(self.match_df)):
            for turn in range(1, self.num_turns+2):
                index2match_and_turn[idx] = (match_idx, turn)
                idx += 1
        
        return index2match_and_turn
    
    def __len__(self):
        return len(self.index2match_and_turn)
    
    def __getitem__(self, index):    
        match_idx, turn = self.index2match_and_turn[index]
        match = self.match_df.iloc[match_idx]
        
        T = self.num_turns
        S = self.max_seq_len
        ST = self.num_stats
        
        # match-level input
        champions = np.zeros(T)
        roles = np.zeros(T)
        teams = np.zeros(T)
        bans = np.zeros(T)
        champion_masks = np.zeros(T)
        player_masks = np.zeros(T)
        
        # match-level output (if turn > self.num_turns, do not train the policy network)
        outcome, target_champion = None, None
        
        # user-level input
        user_champions = np.zeros((T, S))
        user_roles = np.zeros((T, S))
        user_outcomes = np.zeros((T, S))
        user_stats = np.zeros((T, S, ST))
        
        for t in range(1, self.num_turns+1):
            user_id, user_history_idx = match['User' + str(t)]
            user_history = self.user_history_dict[user_id]
            cur_user_history = user_history[user_history_idx]
            
            # match-level input
            champion = cur_user_history['champion']
            role = cur_user_history['role']
            team = cur_user_history['team']
            
            champions[t-1] = champion
            roles[t-1] = role
            teams[t-1] = team
            
            # match-level output
            # outcome and ban is computed in terms of the User1
            if t == 1:
                bans = np.array(cur_user_history['ban'])
                outcome = cur_user_history['win'] - self.args.num_special_tokens
            # target champion is computed in terms of the current user
            if t == turn:
                target_champion = champion
                
            # user-level input
            begin_idx = max(0, user_history_idx - S)
            end_idx = user_history_idx
            prev_user_histories = user_history[begin_idx:end_idx]
            pad_len = S - len(prev_user_histories)
            
            for s_idx, prev_user_history in enumerate(prev_user_histories):
                user_champions[t-1][s_idx+pad_len] = prev_user_history['champion']
                user_roles[t-1][s_idx+pad_len] = prev_user_history['role']
                user_outcomes[t-1][s_idx+pad_len] = prev_user_history['win']
                user_stats[t-1][s_idx+pad_len] = np.array(prev_user_history['stat'])
        
        # champions over than the current turn to the end turn must be masked
        champion_masks[turn-1:] = 1
        # players from the opponent team must be masked
        # after the last turn, train in terms of the first team
        if turn-1 == T:
            cur_user_team = teams[0]
            target_champion = self.args.PAD
        else:
            cur_user_team = teams[turn-1]
        player_masks = 1 - (teams == cur_user_team).astype(float)
        
        d = {
            # match
            'champions':torch.from_numpy(champions).long(), 
            'roles':torch.from_numpy(roles).long(),
            'teams':torch.from_numpy(teams).long(),
            'bans': torch.from_numpy(bans).long(),
            'champion_masks': torch.from_numpy(champion_masks),
            'player_masks': torch.from_numpy(player_masks),
            'turn': torch.LongTensor([turn]),
            'outcome': torch.LongTensor([outcome]),
            'target_champion': torch.LongTensor([target_champion]),
            # user
            'user_champions':torch.from_numpy(user_champions).long(),  
            'user_roles':torch.from_numpy(user_roles).long(),
            'user_outcomes':torch.from_numpy(user_outcomes).long(),            
            'user_stats':torch.FloatTensor(user_stats),            
        }
        
        return d