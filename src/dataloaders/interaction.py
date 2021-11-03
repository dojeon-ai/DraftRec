from .base import BaseDataloader
import torch
import tqdm
import numpy as np
import pickle
import os
from collections import defaultdict
import copy

class InteractionDataloader(BaseDataloader):
    def __init__(self, args, mode, match_df, user_history_dict):
        super().__init__(args, mode, match_df, user_history_dict)

    @classmethod
    def code(cls):
        return 'interaction'

    def _get_dataset(self):
        dataset = InteractionDataset(self.args,
                                     self.match_df,
                                     self.user_history_dict,
                                     self.rng)
        return dataset

    
class InteractionDataset(torch.utils.data.Dataset):
    def __init__(self, args, match_df, user_history_dict, rng):
        self.args = args
        self.match_df = match_df
        self.user_history_dict = user_history_dict
        self.rng = rng

        self.interaction_matrix = self._build_interaction_matrix()
        self.data = self._build_dataset(self.interaction_matrix)
        self.pop_dict = self._build_pop_dict(self.interaction_matrix)
    
    def _build_interaction_matrix(self):
        num_participants = self.args.num_turns
        num_users = len(self.user_history_dict.user_id_to_array_idx.keys())

        
        #TODO: change this with arguements...?!
        #self.args.num_champions
        num_champions = self.args.num_champions
        interaction_matrix = np.zeros((num_users, num_champions), dtype=int)

        for i in tqdm.tqdm(range(len(self.match_df))):
            match = self.match_df.iloc[i]
            for p_idx in range(num_participants):
                (user_idx, history_idx) = match['User'+str(p_idx+1)]

                champion_idx = int(self.user_history_dict.get_value(user_idx, history_idx, 'champion').item())
                user_idx = self.user_history_dict.user_id_to_array_idx[user_idx]
                interaction_matrix[user_idx][champion_idx] += 1

        return interaction_matrix
    
    def _build_dataset(self, interaction_matrix):
        num_users, num_items = interaction_matrix.shape
        user_ids, item_ids, item_labels, scale_labels = [], [], [], []
        for user_idx in tqdm.tqdm(range(num_users)):
            # since data is very large, directly storing float data is infeasible.
            row = interaction_matrix[user_idx]
            scale = np.sum(row)
            # positive instances
            positives = np.where(row > 0)[0]
            for item_idx in positives:
                user_ids.append(user_idx)
                item_ids.append(item_idx)
                if self.args.model_type == 'dmf':
                    eps = 1e-5
                    item_labels.append(np.float64(row[item_idx])/(scale + eps))
                else:
                    item_labels.append(np.sign(row[item_idx], dtype=np.float64))

            # negative instances
            negatives = np.where(row == 0)[0]
            replace = False
            if len(negatives) < self.args.num_negatives:
                replace = True
            negatives = np.random.choice(negatives, size=self.args.num_negatives, replace=replace)
            for item_idx in negatives:
                user_ids.append(user_idx)
                item_ids.append(item_idx)
                item_labels.append(0)

        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        item_labels = np.array(item_labels)

        return np.column_stack((user_ids, item_ids, item_labels))

    def _build_pop_dict(self, interaction_matrix):
        num_users, num_items = interaction_matrix.shape
        pop_dict = defaultdict(list)
        for user_idx in range(num_users):
            # since data is very large, directly storing float data is infeasible.
            row = interaction_matrix[user_idx]
            pop_dict[user_idx] = row.argsort()[::-1]

        return pop_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.data[index][0]
        item = self.data[index][1]
        item_label = self.data[index][2]

        d = {
            'user_idx':torch.tensor(user).long(),  
            'champion_idx':torch.tensor(item).long(),
            'champion_label':torch.tensor(item_label).float(),              
        }
        
        return d