from .base import BaseDataloader
import torch
import numpy as np

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
    
    def __len__(self):
        pass
    
    def __getitem__(self, index):    
        pass
    