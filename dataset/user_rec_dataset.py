import torch
import numpy as np
from torch.utils.data import Dataset


# TODO: create user-history dataset
class UserRecDataset(Dataset):
    def __init__(self, args, data, categorical_ids):
        super().__init__(args, data, categorical_ids)
        # we do not convert the dataset in prior due to the large memory requirements
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        match_history = self.data[index]
        is_train_data = False
        while not is_train_data:
            rand_match_idx = np.random.randint(0, high=len(match_history))
            is_train_data = (match_history[rand_match_idx]['data_type'] == 'train')
        pass
