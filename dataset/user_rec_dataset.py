import torch
import numpy as np
import random
from torch.utils.data import Dataset


# TODO: create user-history dataset
class UserRecDataset(Dataset):
    def __init__(self, args, data, categorical_ids):
        # we do not convert the dataset in prior due to the large memory requirements
        self.args = args
        self.data = data
        self.categorical_ids = categorical_ids
        self.PAD = 0
        self.MASK = 1
        self.CLS = 2
        self.UNK = 3
        self.num_special_tokens = 4

    def __len__(self):
        # identical to the number of users
        return len(self.data) - self.num_special_tokens

    def _mask_item(self, user_history, num_padding, item_ids, item_labels, win_ids, win_labels, win_mask_labels):
        if self.args.model_type == 'bert':
            for s in range(num_padding, len(user_history)):
                prob = random.random()
                if prob < self.args.mask_prob:
                    op = random.random()
                    # Mask item
                    if op < 0.5:
                        item = item_ids[s]
                        item_ids[s] = self.MASK
                        item_labels[s] = item
                    # Mask win
                    else:
                        win = win_ids[s]
                        win_ids[s] = self.MASK
                        win_labels[s] = (win - self.num_special_tokens)
                        win_mask_labels[s] = 1
        else:
            op = random.random()
            # Mask item
            if op < 0.5:
                item = item_ids[-1]
                item_ids[-1] = self.MASK
                item_labels[-1] = item
            # Mask win
            else:
                win = win_ids[-1]
                win_ids[-1] = self.MASK
                win_labels[-1] = (win - self.num_special_tokens)
                win_mask_labels[-1] = 1

        return item_ids, item_labels, win_ids, win_labels, win_mask_labels

    def __getitem__(self, index):
        user_idx = index + self.num_special_tokens
        match_history = self.data[user_idx]
        is_train_data = False
        # randomly selects the match_idx to train
        while not is_train_data:
            rand_match_idx = np.random.randint(0, high=len(match_history))
            is_train_data = (match_history[rand_match_idx]['data_type'] == 'train')
        history_start_idx = int(max(rand_match_idx + 1 - self.args.max_seq_len, 0))
        history_end_idx = int(rand_match_idx + 1)
        user_history = match_history[history_start_idx:history_end_idx]

        if not isinstance(user_history, list):
            user_history = [user_history]
        # append history to ids
        ban_ids, item_ids, lane_ids, stat_ids, win_ids = [], [], [], [], []
        # win_mask_labels are needed to
        win_labels, win_mask_labels, item_labels = [], [], []
        # append padding if history is less than the max_seq_len
        num_padding = self.args.max_seq_len - len(user_history)
        for _ in range(num_padding):
            # TODO: append team information?
            ban_ids.append([self.PAD] * 10)
            item_ids.append(self.PAD)
            lane_ids.append(self.PAD)
            stat_ids.append([self.PAD]*self.args.num_stats)
            win_ids.append(self.PAD)
            win_labels.append(self.PAD)
            win_mask_labels.append(0)
            item_labels.append(self.PAD)

        # history is already ordered in the time sequence
        for history in user_history:
            ban_ids.append(history['bans'])
            item_ids.append(history['item'])
            lane_ids.append(history['lane'])
            stat_ids.append(history['stat'])
            win_ids.append(history['win'])
            win_labels.append(self.PAD)
            win_mask_labels.append(0)
            item_labels.append(self.PAD)

        # Mask item
        item_ids, item_labels, win_ids, win_labels, win_mask_labels =\
            self._mask_item(user_history, num_padding, item_ids, item_labels, win_ids, win_labels, win_mask_labels)
        stat_ids[-1] = [self.PAD]*self.args.num_stats
        stat_ids = np.array(stat_ids)

        ban_ids = torch.LongTensor(ban_ids)
        item_ids = torch.LongTensor(item_ids)
        lane_ids = torch.LongTensor(lane_ids)
        stat_ids = torch.FloatTensor(stat_ids)
        win_ids = torch.LongTensor(win_ids)
        win_labels = torch.FloatTensor(win_labels)
        win_mask_labels = torch.FloatTensor(win_mask_labels)
        item_labels = torch.LongTensor(item_labels)

        return (ban_ids, item_ids, lane_ids, stat_ids, win_ids), (win_labels, win_mask_labels, item_labels)
