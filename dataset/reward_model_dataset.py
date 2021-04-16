import numpy as np
import torch
import tqdm
import random
from torch.utils.data import Dataset


class RewardModelDataset(Dataset):
    def __init__(self, args, data, categorical_ids):
        self.args = args
        self.categorical_ids = categorical_ids
        self.board_len = 11
        self.PAD = 0
        self.MASK = 1
        self.CLS = 2
        self.UNK = 3
        self.data = self._build_dataset(data)

    # noinspection PyMethodMayBeStatic
    def _build_dataset(self, data):
        num_matches, _ = data.shape
        team_ids, ban_ids, user_ids, item_ids, lane_ids = [], [], [], [], []
        win_labels, item_labels = [], []
        for match_idx in tqdm.tqdm(range(num_matches)):
            match = data.iloc[match_idx]
            win = float(match['win'] == 'Win')

            # order-of-pick: https://riot-api-libraries.readthedocs.io/en/latest/specifics.html
            pick_order = [1, 6, 7, 2, 3, 8, 9, 4, 5, 10]
            team_id, ban_id, user_id, item_id, lane_id = \
                [self.CLS], [self.CLS], [self.CLS], [self.CLS], [self.CLS]
            win_label, item_label = [win], [self.PAD]
            for i, order in enumerate(pick_order):
                team = match['User' + str(order)+'_team']
                team_id.append(team)
                ban_id.append(match['User' + str(order)+'_ban'])
                user_id.append(match['User'+str(order)+'_id'])
                item_id.append(match['User'+str(order)+'_champion'])
                lane_id.append(match['User' + str(order) + '_lane'])

                if team == self.categorical_ids['team']['BLUE']:
                    win_label.append(win - self.args.label_smooth)
                else:
                    win_label.append(1 - win + self.args.label_smooth)

                item_label.append(self.PAD)

            team_ids.append(team_id)
            ban_ids.append(ban_id)
            user_ids.append(user_id)
            item_ids.append(item_id)
            lane_ids.append(lane_id)

            win_labels.append(win_label)
            item_labels.append(item_label)

        return np.column_stack((team_ids, ban_ids, user_ids, item_ids, lane_ids, win_labels, item_labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # using the numpy array to provide faster indexing (5x faster than pandas dictionary?)
        B = self.board_len
        team_ids = self.data[index][:B].copy()
        ban_ids = self.data[index][B:2*B].copy()
        user_ids = self.data[index][2*B:3*B].copy()
        item_ids = self.data[index][3*B:4*B].copy()
        lane_ids = self.data[index][4*B:5*B].copy()
        win_labels = self.data[index][5*B:6*B].copy()
        item_labels = self.data[index][6*B:7*B].copy()

        team_prob = random.random()
        if team_prob < 0.5:
            team = self.categorical_ids['team']['BLUE']
            team_mask = (team_ids == team).astype(float)
        else:
            team = self.categorical_ids['team']['RED']
            team_mask = (team_ids == team).astype(float)
        team_mask[0] = self.CLS

        for b in range(1, B):
            item_mask_prob = random.random()
            if item_mask_prob < self.args.mask_item_prob:
                item_ids[b] = self.MASK

        # blue-team cannot see the 'user' of the red-team and vice-versa
        user_ids = np.where(team_mask == 1, user_ids, self.UNK)

        team_ids = torch.LongTensor(team_ids)
        ban_ids = torch.LongTensor(ban_ids)
        user_ids = torch.LongTensor(user_ids)
        item_ids = torch.LongTensor(item_ids)
        lane_ids = torch.LongTensor(lane_ids)
        win_labels = torch.FloatTensor(win_labels)
        item_labels = torch.LongTensor(item_labels)

        return (team_ids, ban_ids, user_ids, item_ids, lane_ids), (win_labels, item_labels)