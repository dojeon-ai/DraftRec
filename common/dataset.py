import torch
import wandb
import numpy as np
import tqdm
import random
from torch.utils.data import Dataset


# TODO: needs to be fixed for new-preprocessed dataset
class InteractionDataset(Dataset):
    def __init__(self, args, data, categorical_ids, is_train=True):
        self.args = args
        self.categorical_ids = categorical_ids
        if is_train:
            self.data = self._build_train_dataset(data)
        else:
            self.data = self._build_test_dataset(data)

    def _build_train_dataset(self, data):
        num_users, num_items = data.shape
        user_inputs, item_inputs, labels, scales = [], [], [], []
        for user_idx in tqdm.tqdm(range(num_users)):
            # since data is very large, directly storing float data is infeasible.
            row = data[user_idx]
            scale = np.sum(row)
            # positive instances
            positives = np.where(row > 0)[0]
            for item_idx in positives:
                user_inputs.append(user_idx)
                item_inputs.append(item_idx)
                labels.append(row[item_idx])
                scales.append(scale)
            # negative instances
            negatives = np.where(row == 0)[0]
            replace = False
            if len(negatives) < self.args.num_negatives:
                replace = True
            negatives = np.random.choice(negatives, size=self.args.num_negatives, replace=replace)
            for item_idx in negatives:
                user_inputs.append(user_idx)
                item_inputs.append(item_idx)
                labels.append(0)
                scales.append(scale)

        return np.column_stack((user_inputs, item_inputs, labels, scales))

    # noinspection PyMethodMayBeStatic
    def _build_test_dataset(self, data):
        num_users, num_items = data.shape
        user_inputs, item_inputs, labels, scales = [], [], [], []
        for user_idx in range(num_users):
            # since data is very large, directly storing float data is infeasible.
            row = data[user_idx]
            scale = np.sum(row)
            # do not include users without interaction
            if scale == 0:
                continue
            else:
                for item_idx in range(num_items):
                    user_inputs.append(user_idx)
                    item_inputs.append(item_idx)
                    labels.append(row[item_idx])
                    scales.append(scale)

        return np.column_stack((user_inputs, item_inputs, labels, scales))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.data[index][0]
        item = self.data[index][1]
        label = self.data[index][2]
        scale = self.data[index][3]
        # convert to the portion of interaction
        eps = 1e-5
        label = label / (scale+eps)

        return (user, item), label


class MatchTrainDataset(Dataset):
    def __init__(self, args, data, categorical_ids):
        self.args = args
        self.categorical_ids = categorical_ids
        self.seq_len = 11
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
            team_id, ban_id, user_id, item_id, lane_id = [self.CLS], [self.CLS], [self.CLS], [self.CLS], [self.CLS]
            win_label, item_label = [win], [self.PAD]
            for i, order in enumerate(pick_order):
                team_id.append(match['User' + str(order)+'_team'])
                ban_id.append(match['User' + str(order)+'_ban'])
                user_id.append(match['User'+str(order)+'_id'])
                item_id.append(match['User'+str(order)+'_champion'])
                lane_id.append(match['User' + str(order) + '_lane'])

                win_label.append(self.PAD)
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
        S = self.seq_len
        team_ids = self.data[index][:S].copy()
        ban_ids = self.data[index][S:2*S].copy()
        user_ids = self.data[index][2*S:3*S].copy()
        item_ids = self.data[index][3*S:4*S].copy()
        lane_ids = self.data[index][4*S:5*S].copy()
        win_labels = self.data[index][5*S:6*S].copy()
        item_labels = self.data[index][6*S:7*S].copy()

        team_prob = random.random()
        if team_prob < 0.5:
            team = self.categorical_ids['team']['BLUE']
            team_mask = (team_ids == team).astype(float)
        else:
            team = self.categorical_ids['team']['RED']
            team_mask = (team_ids == team).astype(float)
        team_mask[0] = 1.0  # [CLS]

        # blue-team cannot see the 'lane' and 'user' of the red-team and vice-versa
        lane_ids = np.where(team_mask == 1, lane_ids, self.UNK)
        user_ids = np.where(team_mask == 1, user_ids, self.UNK)

        for s in range(1, S):
            prob = random.random()
            if prob < self.args.mask_prob:
                item = item_ids[s]
                item_ids[s] = self.MASK

                # item_labels[s] = item
                # TODO: remove below formula if it does not work
                user = user_ids[s]
                if user == self.UNK:
                    item_labels[s] = self.PAD
                else:
                    item_labels[s] = item

        team_ids = torch.LongTensor(team_ids)
        ban_ids = torch.LongTensor(ban_ids)
        user_ids = torch.LongTensor(user_ids)
        item_ids = torch.LongTensor(item_ids)
        lane_ids = torch.LongTensor(lane_ids)
        win_labels = torch.FloatTensor(win_labels)
        item_labels = torch.LongTensor(item_labels)

        return (team_ids, ban_ids, user_ids, item_ids, lane_ids), (win_labels, item_labels)


class MatchTestDataset(MatchTrainDataset):
    def __init__(self, args, data, categorical_ids):
        super().__init__(args, data, categorical_ids)
        self.data = self._convert_dataset()

    def _convert_dataset(self):
        num_matches, _ = self.data.shape
        team_ids, ban_ids, user_ids, item_ids, lane_ids = [], [], [], [], []
        win_labels, item_labels = [], []
        for index in range(num_matches):
            S = self.seq_len
            team_id = self.data[index][:S].copy()
            ban_id = self.data[index][S:2 * S].copy()
            user_id = self.data[index][2 * S:3 * S].copy()
            item_id = self.data[index][3 * S:4 * S].copy()
            lane_id = self.data[index][4 * S:5 * S].copy()
            win_label = self.data[index][5 * S:6 * S].copy()
            item_label = self.data[index][6 * S:7 * S].copy()

            for s in range(1, S):
                team_ids.append(team_id)
                ban_ids.append(ban_id)
                team = team_id[s]
                team_mask = (team_id == team).astype(float)
                team_mask[0] = 1.0  # [CLS]

                lane_ids.append(np.where(team_mask == 1, lane_id, self.UNK).copy())
                user_ids.append(np.where(team_mask == 1, user_id, self.UNK).copy())

                cur_item_id = item_id.copy()
                cur_item_id[s:] = self.MASK
                item_ids.append(cur_item_id)

                item = item_id[s]
                cur_item_label = item_label.copy()
                cur_item_label[s] = item
                item_labels.append(cur_item_label)
                win_labels.append(win_label)

        return np.column_stack((team_ids, ban_ids, user_ids, item_ids, lane_ids, win_labels, item_labels))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        S = self.seq_len
        team_ids = torch.LongTensor(self.data[index][:S])
        ban_ids = torch.LongTensor(self.data[index][S:2*S])
        user_ids = torch.LongTensor(self.data[index][2*S:3*S])
        item_ids = torch.LongTensor(self.data[index][3*S:4*S])
        lane_ids = torch.LongTensor(self.data[index][4*S:5*S])
        win_labels = torch.FloatTensor(self.data[index][5*S:6*S])
        item_labels = torch.LongTensor(self.data[index][6*S:7*S])

        return (team_ids, ban_ids, user_ids, item_ids, lane_ids), (win_labels, item_labels)
