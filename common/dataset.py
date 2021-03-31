import torch
import wandb
import numpy as np
import tqdm
from torch.utils.data import Dataset


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


class MatchDataset(Dataset):
    def __init__(self, args, data, categorical_ids):
        self.args = args
        self.categorical_ids = categorical_ids
        self.data = self._build_dataset(data)

    # noinspection PyMethodMayBeStatic
    def _build_dataset(self, data):
        num_matches, _ = data.shape
        team_inputs, ban_inputs, user_inputs, item_inputs, lane_inputs = [], [], [], [], []
        version_inputs, order_inputs = [], []
        user_labels, item_labels, win_labels = [], [], []
        for match_idx in tqdm.tqdm(range(num_matches)):
            match = data.iloc[match_idx]
            # order-of-pick: https://riot-api-libraries.readthedocs.io/en/latest/specifics.html
            pick_order = [1, 6, 7, 2, 3, 8, 9, 4, 5, 10]
            team_input, ban_input, user_input, item_input, lane_input = [], [], [], [], []
            for i, order in enumerate(pick_order):
                team_input.append((order-1)//5 + 1)
                ban_input.append(match['User' + str(order) + '_ban'])
                user_input.append(match['User'+str(order)+'_id'])
                item_input.append(match['User'+str(order)+'_champion'])
                lane_input.append(match['User' + str(order) + '_lane'])

            # TODO: after re-processing ,remove below function
            lane_input = self._append_mask_token_in_lane(lane_input)
            MASK = 1
            for i, order in enumerate(pick_order):
                # assign user & item
                team = team_input[i]
                user = user_input[i]
                item = match['User'+str(order)+'_champion']
                version = match['version']
                win = float(match['win'] == 'Win')

                # team & ban
                team_inputs.append(np.array(team_input))
                ban_inputs.append(np.array(ban_input))

                # blue-team cannot view the 'lane' and 'user' of the red-team and vice-versa
                team_mask = (np.array(team_input) == team).astype(float)
                lane_inputs.append(np.where((team_mask == 1), np.array(lane_input), MASK))
                user_inputs.append(np.where((team_mask == 1), np.array(user_input), MASK))

                # current-item should be masked
                cur_item_input = item_input.copy()
                cur_item_input[i:] = np.array([MASK] * (len(pick_order) - i))
                item_inputs.append(cur_item_input)

                # mask & labels
                order_inputs.append(i)
                version_inputs.append(version)
                user_labels.append(user)
                item_labels.append(item)
                win_labels.append(win)

        # numpy array provides faster indexing compared to the pandas dataframe
        return np.column_stack((team_inputs, ban_inputs, user_inputs, item_inputs, lane_inputs,
                                version_inputs, order_inputs,
                                user_labels, item_labels, win_labels))

    # TODO: needs to be removed
    def _append_mask_token_in_lane(self, lane_input):
        lane_input = np.array(lane_input) + 1
        lane_input = np.where(lane_input == 1, 0, lane_input)
        return lane_input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # using the numpy array to provide faster indexing (5x faster than dictionary?)
        team = self.data[index][:10]
        ban = self.data[index][10:20]
        user = self.data[index][20:30]
        item = self.data[index][30:40]
        lane = self.data[index][40:50]
        version = self.data[index][50]
        order = self.data[index][51]
        user_label = self.data[index][52]
        item_label = self.data[index][53]
        win_label = self.data[index][54]

        return (team, ban, user, item, lane, version, order), (user_label, item_label, win_label)
