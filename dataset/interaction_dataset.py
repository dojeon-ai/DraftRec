import numpy as np
import tqdm
from torch.utils.data import Dataset
from collections import defaultdict


# TODO: needs to be fixed for new-preprocessed dataset
class InteractionDataset(Dataset):
    def __init__(self, args, data, categorical_ids):
        self.args = args
        self.categorical_ids = categorical_ids
        self.data = self._build_dataset(data)
        self.pop_dict = self._build_pop_dict(data)

    def _build_dataset(self, data):
        num_users, num_items = data.shape
        user_ids, item_ids, item_labels, scale_labels = [], [], [], []
        for user_idx in tqdm.tqdm(range(num_users)):
            # since data is very large, directly storing float data is infeasible.
            row = data[user_idx]
            scale = np.sum(row)
            # positive instances
            positives = np.where(row > 0)[0]
            for item_idx in positives:
                user_ids.append(user_idx)
                item_ids.append(item_idx)
                item_labels.append(row[item_idx])
                scale_labels.append(scale)
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
                scale_labels.append(scale)

        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        item_labels = np.array(item_labels)
        scale_labels = np.array(scale_labels)

        return np.column_stack((user_ids, item_ids, item_labels, scale_labels))

    def _build_pop_dict(self, data):
        num_users, num_items = data.shape
        pop_dict = defaultdict(list)
        for user_idx in range(num_users):
            # since data is very large, directly storing float data is infeasible.
            row = data[user_idx]
            pop_dict[user_idx] = row.argsort()[::-1]

        return pop_dict

    # noinspection PyMethodMayBeStatic
    # def _build_test_dataset(self, data):
    #    num_users, num_items = data.shape
    #    user_inputs, item_inputs, labels, scales = [], [], [], []
    #    for user_idx in range(num_users):
    #        # since data is very large, directly storing float data is infeasible.
    #        row = data[user_idx]
    #        scale = np.sum(row)
    #        # do not include users without interaction
    #        if scale == 0:
    #            continue
    #        else:
    #            for item_idx in range(num_items):
    #                user_inputs.append(user_idx)
    #                item_inputs.append(item_idx)
    #                labels.append(row[item_idx])
    #                scales.append(scale)
    #
    #    return np.column_stack((user_inputs, item_inputs, labels, scales))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        user = self.data[index][0]
        item = self.data[index][1]
        item_label = self.data[index][2]
        scale_label = self.data[index][3]

        return (user, item), (item_label, scale_label)