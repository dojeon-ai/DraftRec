import numpy as np
import tqdm
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