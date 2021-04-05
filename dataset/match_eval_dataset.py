import torch
import numpy as np
from dataset.context_rec_dataset import ContextRecDataset


class MatchEvalDataset(ContextRecDataset):
    def __init__(self, args, data, categorical_ids):
        super().__init__(args, data, categorical_ids)
        self.data = self._convert_dataset()

    def _convert_dataset(self):
        num_matches, _ = self.data.shape
        team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids = [], [], [], [], [], []
        win_labels, item_labels = [], []
        for index in range(num_matches):
            S = self.seq_len
            team_id = self.data[index][:S].copy()
            ban_id = self.data[index][S:2 * S].copy()
            user_id = self.data[index][2 * S:3 * S].copy()
            item_id = self.data[index][3 * S:4 * S].copy()
            lane_id = self.data[index][4 * S:5 * S].copy()
            history_id = self.data[index][5 * S:6 * S].copy()
            win_label = self.data[index][6 * S:7 * S].copy()
            item_label = self.data[index][7 * S:8 * S].copy()

            for s in range(1, S):
                team_ids.append(team_id)
                ban_ids.append(ban_id)
                history_ids.append(history_id)

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

        return np.column_stack((team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids, win_labels, item_labels))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        S = self.seq_len
        team_ids = torch.LongTensor(self.data[index][:S])
        ban_ids = torch.LongTensor(self.data[index][S:2*S])
        user_ids = torch.LongTensor(self.data[index][2*S:3*S])
        item_ids = torch.LongTensor(self.data[index][3*S:4*S])
        lane_ids = torch.LongTensor(self.data[index][4*S:5*S])
        history_ids = torch.LongTensor(self.data[index][5*S:6*S])
        win_labels = torch.FloatTensor(self.data[index][6*S:7*S])
        item_labels = torch.LongTensor(self.data[index][7*S:8*S])

        return (team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids), (win_labels, item_labels)