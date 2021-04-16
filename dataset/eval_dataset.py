import torch
import tqdm
import numpy as np


class EvalDataset():
    def __init__(self, args, match_data, user_history_data, categorical_ids):
        self.args = args
        self.categorical_ids = categorical_ids
        self.board_len = 11
        if hasattr(self.args, 'max_seq_len'):
           self.max_seq_len = self.args.max_seq_len
        else:
            self.max_seq_len = 1  # Faster fetch if not used
        self.PAD = 0
        self.MASK = 1
        self.CLS = 2
        self.UNK = 3
        self.num_special_tokens = 4
        self.match_data, self.user_history_data = self._build_dataset(match_data, user_history_data)

    # noinspection PyMethodMayBeStatic
    def _build_dataset(self, match_data, user_history_data):
        # convert to match-level data
        num_matches, _ = match_data.shape
        team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids = [], [], [], [], [], []
        win_labels, item_labels, user_labels = [], [], []
        for match_idx in range(num_matches):
            match = match_data.iloc[match_idx]
            win = float(match['win'] == 'Win')

            # order-of-pick: https://riot-api-libraries.readthedocs.io/en/latest/specifics.html
            pick_order = [1, 6, 7, 2, 3, 8, 9, 4, 5, 10]
            team_id, ban_id, user_id, item_id, lane_id, history_id = \
                [self.CLS], [self.CLS], [self.CLS], [self.CLS], [self.CLS], [self.CLS]
            win_label, item_label, user_label = [win], [self.PAD], [self.PAD]
            for i, order in enumerate(pick_order):
                team_id.append(match['User' + str(order)+'_team'])
                ban_id.append(match['User' + str(order)+'_ban'])
                user_id.append(match['User'+str(order)+'_id'])
                item_id.append(match['User'+str(order)+'_champion'])
                lane_id.append(match['User' + str(order) + '_lane'])
                history_id.append(match['User' + str(order) + '_history'])

                #win_label.append(self.PAD)
                win_label.append(win)
                item_label.append(self.PAD)
                user_label.append(self.PAD)

            team_ids.append(team_id)
            ban_ids.append(ban_id)
            user_ids.append(user_id)
            item_ids.append(item_id)
            lane_ids.append(lane_id)
            history_ids.append(history_id)

            win_labels.append(win_label)
            item_labels.append(item_label)
            user_labels.append(user_label)

        matches = np.column_stack((team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids,
                                   win_labels, item_labels, user_labels))
        # convert to user-level data
        team_ids, ban_ids, user_ids, item_ids, lane_ids, history_ids = [], [], [], [], [], []
        win_labels, item_labels, user_labels = [], [], []
        history_eval_data = []
        for match in tqdm.tqdm(matches):
            B = self.board_len
            team_id = match[:B].copy()
            ban_id = match[B:2 * B].copy()
            user_id = match[2 * B:3 * B].copy()
            item_id = match[3 * B:4 * B].copy()
            lane_id = match[4 * B:5 * B].copy()
            history_id = match[5 * B:6 * B].copy()
            win_label = match[6 * B:7 * B].copy()
            item_label = match[7 * B:8 * B].copy()
            user_label = match[8 * B:9 * B].copy()

            user_histories = []
            mask_histories = []
            unk_mask_histories = []
            for b in range(1, B):
                user_idx = user_id[b]
                history_idx = history_id[b]
                lane = lane_id[b]
                item = item_id[b]
                win = win_label[0]

                unk_match_history = self._get_recent_match_history_of_unk(ban_id[1:], lane, item, win)
                unk_history = self._build_recent_history(unk_match_history)
                if user_idx != self.UNK:
                    match_history = self._get_recent_match_history(user_history_data, user_idx, history_idx)
                    user_history = self._build_recent_history(match_history)
                    mask_history = self._build_recent_history(match_history, mask=True)
                else:
                    # ban_id[1:]: exclude [CLS] token
                    user_history = unk_history
                    mask_history = self._build_recent_history(unk_match_history, mask=True)
                user_histories.append(user_history)
                unk_histories.append(unk_history)
                mask_histories.append(mask_history)

            for b in range(1, B):
                user = user_id[b]
                #if user == self.UNK:
                #    continue
                team_ids.append(team_id)
                ban_ids.append(ban_id)
                history_ids.append(history_id)

                team = team_id[b]
                team_mask = (team_id == team).astype(float)
                team_mask[0] = 1.0  # [CLS]

                # lane_ids.append(np.where(team_mask == 1, lane_id, self.UNK).copy())
                lane_ids.append(lane_id)
                user_ids.append(np.where(team_mask == 1, user_id, self.UNK).copy())

                cur_item_id = item_id.copy()
                cur_item_id[b:] = self.MASK
                item_ids.append(cur_item_id)

                item = item_id[b]
                cur_item_label = item_label.copy()
                cur_item_label[b] = item
                item_labels.append(cur_item_label)
                win_labels.append(win_label)
                cur_user_label = user_label.copy()
                cur_user_label[b] = user
                user_labels.append(cur_user_label)

                # opponent user's histories should be masked
                cur_user_histories = user_histories.copy()
                cur_user_histories[b-1] = mask_histories[b-1]
                for c in range(0, B-1):
                    cur_user_team_mask = team_mask[c]
                    if c > b:
                        if cur_user_team_mask == 0:
                            cur_user_histories[c] = unk_mask_histories[c]
                        else:
                            cur_user_histories[c] = mask_histories[c]
                cur_user_histories = [torch.stack(feature) for feature in list(map(list, zip(*cur_user_histories)))]
                history_eval_data.append(cur_user_histories)

        # len(match_eval_data) = num_matches * 10
        team_ids = np.array(team_ids)
        ban_ids = np.array(ban_ids)
        user_ids = np.array(user_ids)
        item_ids = np.array(item_ids)
        lane_ids = np.array(lane_ids)
        win_labels = np.array(win_labels)
        item_labels = np.array(item_labels)
        user_labels = np.array(user_labels)
        match_eval_data = np.column_stack((team_ids,
                                           ban_ids,
                                           user_ids,
                                           item_ids,
                                           lane_ids,
                                           win_labels,
                                           item_labels,
                                           user_labels))
        # len(num_matches) = num_matches * 10
        # len(num_matches[0]) = 10  # number of users in each match
        return match_eval_data, history_eval_data

    def _get_recent_match_history(self, user_history_data, user_idx, history_idx):
        match_history = user_history_data[user_idx]
        history_start_idx, history_end_idx = -1, -1
        for idx, match in enumerate(match_history):
            if match['id'] == history_idx:
                history_end_idx = idx + 1
                break
        if history_end_idx == -1:
            raise IndexError
        history_start_idx = int(max(history_end_idx - self.max_seq_len, 0))
        match_history = match_history[history_start_idx:history_end_idx]
        if not isinstance(match_history, list):
            match_history = [match_history]
        return match_history

    def _get_recent_match_history_of_unk(self, bans, lane, item, win):
        history = {}
        history['bans'] = bans
        history['lane'] = lane
        # Though we are converting item and win to MASK, assign the correct value to avoid confusion
        history['item'] = item
        history['win'] = win + self.num_special_tokens
        match_history = [history]
        return match_history

    def _build_recent_history(self, match_history, mask=False):
        # append history to ids
        ban_ids, item_ids, lane_ids, win_ids = [], [], [], []
        # win_mask_labels are needed to
        win_labels, win_mask_labels, item_labels = [], [], []
        # append padding if history is less than the max_seq_len
        num_padding = self.max_seq_len - len(match_history)
        for _ in range(num_padding):
            ban_ids.append([self.PAD] * 10)
            item_ids.append(self.PAD)
            lane_ids.append(self.PAD)
            win_ids.append(self.PAD)
            win_labels.append(self.PAD)
            win_mask_labels.append(0)
            item_labels.append(self.PAD)

        # history is already ordered in the time sequence
        for history in match_history:
            ban_ids.append(history['bans'])
            item_ids.append(history['item'])
            lane_ids.append(history['lane'])
            win_ids.append(history['win'])
            win_labels.append(self.PAD)
            win_mask_labels.append(0)
            item_labels.append(self.PAD)

        # only mask the last-element
        item = item_ids[-1]
        win_ids[-1] = self.MASK
        win_labels[-1] = (history['win'] - self.num_special_tokens)
        win_mask_labels[-1] = 1  # no effect
        if mask:
            item_ids[-1] = self.MASK
            item_labels[-1] = item

        ban_ids = torch.LongTensor(ban_ids)
        item_ids = torch.LongTensor(item_ids)
        lane_ids = torch.LongTensor(lane_ids)
        win_ids = torch.LongTensor(win_ids)
        win_labels = torch.FloatTensor(win_labels)
        win_mask_labels = torch.FloatTensor(win_mask_labels)
        item_labels = torch.LongTensor(item_labels)

        return (ban_ids, item_ids, lane_ids, win_ids, win_labels, win_mask_labels, item_labels)

    def __len__(self):
        return len(self.match_data)

    def __getitem__(self, index):
        B = self.board_len
        team_ids = torch.LongTensor(self.match_data[index][:B])
        ban_ids = torch.LongTensor(self.match_data[index][B:2*B])
        user_ids = torch.LongTensor(self.match_data[index][2*B:3*B])
        item_ids = torch.LongTensor(self.match_data[index][3*B:4*B])
        lane_ids = torch.LongTensor(self.match_data[index][4*B:5*B])
        win_labels = torch.FloatTensor(self.match_data[index][5*B:6*B])
        item_labels = torch.LongTensor(self.match_data[index][6*B:7*B])
        user_labels = torch.LongTensor(self.match_data[index][7*B:8*B])

        match_x = (team_ids, ban_ids, user_ids, item_ids, lane_ids)
        match_y = (win_labels, item_labels, user_labels)

        ban_ids = torch.LongTensor(self.user_history_data[index][0])
        item_ids = torch.LongTensor(self.user_history_data[index][1])
        lane_ids = torch.LongTensor(self.user_history_data[index][2])
        win_ids = torch.LongTensor(self.user_history_data[index][3])
        win_labels = torch.FloatTensor(self.user_history_data[index][4])
        win_mask_labels = torch.FloatTensor(self.user_history_data[index][5])
        item_labels = torch.LongTensor(self.user_history_data[index][6])

        user_history_x = (ban_ids, item_ids, lane_ids, win_ids)
        user_history_y = (win_labels, win_mask_labels, item_labels)

        return (match_x, match_y), (user_history_x, user_history_y)