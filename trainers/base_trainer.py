import tqdm
import wandb
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from common.metrics import *
from models.interaction_models import NMF, DMF

class BaseTrainer():
    def __init__(self, args, train_loader, val_loader, test_loader, categorical_ids, device):
        self.args = args
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.categorical_ids = categorical_ids
        self.device = device

        self.num_users = len(self.categorical_ids['user'])
        self.num_items = len(self.categorical_ids['champion'])

    def _initialize_model(self):
        pass

    def _initialize_criterion(self):
        pass

    def _initialize_optimizer(self):
        pass

    def _save_model(self, model_name, epoch):
        model_path = wandb.run.dir + '/' + str(model_name)
        torch.save({"state_dict": self.model.state_dict(),
                    "epoch": epoch}, model_path)

    def train(self):
        pass

    def evaluate(self, loader):
        self.model.eval()
        # initialize metrics
        summary = {}
        for k in self.args.k_list:
            summary['HR@'+str(k)] = 0.0
            summary['NDCG@'+str(k)] = 0.0
        summary['MRR'] = 0.0
        for metric in ['ACC', 'MAE', 'MSE', 'LAST_ACC', 'LAST_MAE', 'LAST_MSE']:
            summary[metric] = 0.0

        total_eval_num = 0
        for match_batch, user_history_batch in tqdm.tqdm(loader):
            match_x_batch, match_y_batch = match_batch
            match_x_batch = [feature.to(self.device) for feature in match_x_batch]
            match_y_batch = [feature.to(self.device) for feature in match_y_batch]
            user_history_x_batch, user_history_y_batch = user_history_batch
            user_history_x_batch = [feature.to(self.device) for feature in user_history_x_batch]
            user_history_y_batch = [feature.to(self.device) for feature in user_history_y_batch]
            """
            N: batch_size 
            B: board_size
            S: max_seq_len
            C: num_champion
            match_team_ids: (N, [CLS] + B)
            match_ban_ids: (N, [CLS] + B)
            match_user_ids: (N, [CLS] + B)
            match_item_ids: (N, [CLS] + B)
            match_lane_ids: (N, [CLS] + B)
            
            match_win_labels: (N, [CLS] + B)
            match_item_labels: (N, [CLS] + B)
            match_user_labels: (N, [CLS] + B)
            """
            (match_team_ids, match_ban_ids, match_user_ids, match_item_ids, match_lane_ids) = match_x_batch
            (match_win_labels, match_item_labels, match_user_labels) = match_y_batch

            """
            user_ban_ids: (N, B, S, # of bans)
            user_item_ids: (N, B, S)
            user_lane_ids: (N, B, S)
            user_win_ids: (N, B, S)
            
            user_win_labels: (N, B, S)
            user_win_mask_labels: (N, B, S)
            user_item_labels: (N, B, S)
            """
            (user_ban_ids, user_item_ids, user_lane_ids, user_stat_ids, user_win_ids) = user_history_x_batch
            (user_win_labels, user_win_mask_labels, user_item_labels) = user_history_y_batch

            N, B, S = user_item_ids.shape
            C = self.num_items

            pi_pred, v_pred = None, None
            pi_true, v_true = None, None
            if self.args.op == 'train_interaction':
                pi_true, _ = torch.max(match_item_labels, 1)  # (N)
                pi_true = torch.eye(self.num_items)[pi_true].detach().cpu().numpy()  # (N, C)

                user_input_ids = torch.max(match_user_labels, 1)[0].repeat_interleave(C)  # (N*C)
                item_input_ids = torch.arange(C, device=self.device).repeat(N)  # (N*C)
                logits = self.model((user_input_ids, item_input_ids)).reshape(N, C)
                pi_pred = F.sigmoid(logits).detach().cpu().numpy()

            elif self.args.op == 'train_user_rec':
                # only test the user of current turn
                _, user_true_idx = torch.max(match_item_labels, 1)
                user_history_x_batch = [feature[torch.arange(N, device=self.device), user_true_idx - 1, :]
                                        for feature in user_history_x_batch]
                user_history_y_batch = [feature[torch.arange(N, device=self.device), user_true_idx - 1, :]
                                        for feature in user_history_y_batch]
                (user_ban_ids, user_item_ids, user_lane_ids, user_stat_ids, user_win_ids) = user_history_x_batch
                (user_win_labels, user_win_mask_labels, user_item_labels) = user_history_y_batch

                # get last-item of the user-history
                pi_true, pi_true_idx = torch.max(user_item_labels, 1)  # [N]
                pi_true = torch.eye(self.num_items)[pi_true].detach().cpu().numpy()  # [N, C]
                v_true = user_win_labels.cpu().numpy()
                v_true = v_true[:, -1]  # see only last item

                # policy: result of the current game should be masked.
                pi_pred, _ = self.model(user_history_x_batch)
                pi_pred = torch.exp(pi_pred)
                # value: current item should be given
                user_item_ids[torch.arange(N, device=self.device), -1] \
                    = user_item_labels[torch.arange(N, device=self.device), -1]
                user_history_x_batch = (user_ban_ids, user_item_ids, user_lane_ids, user_stat_ids, user_win_ids)
                _, v_pred = self.model(user_history_x_batch)
                v_pred = F.sigmoid(v_pred)

                # Inference the right sequence (last)
                pi_pred = pi_pred[torch.arange(N, device=self.device), -1, :].detach().cpu().numpy()
                v_pred = v_pred[torch.arange(N, device=self.device), -1, :].squeeze(-1).detach().cpu().numpy()

            elif self.args.op == 'train_draft_rec':
                pi_true, pi_true_idx = torch.max(match_item_labels, 1)  # [N]
                pi_true = torch.eye(self.num_items)[pi_true].detach().cpu().numpy()  # [N, C]
                v_true = match_win_labels.cpu().numpy()
                v_true = v_true[:, 0] # See only 0 since Win label is stored in position of CLS token

                pi_pred, _ = self.model(user_history_x_batch)
                pi_pred = torch.exp(pi_pred)
                # value: current item should be given
                user_history_x_batch[1][torch.arange(N, device=self.device), pi_true_idx-1, -1] = \
                    user_history_y_batch[2][torch.arange(N, device=self.device), pi_true_idx-1, -1]
                _, v_pred = self.model(user_history_x_batch)
                v_pred = F.sigmoid(v_pred)
                # Inference the right sequence  (pi: [MASK], v: [CLS])
                pi_pred = pi_pred[torch.arange(N, device=self.device), pi_true_idx, :].detach().cpu().numpy()  # [N, C]
                v_pred = v_pred[torch.arange(N, device=self.device), 0, :].squeeze(-1).detach().cpu().numpy()

            else:
                raise NotImplementedError

            for k in self.args.k_list:
                summary['HR@' + str(k)] += recall_at_k(pi_true, pi_pred, k) * N
                summary['NDCG@' + str(k)] += ndcg_at_k(pi_true, pi_pred, k) * N
            summary['MRR'] += average_precision_at_k(pi_true, pi_pred, k=C) * N

            if v_pred is not None:
                summary['ACC'] += np.sum((v_pred >= 0.5) == v_true)
                summary['MAE'] += np.sum(np.abs(v_pred - v_true))
                summary['MSE'] += np.sum(np.square(v_pred - v_true))

                last_match_idx = (match_item_labels[:, -1] != 0).detach().cpu().numpy()
                v_true = v_true[last_match_idx]
                v_pred = v_pred[last_match_idx]
                summary['LAST_ACC'] += np.sum((v_pred >= 0.5) == v_true) * (N / np.sum(last_match_idx))
                summary['LAST_MAE'] += np.sum(np.abs(v_pred - v_true)) * (N / np.sum(last_match_idx))
                summary['LAST_MSE'] += np.sum(np.square(v_pred - v_true)) * (N / np.sum(last_match_idx))

            total_eval_num += N

        for metric, value in summary.items():
            summary[metric] = (value / total_eval_num)

        return summary
