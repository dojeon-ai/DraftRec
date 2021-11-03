from .base import BaseTrainer
from ..common.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class POPTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, model):
        super().__init__(args, train_loader, val_loader, test_loader, model)
        
    @classmethod
    def code(cls):
        return 'pop'

    def calculate_loss(self, batch):
        loss, extra_info = None, {}
        return loss, extra_info

    def calculate_metrics(self, batch):
        #현재 match에 등장한거랑 ban한거 빼곤 인기순

        args = self.args
        num_champions = args.num_champions
        num_turns = args.num_turns
        batch_size = batch['user_ids'].shape[0]

        metrics = {}
        pi_true = batch['target_champion'].squeeze(1)
        pi_true = torch.eye(num_champions)[pi_true].detach().cpu().numpy()  # (N, C)

        pi_pred = self.model(batch, self.pop_dict)

        #Because we need this instance, this can't be done within the model.
        is_draft_finished = batch['is_draft_finished'].cpu().numpy().flatten()

        pi_pred = pi_pred[~is_draft_finished].cpu().numpy()
        pi_true = pi_true[~is_draft_finished]

        rec_metrics = get_recommendation_metrics_for_ks(pi_pred, pi_true, args.metric_ks)
        metrics.update(rec_metrics)

        return metrics

    def train(self):
        self.pop_dict = self.train_loader.dataset.pop_dict
        test_log_data = self.validate(mode='test')
        self.logger.log_test(test_log_data)
"""
    def calculate_metrics(self, batch):
        #현재 match에 등장한거랑 ban한거 빼곤 인기순

        args = self.args
        num_champions = args.num_champions
        num_turns = args.num_turns
        batch_size = batch['user_ids'].shape[0]

        metrics = {}
        pi_true = batch['target_champion'].squeeze(1)
        pi_true = torch.eye(num_champions)[pi_true].detach().cpu().numpy()  # (N, C)

        turn_idx = copy.deepcopy(batch['turn'])
        turn_idx[turn_idx > num_turns] = 1
        turn_idx = turn_idx - 1
        #Because we need this instance, this can't be done within the model.
        is_draft_finished = batch['is_draft_finished'].cpu().numpy().flatten()

        user_input_ids = torch.gather(batch['user_ids'], 1, turn_idx).repeat_interleave(num_champions)  # (N*C)
        champion_input_ids = torch.arange(num_champions, device=self.device).repeat(batch_size)  # (N*C)

        eval_batch = {}
        eval_batch['user_idx'] = user_input_ids
        eval_batch['champion_idx'] = champion_input_ids
        eval_batch['pop_dict'] = self.pop_dict


        pi_pred = self.model(eval_batch).reshape(batch_size, num_champions)
        #print(logits.shape)



        # Mask the banned champions
        pi_pred[torch.arange(batch_size).unsqueeze(1), batch['bans'].cpu().numpy()] = 1e-10
        
        pi_pred = pi_pred[~is_draft_finished].cpu().numpy()
        pi_true = pi_true[~is_draft_finished]

        rec_metrics = get_recommendation_metrics_for_ks(pi_pred, pi_true, args.metric_ks)
        metrics.update(rec_metrics)

        return metrics
"""