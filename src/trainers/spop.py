from .base import BaseTrainer
from ..common.metrics import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class SPOPTrainer(BaseTrainer):
    def __init__(self, args, train_loader, val_loader, test_loader, model):
        super().__init__(args, train_loader, val_loader, test_loader, model)
        
    @classmethod
    def code(cls):
        return 'spop'

    def calculate_loss(self, batch):
        loss, extra_info = None, {}
        return loss, extra_info

    def calculate_metrics(self, batch):
        #현재 match에 등장한거랑 ban한거 빼곤 인기순
        args = self.args
        metrics = {}
        pi = self.model(batch)
        pi = pi.cpu().numpy()
        
        pi_true = batch['target_champion'].cpu().numpy()
        pi_true = np.eye(args.num_champions)[pi_true].squeeze(1)
        
        is_draft_finished = batch['is_draft_finished'].cpu().numpy().flatten()
        
        pi = pi[~is_draft_finished]
        pi_true = pi_true[~is_draft_finished]
        rec_metrics = get_recommendation_metrics_for_ks(pi, pi_true, args.metric_ks)
        metrics.update(rec_metrics)
        
        return metrics

    def train(self):
        test_log_data = self.validate(mode='test')
        self.logger.log_test(test_log_data)
